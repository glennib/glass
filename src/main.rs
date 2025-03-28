use std::borrow::Borrow;
use std::fs;
use std::io;
use std::net::SocketAddr;
use std::path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::body::Body;
use axum::extract;
use axum::extract::Path;
use axum::http::StatusCode;
use axum::http::header;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use clap::Args;
use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use fast_image_resize::PixelType;
use fast_image_resize::ResizeOptions;
use fast_image_resize::Resizer;
use fast_image_resize::images::Image;
use image::DynamicImage;
use image::ImageError;
use image::ImageReader;
use ravif::EncodedImage;
use ravif::Encoder;
use ravif::Img;
use tokio::net::TcpListener;
use tower::limit::ConcurrencyLimitLayer;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::instrument;

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Command,
    #[clap(flatten)]
    config: Config,
}

#[derive(Args, Debug, Clone)]
struct Config {
    /// 1 <= quality <= 100
    #[clap(long, default_value = "90")]
    quality: f32,
    /// 1 <= speed <= 10
    #[clap(long, default_value = "4")]
    speed: u8,
    #[clap(long = "filter", default_value = "lanczos3")]
    filter_type: FilterType,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
enum FilterType {
    Bilinear,
    Box,
    CatmullRom,
    Gaussian,
    Hamming,
    Lanczos3,
    Mitchell,
}

impl From<FilterType> for fast_image_resize::FilterType {
    fn from(value: FilterType) -> Self {
        match value {
            FilterType::Bilinear => Self::Bilinear,
            FilterType::Box => Self::Box,
            FilterType::CatmullRom => Self::CatmullRom,
            FilterType::Gaussian => Self::Gaussian,
            FilterType::Hamming => Self::Hamming,
            FilterType::Lanczos3 => Self::Lanczos3,
            FilterType::Mitchell => Self::Mitchell,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start a server with an HTTP AVIF conversion endpoint
    ///
    /// The endpoint is at /images/resized/{width}/{height}/{image}
    Server(Server),
    /// Converts an image and writes to file
    Convert(Convert),
}

#[derive(Debug, Args)]
struct Server {
    /// Socket to bind TCP listener
    #[clap(long, default_value = "0.0.0.0:3000")]
    addr: SocketAddr,
    /// Directory of images where we look up {image}
    #[clap(long, default_value = "images")]
    images: PathBuf,
    /// Can maximally serve this many requests concurrently
    #[clap(long = "concurrency", default_value = "50")]
    concurrency_limit: usize,
}

#[derive(Debug, Args)]
struct Convert {
    /// Source image
    image: PathBuf,
    width: u32,
    height: u32,
    /// Target file
    output: PathBuf,
}

#[derive(Clone, Debug)]
struct State {
    config: Arc<Config>,
    images: Arc<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    debug!(?cli);
    match cli.command {
        Command::Server(Server {
            addr,
            images,
            concurrency_limit,
        }) => {
            assert!(images.is_dir());
            let config = cli.config;
            let state = State {
                images: Arc::new(images),
                config: Arc::new(config),
            };
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(async {
                    let app = Router::new()
                        .route("/images/resized/{width}/{height}/{image}", get(h_resize))
                        .layer(ConcurrencyLimitLayer::new(concurrency_limit))
                        .with_state(state);
                    let listener = TcpListener::bind(&addr).await?;
                    info!(?addr, "serving");
                    axum::serve(listener, app).await
                })?;
        }
        Command::Convert(Convert {
            image,
            width,
            height,
            output,
        }) => {
            let image = resize(cli.config, &image, width, height)?;
            let begin = Instant::now();
            fs::write(&output, image.avif_file)?;
            let elapsed = begin.elapsed();
            info!(elapsed_secs = elapsed.as_secs_f64(), output=%output.display(), "wrote");
        }
    }

    Ok(())
}

async fn h_resize(
    extract::State(state): extract::State<State>,
    Path((target_width, target_height, image)): Path<(u32, u32, String)>,
) -> Result<Avif, Error> {
    let image = state.images.join(&image);
    Ok(Avif(
        tokio::task::spawn_blocking(move || {
            resize(state.config, &image, target_width, target_height)
        })
        .await
        .unwrap()?,
    ))
}

#[instrument(skip(config))]
fn resize(
    config: impl Borrow<Config> + 'static,
    image: &path::Path,
    target_width: u32,
    target_height: u32,
) -> Result<EncodedImage, Error> {
    info!("resizing");
    let config = config.borrow();
    let begin = Instant::now();
    let original = ImageReader::open(image)?.decode()?;
    let original = original.into_rgba8();
    let original = DynamicImage::from(original);
    let mut resized = Image::new(target_width, target_height, PixelType::U8x4);
    let mut resizer = Resizer::new();
    resizer
        .resize(
            &original,
            &mut resized,
            &ResizeOptions::new().resize_alg(fast_image_resize::ResizeAlg::Convolution(
                config.filter_type.into(),
            )),
        )
        .unwrap();
    let rgba: &[rgb::Rgba<u8>] = rgb::bytemuck::cast_slice(resized.buffer());
    let img = Img::new(
        rgba,
        usize::try_from(resized.width()).unwrap(),
        usize::try_from(resized.height()).unwrap(),
    );

    let res = Encoder::new()
        .with_quality(config.quality)
        .with_speed(config.speed)
        .encode_rgba(img)
        .unwrap();
    let elapsed = begin.elapsed();
    info!(elapsed_secs = elapsed.as_secs_f64(), "done");
    Ok(res)
}

struct Avif(EncodedImage);

impl IntoResponse for Avif {
    fn into_response(self) -> Response {
        axum::response::Response::builder()
            .header(header::CONTENT_TYPE, "image/avif")
            .body(Body::from(self.0.avif_file))
            .unwrap()
    }
}

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("image not found")]
    NotFound,
    #[error("failed to resize: {message}")]
    FailedToResize { message: String },
}

impl From<io::Error> for Error {
    fn from(_error: io::Error) -> Self {
        Self::NotFound
    }
}

impl From<ImageError> for Error {
    fn from(value: ImageError) -> Self {
        Self::FailedToResize {
            message: value.to_string(),
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        error!(error=%self, error_dbg=?self);
        match self {
            Error::NotFound => (StatusCode::NOT_FOUND, "not found").into_response(),
            Error::FailedToResize { message } => {
                (StatusCode::INTERNAL_SERVER_ERROR, message).into_response()
            }
        }
    }
}

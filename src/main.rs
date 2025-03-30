use std::borrow::Borrow;
use std::fs;
use std::io;
use std::net::SocketAddr;
use std::path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::bail;
use clap::Args;
use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use fast_image_resize::PixelType;
use fast_image_resize::ResizeOptions;
use fast_image_resize::Resizer;
use fast_image_resize::images::Image;
use image::DynamicImage;
use image::ExtendedColorType;
use image::ImageEncoder;
use image::ImageError;
use image::ImageReader;
use image::RgbaImage;
use image::codecs::avif;
use tokio::net::TcpListener;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::instrument;

use crate::server::router;

mod server;

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
    #[clap(long)]
    width: Option<u32>,
    #[clap(long)]
    height: Option<u32>,
    #[clap(long)]
    scale: Option<f64>,
    /// Target file
    output: PathBuf,
}

#[derive(Clone, Debug, Copy)]
enum ResizeTo {
    Width(u32),
    Height(u32),
    WidthAndHeight(u32, u32),
    Scale(f64),
}

#[derive(Debug, Clone)]
struct Encoded {
    name: Option<String>,
    bytes: Vec<u8>,
    encoding: Encoding,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Encoding {
    Avif,
    Jpeg,
}

impl Encoded {
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }
}

impl Encoding {
    fn mime(self) -> &'static str {
        match self {
            Encoding::Avif => "image/avif",
            Encoding::Jpeg => "image/jpeg",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Encoding::Avif => "avif",
            Encoding::Jpeg => "jpg",
        }
    }
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
            let state = server::State {
                images: Arc::new(images),
                config: Arc::new(config),
            };
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(async {
                    let router = router(concurrency_limit, state);
                    let listener = TcpListener::bind(&addr).await?;
                    info!(?addr, "serving");
                    axum::serve(listener, router).await
                })?;
        }
        Command::Convert(Convert {
            image,
            width,
            height,
            scale,
            output,
        }) => {
            let to = match (width, height, scale) {
                (Some(width), Some(height), None) => ResizeTo::WidthAndHeight(width, height),
                (Some(width), None, None) => ResizeTo::Height(width),
                (None, Some(height), None) => ResizeTo::Width(height),
                (None, None, Some(scale)) => ResizeTo::Scale(scale),
                _ => {
                    bail!("provide one or both of width and height, or only scale");
                }
            };
            let image = load_resize_encode(cli.config, &image, Encoding::Avif, to)?;
            let begin = Instant::now();
            fs::write(&output, image.bytes)?;
            let elapsed = begin.elapsed();
            debug!(elapsed_secs = elapsed.as_secs_f64(), output=%output.display(), "wrote");
        }
    }

    Ok(())
}

#[instrument(skip_all)]
fn load(image: &path::Path) -> Result<RgbaImage, Error> {
    let begin = Instant::now();
    let original = ImageReader::open(image)?.decode()?;
    let rgba8 = original.to_rgba8();
    let elapsed = begin.elapsed();
    debug!(elapsed_secs = elapsed.as_secs_f64(), "loaded image");
    Ok(rgba8)
}

fn aspect_ratio(width: u32, height: u32) -> f64 {
    f64::from(width) / f64::from(height)
}

#[instrument(skip_all)]
fn resize(
    original: impl Into<DynamicImage>,
    to: ResizeTo,
    filter_type: FilterType,
) -> Result<Image<'static>, Error> {
    let begin = Instant::now();
    let original = original.into();
    let (width, height) = match to {
        ResizeTo::Width(width) => {
            let ar = aspect_ratio(original.width(), original.height());
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let height = (f64::from(width) / ar).round() as u32;
            (width, height)
        }
        ResizeTo::Height(height) => {
            let ar = aspect_ratio(original.width(), original.height());
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let width = (f64::from(height) * ar).round() as u32;
            (width, height)
        }
        ResizeTo::WidthAndHeight(width, height) => (width, height),
        ResizeTo::Scale(scale) => {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let width = (f64::from(original.width()) * scale).round() as u32;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let height = (f64::from(original.height()) * scale).round() as u32;
            (width, height)
        }
    };
    debug!(width, height);
    let mut resized = Image::new(width, height, PixelType::U8x4);
    let mut resizer = Resizer::new();
    resizer
        .resize(
            &original,
            &mut resized,
            &ResizeOptions::new().resize_alg(fast_image_resize::ResizeAlg::Convolution(
                filter_type.into(),
            )),
        )
        .map_err(|error| Error::FailedToResize {
            message: error.to_string(),
        })?;
    let elapsed = begin.elapsed();
    #[allow(clippy::cast_precision_loss)]
    let kilobytes = resized.buffer().len() as f64 / 1024.0;
    debug!(elapsed_secs = elapsed.as_secs_f64(), kilobytes, "resized");
    Ok(resized)
}

#[instrument(skip_all)]
fn encode(image: Image, _encoding: Encoding, quality: f32, speed: u8) -> Result<Encoded, Error> {
    let begin = Instant::now();

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let quality = quality.round() as u8;

    let mut encoded = Vec::new();
    let encoder = avif::AvifEncoder::new_with_speed_quality(&mut encoded, speed, quality);
    encoder.write_image(
        image.buffer(),
        image.width(),
        image.height(),
        ExtendedColorType::Rgba8,
    )?;
    let bytes = encoded.len();
    #[allow(clippy::cast_precision_loss)]
    let kilobytes = bytes as f64 / 1024.0;
    let elapsed = begin.elapsed();
    debug!(
        elapsed_secs = elapsed.as_secs_f64(),
        kilobytes, "encoded image"
    );
    Ok(Encoded {
        name: None,
        encoding: Encoding::Avif,
        bytes: encoded,
    })
}

#[instrument(skip(config))]
fn load_resize_encode(
    config: impl Borrow<Config> + 'static,
    image: &path::Path,
    encoding: Encoding,
    to: ResizeTo,
) -> Result<Encoded, Error> {
    let config = config.borrow();
    let begin = Instant::now();
    let original = load(image)?;
    let resized = resize(original, to, config.filter_type)?;
    let mut encoded = encode(resized, encoding, config.quality, config.speed)?;
    encoded.name = image
        .file_stem()
        .map(|name| name.to_str().unwrap().to_string());
    let elapsed = begin.elapsed();
    debug!(elapsed_secs = elapsed.as_secs_f64(), "done");
    Ok(encoded)
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

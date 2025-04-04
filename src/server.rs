use std::borrow::Borrow;
use std::path;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract;
use axum::http::StatusCode;
use axum::http::header;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use tokio::task::spawn_blocking;
use tower::limit::ConcurrencyLimitLayer;
use tracing::error;

use crate::Config;
use crate::Encoded;
use crate::Encoding;
use crate::Error;
use crate::ResizeTo;
use crate::load_resize_encode;

#[derive(Clone, Debug)]
pub struct State {
    pub config: Arc<Config>,
    pub images: Arc<PathBuf>,
}

impl IntoResponse for Encoded {
    fn into_response(self) -> Response {
        let name = self.name.as_deref().unwrap_or("image");
        axum::response::Response::builder()
            .header(header::CONTENT_TYPE, self.encoding().mime())
            .header(
                header::CONTENT_DISPOSITION,
                format!("inline; filename=\"{name}.{}\"", self.encoding.extension()),
            )
            .body(Body::from(self.bytes))
            .unwrap()
    }
}

impl FromStr for Encoding {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let enc = match s {
            "avif" => Self::Avif,
            "jpeg" | "jpg" => Self::Jpeg,
            _ => return Err(()),
        };
        Ok(enc)
    }
}

pub fn router(concurrency_limit: usize, state: State) -> Router {
    Router::new()
        .route(
            "/images/{image}/size/{width}/{height}/encoding/{encoding}",
            get(h_wh),
        )
        .route(
            "/images/{image}/size/width/{width}/encoding/{encoding}",
            get(h_w),
        )
        .route(
            "/images/{image}/size/height/{width}/encoding/{encoding}",
            get(h_h),
        )
        .route(
            "/images/{image}/size/scale/{scale}/encoding/{encoding}",
            get(h_s),
        )
        .layer(ConcurrencyLimitLayer::new(concurrency_limit))
        .with_state(state)
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

async fn h_wh(
    extract::State(state): extract::State<State>,
    extract::Path((image, width, height, encoding)): extract::Path<(String, u32, u32, String)>,
) -> Result<Encoded, Error> {
    let image = state.images.join(&image);
    load_resize_encode_async(
        state.config,
        image,
        encoding.parse().unwrap(),
        ResizeTo::WidthAndHeight(width, height),
    )
    .await
}

async fn h_w(
    extract::State(state): extract::State<State>,
    extract::Path((image, width, encoding)): extract::Path<(String, u32, String)>,
) -> Result<Encoded, Error> {
    let image = state.images.join(&image);
    load_resize_encode_async(
        state.config,
        image,
        encoding.parse().unwrap(),
        ResizeTo::Width(width),
    )
    .await
}

async fn h_h(
    extract::State(state): extract::State<State>,
    extract::Path((image, height, encoding)): extract::Path<(String, u32, String)>,
) -> Result<Encoded, Error> {
    let image = state.images.join(&image);
    load_resize_encode_async(
        state.config,
        image,
        encoding.parse().unwrap(),
        ResizeTo::Height(height),
    )
    .await
}

async fn h_s(
    extract::State(state): extract::State<State>,
    extract::Path((image, scale, encoding)): extract::Path<(String, f64, String)>,
) -> Result<Encoded, Error> {
    let image = state.images.join(&image);
    load_resize_encode_async(
        state.config,
        image,
        encoding.parse().unwrap(),
        ResizeTo::Scale(scale),
    )
    .await
}

async fn load_resize_encode_async(
    config: impl Borrow<Config> + Send + 'static,
    image: impl AsRef<path::Path> + Send + 'static,
    encoding: Encoding,
    to: ResizeTo,
) -> Result<Encoded, Error> {
    let image = spawn_blocking(move || load_resize_encode(config, image.as_ref(), encoding, to))
        .await
        .unwrap()?;
    Ok(image)
}

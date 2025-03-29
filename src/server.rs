use std::borrow::Borrow;
use std::path;
use std::path::PathBuf;
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
use crate::Error;
use crate::ResizeTo;
use crate::load_resize_encode;

#[derive(Clone, Debug)]
pub struct State {
    pub config: Arc<Config>,
    pub images: Arc<PathBuf>,
}

struct Avif(Vec<u8>);

impl IntoResponse for Avif {
    fn into_response(self) -> Response {
        axum::response::Response::builder()
            .header(header::CONTENT_TYPE, "image/avif")
            .body(Body::from(self.0))
            .unwrap()
    }
}

pub fn router(concurrency_limit: usize, state: State) -> Router {
    Router::new()
        .route("/images/resized/{width}/{height}/{image}", get(h_wh))
        .route("/images/resized/width/{width}/{image}", get(h_w))
        .route("/images/resized/height/{width}/{image}", get(h_h))
        .route("/images/resized/scale/{scale}/{image}", get(h_s))
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
    extract::Path((width, height, image)): extract::Path<(u32, u32, String)>,
) -> Result<Avif, Error> {
    let image = state.images.join(&image);
    load_resize_encode_async(state.config, image, ResizeTo::WidthAndHeight(width, height)).await
}

async fn h_w(
    extract::State(state): extract::State<State>,
    extract::Path((width, image)): extract::Path<(u32, String)>,
) -> Result<Avif, Error> {
    let image = state.images.join(&image);
    load_resize_encode_async(state.config, image, ResizeTo::Width(width)).await
}

async fn h_h(
    extract::State(state): extract::State<State>,
    extract::Path((height, image)): extract::Path<(u32, String)>,
) -> Result<Avif, Error> {
    let image = state.images.join(&image);
    load_resize_encode_async(state.config, image, ResizeTo::Height(height)).await
}

async fn h_s(
    extract::State(state): extract::State<State>,
    extract::Path((scale, image)): extract::Path<(f64, String)>,
) -> Result<Avif, Error> {
    let image = state.images.join(&image);
    load_resize_encode_async(state.config, image, ResizeTo::Scale(scale)).await
}

async fn load_resize_encode_async(
    config: impl Borrow<Config> + Send + 'static,
    image: impl AsRef<path::Path> + Send + 'static,
    to: ResizeTo,
) -> Result<Avif, Error> {
    let image = spawn_blocking(move || load_resize_encode(config, image.as_ref(), to))
        .await
        .unwrap()?;
    Ok(Avif(image))
}

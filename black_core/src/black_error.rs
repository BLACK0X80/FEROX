use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum BlackError {
    #[error("shape error: {black_msg}")]
    BlackShapeError { black_msg: String },

    #[error("dtype error: {black_msg}")]
    BlackDTypeError { black_msg: String },

    #[error("device error: {black_msg}")]
    BlackDeviceError { black_msg: String },

    #[error("memory error: {black_msg}")]
    BlackMemoryError { black_msg: String },

    #[error("grad error: {black_msg}")]
    BlackGradError { black_msg: String },

    #[error("ops error: {black_msg}")]
    BlackOpsError { black_msg: String },

    #[error("io error: {black_msg}")]
    BlackIOError { black_msg: String },

    #[error("index error: {black_msg}")]
    BlackIndexError { black_msg: String },

    #[error("internal error: {black_msg}")]
    BlackInternalError { black_msg: String },
}

pub type BlackResult<T> = Result<T, BlackError>;

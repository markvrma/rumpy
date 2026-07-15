//! rumpy — a bare-bones, pure-Rust NumPy.
//!
//! Learning project: every `todo!()` body is yours to write. The tests are the
//! spec — `cargo test` starts red and goes green milestone by milestone.
//! See README.md for the roadmap and reading list.

/// Error for any shape-level failure. All fallible ops return
/// `Result<_, ShapeError>` — no panicking APIs in the library itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// Total element count doesn't match (e.g. reshape 2x3 into 4x2,
    /// or from_vec with the wrong-length buffer).
    SizeMismatch { expected: usize, got: usize },
    /// Two shapes can't be combined (elementwise / broadcast / matmul).
    Incompatible { a: Vec<usize>, b: Vec<usize> },
    /// Axis argument >= number of dimensions.
    AxisOutOfBounds { axis: usize, ndim: usize },
}

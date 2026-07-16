//! rumpy — a bare-bones, pure-Rust NumPy.
//!
//! Learning project: every `todo!()` body is yours to write. The tests are the
//! spec — `cargo test` starts red and goes green milestone by milestone.
//! See README.md for the roadmap and reading list.

use std::fmt;

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

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::SizeMismatch { expected, got } => {
                write!(f, "size mismatch: expected {expected} elements, got {got}")
            }
            ShapeError::Incompatible { a, b } => {
                write!(f, "incompatible shapes: {a:?} vs {b:?}")
            }
            ShapeError::AxisOutOfBounds { axis, ndim } => {
                write!(f, "axis {axis} out of bounds for {ndim}-d array")
            }
        }
    }
}

impl std::error::Error for ShapeError {}

/// Test helper: panic unless `a` and `b` are within `eps` of each other.
/// Floats accumulate rounding, so tests never use exact `assert_eq!` on
/// computed values — this is the whole float-comparison story, no crate needed.
pub fn assert_close(a: f64, b: f64, eps: f64) {
    assert!(
        (a - b).abs() <= eps,
        "not close: {a} vs {b} (eps {eps})"
    );
}

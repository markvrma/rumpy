//! The core n-dimensional array: one flat buffer + shape + strides.
//!
//! Read before implementing (M1):
//! - NumPy internals (memory layout, why strides exist):
//!   https://numpy.org/doc/stable/dev/internals.html
//! - ndarray docs (shape/strides/order):
//!   https://numpy.org/doc/stable/reference/arrays.ndarray.html
//!
//! Conventions (documented decisions, see README):
//! - `f64` only. C-order (row-major) only. Strides count *elements*, not bytes
//!   (NumPy counts bytes — same idea, simpler arithmetic).
//! - 0-d arrays (scalars) are out of scope.

/// N-dimensional array of `f64` owning a flat, contiguous, C-order buffer.
///
/// Invariant every constructor must uphold:
/// `data.len() == shape.iter().product()` and `strides == Self::strides_for(&shape)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    pub(crate) data: Vec<f64>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl Array {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

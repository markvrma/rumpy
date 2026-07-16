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
    // ---------------------------------------------------------------- M1: core

    /// C-order strides for a shape: how many *elements* one step along each
    /// axis skips in the flat buffer. E.g. shape `[2, 3, 4]` → `[12, 4, 1]`.
    ///
    /// This one function is the heart of the whole library — get it right and
    /// indexing, transpose and broadcasting all fall out of it.
    pub(crate) fn strides_for(shape: &[usize]) -> Vec<usize> {
        // walk right to left, each stride is the product of everything to its
        // right. last axis is always 1 because neighbours there are adjacent.
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

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

// ---------------------------------------------------------------------- tests
// Spec tests: expected values are hand-computed. They fail on todo!() and go
// green as you implement. Do not edit the expected values — edit your code.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_close;

    const EPS: f64 = 1e-12;

    // ------------------------------------------------------------------- M1

    #[test]
    fn strides_row_major() {
        assert_eq!(Array::strides_for(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(Array::strides_for(&[5]), vec![1]);
        assert_eq!(Array::strides_for(&[4, 7]), vec![7, 1]);
    }
}

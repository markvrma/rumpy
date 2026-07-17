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

    /// Array of the given shape, all elements `0.0`.
    pub fn zeros(shape: &[usize]) -> Array {
        Array {
            data: vec![0.0; shape.iter().product()],
            shape: shape.to_vec(),
            strides: Self::strides_for(shape),
        }
    }

    /// Array of the given shape, all elements `1.0`.
    pub fn ones(shape: &[usize]) -> Array {
        Array {
            data: vec![1.0; shape.iter().product()],
            shape: shape.to_vec(),
            strides: Self::strides_for(shape),
        }
    }

    /// 1-d array: `start, start+step, ...` up to but excluding `stop`.
    /// Mirrors `numpy.arange`. Contract is "close enough": exact for
    /// integer-valued steps; float steps accumulate rounding just like NumPy's.
    pub fn arange(start: f64, stop: f64, step: f64) -> Array {
        // same count formula numpy uses. step 0 or a backwards range gives a
        // non-finite / negative count, which we clamp to an empty array.
        let count = ((stop - start) / step).ceil();
        let n = if count.is_finite() && count > 0.0 {
            count as usize
        } else {
            0
        };
        // start + i*step, not a running += , so the error doesn't accumulate
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        Array {
            data,
            shape: vec![n],
            strides: vec![1],
        }
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

    /// Flat-buffer offset of a multi-index, using strides:
    /// `offset = sum(index[i] * strides[i])`.
    /// `None` if `index.len() != ndim` or any coordinate is out of bounds.
    pub(crate) fn offset(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.shape.len() {
            return None;
        }
        let mut off = 0;
        for (axis, &i) in index.iter().enumerate() {
            if i >= self.shape[axis] {
                return None;
            }
            off += i * self.strides[axis];
        }
        Some(off)
    }

    /// Element at a multi-index, `None` when out of bounds — the fallible,
    /// `Result`-friendly access path (Rust's `Index` trait can't be fallible,
    /// so we don't use it; see README decisions table).
    pub fn get(&self, index: &[usize]) -> Option<f64> {
        self.offset(index).map(|o| self.data[o])
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

    #[test]
    fn zeros_shape_and_values() {
        let a = Array::zeros(&[2, 3]);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.len(), 6);
        assert_eq!(a.get(&[1, 2]), Some(0.0));
    }

    #[test]
    fn ones_values() {
        let a = Array::ones(&[3]);
        assert_eq!(a.get(&[0]), Some(1.0));
        assert_eq!(a.get(&[2]), Some(1.0));
    }

    #[test]
    fn arange_integer_step_is_exact() {
        let a = Array::arange(0.0, 5.0, 1.0);
        assert_eq!(a.shape(), &[5]);
        for i in 0..5 {
            assert_eq!(a.get(&[i]), Some(i as f64));
        }
    }

    #[test]
    fn arange_float_step_close_enough() {
        let a = Array::arange(0.0, 1.0, 0.25);
        assert_eq!(a.len(), 4);
        assert_close(a.get(&[3]).unwrap(), 0.75, EPS);
    }

    #[test]
    fn get_out_of_bounds_and_wrong_ndim() {
        let a = Array::zeros(&[2, 3]);
        assert_eq!(a.get(&[2, 0]), None); // row out of range
        assert_eq!(a.get(&[0, 3]), None); // col out of range
        assert_eq!(a.get(&[0]), None); // wrong ndim
    }
}

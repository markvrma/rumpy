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

use crate::ShapeError;

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

    /// Take ownership of a flat buffer and give it a shape.
    /// Errors with `SizeMismatch` if the element counts disagree.
    pub fn from_vec(data: Vec<f64>, shape: &[usize]) -> Result<Array, ShapeError> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(ShapeError::SizeMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Array {
            data,
            strides: Self::strides_for(shape),
            shape: shape.to_vec(),
        })
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

    /// odometer step: bump a C-order multi-index by one, rightmost axis first.
    /// returns false once it rolls over, which is how the walk loops know to stop.
    pub(crate) fn next_index(index: &mut [usize], shape: &[usize]) -> bool {
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                return true;
            }
            index[axis] = 0;
        }
        false
    }

    /// Element at a multi-index, `None` when out of bounds — the fallible,
    /// `Result`-friendly access path (Rust's `Index` trait can't be fallible,
    /// so we don't use it; see README decisions table).
    pub fn get(&self, index: &[usize]) -> Option<f64> {
        self.offset(index).map(|o| self.data[o])
    }

    /// Mutable reference to an element. `set` for free:
    /// `*a.get_mut(&[i, j]).unwrap() = 5.0;`
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut f64> {
        // offset borrows self immutably, so it has to finish before we reborrow
        let off = self.offset(index)?;
        Some(&mut self.data[off])
    }

    // ------------------------------------------------------- M2: shape changes

    /// Same data, new shape. Errors with `SizeMismatch` when counts differ.
    ///
    /// ponytail: copy-based (clone the buffer). NumPy reshapes contiguous
    /// arrays without copying — no-copy views are the named stretch goal, and
    /// they'd force every op after this to handle non-contiguous layouts.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Array, ShapeError> {
        // the buffer is already C-order, so a reshape is literally just new
        // shape + new strides over the same values. check before cloning so a
        // bad call doesn't pay for a copy it's going to throw away.
        let expected: usize = new_shape.iter().product();
        if expected != self.data.len() {
            return Err(ShapeError::SizeMismatch {
                expected,
                got: self.data.len(),
            });
        }
        Array::from_vec(self.data.clone(), new_shape)
    }

    /// Reverse the axes (2-d: rows become columns). Copy-based: builds a fresh
    /// C-order buffer in transposed order. NumPy instead just swaps shape and
    /// strides — zero copy — which is a great README design-note contrast.
    pub fn transpose(&self) -> Array {
        let mut out_shape = self.shape.clone();
        out_shape.reverse();
        let mut out = Array::zeros(&out_shape);
        if self.data.is_empty() {
            return out;
        }
        // walk the output in C-order (so slot just counts up) and for each
        // spot read the source at the reversed index. that's the whole trick.
        let mut idx = vec![0usize; out_shape.len()];
        let mut src = vec![0usize; out_shape.len()];
        for slot in 0..out.data.len() {
            for (axis, &i) in idx.iter().enumerate() {
                src[out_shape.len() - 1 - axis] = i;
            }
            out.data[slot] = self.data[self.offset(&src).unwrap()];
            Array::next_index(&mut idx, &out_shape);
        }
        out
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
    fn from_vec_ok() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(a.shape(), &[2, 3]);
        // C-order: row 0 is [1, 2, 3], row 1 is [4, 5, 6]
        assert_eq!(a.get(&[0, 2]), Some(3.0));
        assert_eq!(a.get(&[1, 0]), Some(4.0));
    }

    #[test]
    fn from_vec_size_mismatch() {
        let err = Array::from_vec(vec![1.0, 2.0, 3.0], &[2, 2]).unwrap_err();
        assert_eq!(err, crate::ShapeError::SizeMismatch { expected: 4, got: 3 });
    }

    #[test]
    fn get_out_of_bounds_and_wrong_ndim() {
        let a = Array::zeros(&[2, 3]);
        assert_eq!(a.get(&[2, 0]), None); // row out of range
        assert_eq!(a.get(&[0, 3]), None); // col out of range
        assert_eq!(a.get(&[0]), None); // wrong ndim
    }

    #[test]
    fn get_mut_writes_through() {
        let mut a = Array::zeros(&[2, 2]);
        *a.get_mut(&[1, 1]).unwrap() = 7.5;
        assert_eq!(a.get(&[1, 1]), Some(7.5));
        assert_eq!(a.get(&[0, 0]), Some(0.0));
    }

    // ------------------------------------------------------------------- M2

    #[test]
    fn reshape_preserves_c_order() {
        let a = Array::arange(1.0, 7.0, 1.0); // [1, 2, 3, 4, 5, 6]
        let b = a.reshape(&[2, 3]).unwrap();
        assert_eq!(b.get(&[0, 0]), Some(1.0));
        assert_eq!(b.get(&[1, 2]), Some(6.0));
        let c = b.reshape(&[3, 2]).unwrap();
        assert_eq!(c.get(&[2, 1]), Some(6.0));
    }

    #[test]
    fn reshape_size_mismatch() {
        let a = Array::zeros(&[2, 3]);
        let err = a.reshape(&[4, 2]).unwrap_err();
        assert_eq!(err, crate::ShapeError::SizeMismatch { expected: 8, got: 6 });
    }

    #[test]
    fn transpose_2d() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let t = a.transpose();
        assert_eq!(t.shape(), &[3, 2]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(t.get(&[j, i]), a.get(&[i, j]));
            }
        }
    }

    #[test]
    fn transpose_twice_is_identity() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(a.transpose().transpose(), a);
    }
}

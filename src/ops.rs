//! Elementwise ops (M3), broadcasting (M4), reductions (M5).
//!
//! Read before implementing:
//! - Broadcasting rules (M4): https://numpy.org/doc/stable/user/basics.broadcasting.html
//! - What a ufunc is (M3 context): https://numpy.org/doc/stable/reference/ufuncs.html

use crate::{Array, ShapeError};

/// m3 version: shapes have to match exactly. broadcasting is m4's problem.
fn zip_same(a: &Array, b: &Array, f: impl Fn(f64, f64) -> f64) -> Result<Array, ShapeError> {
    if a.shape != b.shape {
        return Err(ShapeError::Incompatible {
            a: a.shape.clone(),
            b: b.shape.clone(),
        });
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(&x, &y)| f(x, y))
        .collect();
    Array::from_vec(data, &a.shape)
}

// ------------------------------------------------- M3 + M4: elementwise ops

impl Array {
    /// Elementwise addition with broadcasting.
    ///
    /// Do this in two passes (the rewrite is deliberate pedagogy):
    /// - M3: support equal shapes only; return `Incompatible` otherwise.
    /// - M4: generalize — compute `broadcast_shape`, walk the output's
    ///   multi-indices, and read each operand through its `broadcast_strides`.
    ///   No materialized copies.
    pub fn add(&self, rhs: &Array) -> Result<Array, ShapeError> {
        zip_same(self, rhs, |x, y| x + y)
    }

    /// Elementwise subtraction with broadcasting. Same plan as `add`.
    pub fn sub(&self, rhs: &Array) -> Result<Array, ShapeError> {
        zip_same(self, rhs, |x, y| x - y)
    }

    /// Elementwise multiplication with broadcasting. Same plan as `add`.
    pub fn mul(&self, rhs: &Array) -> Result<Array, ShapeError> {
        zip_same(self, rhs, |x, y| x * y)
    }

    /// Elementwise division with broadcasting. Same plan as `add`.
    /// Division by zero follows IEEE 754 (inf / NaN), same as NumPy.
    pub fn div(&self, rhs: &Array) -> Result<Array, ShapeError> {
        // no zero check: ieee754 already gives inf/nan, same as numpy
        zip_same(self, rhs, |x, y| x / y)
    }
}

// ---------------------------------------------------------------------- tests
// Spec tests: fail on todo!(), go green per milestone. M4 tests stay red while
// add/sub/mul/div are still equal-shape-only — that's the expected order.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_close;

    const EPS: f64 = 1e-12;

    fn arr(data: &[f64], shape: &[usize]) -> Array {
        Array::from_vec(data.to_vec(), shape).unwrap()
    }

    // ------------------------------------------------------------------- M3

    #[test]
    fn add_equal_shapes() {
        let a = arr(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = arr(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.get(&[0, 0]), Some(11.0));
        assert_eq!(c.get(&[1, 1]), Some(44.0));
    }

    #[test]
    fn sub_mul_div_equal_shapes() {
        let a = arr(&[8.0, 6.0], &[2]);
        let b = arr(&[2.0, 3.0], &[2]);
        assert_eq!(a.sub(&b).unwrap().get(&[0]), Some(6.0));
        assert_eq!(a.mul(&b).unwrap().get(&[1]), Some(18.0));
        assert_eq!(a.div(&b).unwrap().get(&[1]), Some(2.0));
    }

    #[test]
    fn mismatched_shapes_error_without_broadcast() {
        // [2,3] vs [4] is incompatible even under full broadcasting (M4).
        let a = Array::zeros(&[2, 3]);
        let b = Array::zeros(&[4]);
        assert!(matches!(
            a.add(&b),
            Err(ShapeError::Incompatible { .. })
        ));
    }
}

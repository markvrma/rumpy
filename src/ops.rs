//! Elementwise ops (M3), broadcasting (M4), reductions (M5).
//!
//! Read before implementing:
//! - Broadcasting rules (M4): https://numpy.org/doc/stable/user/basics.broadcasting.html
//! - What a ufunc is (M3 context): https://numpy.org/doc/stable/reference/ufuncs.html

use crate::{Array, ShapeError};
use std::ops::{Add, Div, Mul, Sub};

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

    /// Add a scalar to every element. Infallible — no shapes to disagree.
    pub fn add_scalar(&self, s: f64) -> Array {
        self.map(|x| x + s)
    }

    /// Multiply every element by a scalar.
    pub fn mul_scalar(&self, s: f64) -> Array {
        self.map(|x| x * s)
    }

    /// Apply `f` to every element — the poor man's ufunc.
    /// `a.map(f64::sqrt)` is rumpy's `np.sqrt(a)`.
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Array {
        // shape is untouched, so we can hand the strides straight over
        Array {
            data: self.data.iter().map(|&x| f(x)).collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

// Operator sugar so `&a + &b` reads like NumPy. The library API is
// Result-everywhere; only these operators unwrap, mirroring how `v[i]`
// panics while `v.get(i)` doesn't. Borrowed impls only — owned variants
// are mechanical additions if call sites ever want them.

impl Add for &Array {
    type Output = Array;
    fn add(self, rhs: &Array) -> Array {
        Array::add(self, rhs).expect("shape mismatch in +")
    }
}

impl Sub for &Array {
    type Output = Array;
    fn sub(self, rhs: &Array) -> Array {
        Array::sub(self, rhs).expect("shape mismatch in -")
    }
}

impl Mul for &Array {
    type Output = Array;
    fn mul(self, rhs: &Array) -> Array {
        Array::mul(self, rhs).expect("shape mismatch in *")
    }
}

impl Div for &Array {
    type Output = Array;
    fn div(self, rhs: &Array) -> Array {
        Array::div(self, rhs).expect("shape mismatch in /")
    }
}

// ------------------------------------------------------------- M5: reductions

impl Array {
    /// Sum of all elements. Empty array sums to 0.0, like NumPy.
    pub fn sum(&self) -> f64 {
        // plain left-to-right sum. numpy pairwise-sums to keep error down on
        // big arrays; noted as a known difference, not fixed here.
        self.data.iter().sum()
    }

    /// Mean of all elements. Empty array → NaN (NumPy warns and returns NaN).
    pub fn mean(&self) -> f64 {
        // empty falls out on its own: 0.0 / 0.0 is nan, which is what we want
        self.sum() / self.data.len() as f64
    }

    /// Smallest element, `None` when empty.
    /// NaN handling: NumPy's `min` propagates NaN; document what yours does.
    pub fn min(&self) -> Option<f64> {
        self.data.iter().copied().reduce(f64::min)
    }

    /// Largest element, `None` when empty.
    pub fn max(&self) -> Option<f64> {
        self.data.iter().copied().reduce(f64::max)
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
    fn operator_sugar() {
        let a = arr(&[1.0, 2.0], &[2]);
        let b = arr(&[3.0, 4.0], &[2]);
        let c = &a + &b;
        assert_eq!(c.get(&[1]), Some(6.0));
        let d = &c * &a;
        assert_eq!(d.get(&[1]), Some(12.0));
    }

    #[test]
    fn scalar_ops() {
        let a = arr(&[1.0, 2.0, 3.0], &[3]);
        assert_eq!(a.add_scalar(10.0).get(&[2]), Some(13.0));
        assert_eq!(a.mul_scalar(-2.0).get(&[0]), Some(-2.0));
    }

    #[test]
    fn map_applies_everywhere() {
        let a = arr(&[1.0, 4.0, 9.0], &[3]);
        let r = a.map(f64::sqrt);
        assert_close(r.get(&[2]).unwrap(), 3.0, EPS);
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

    // ------------------------------------------------------------------- M5

    #[test]
    fn full_reductions() {
        let a = arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_close(a.sum(), 21.0, EPS);
        assert_close(a.mean(), 3.5, EPS);
        assert_eq!(a.min(), Some(1.0));
        assert_eq!(a.max(), Some(6.0));
    }

    #[test]
    fn empty_reductions() {
        let a = Array::zeros(&[0]);
        assert_eq!(a.sum(), 0.0);
        assert!(a.mean().is_nan());
        assert_eq!(a.min(), None);
        assert_eq!(a.max(), None);
    }
}

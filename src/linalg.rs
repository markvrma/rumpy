//! Matrix multiplication (M6): the efficiency showcase.
//!
//! Two implementations with identical results and wildly different speed.
//! The bench (`cargo run --release --bin bench`) measures both — the
//! before/after is the README's centerpiece.
//!
//! Read before implementing:
//! - Why loop order matters (cache lines, row-major traversal): search
//!   "cache-friendly matrix multiplication loop order ikj" — any of the
//!   standard writeups will do.
//! - What NumPy actually does: dispatches to BLAS (OpenBLAS/Accelerate).
//!   Pure-Rust naive code losing to BLAS by 10-100x is the *expected* result;
//!   beating your own naive version is the story.

use crate::{Array, ShapeError};

/// both 2-d and inner dims agreeing? hand back (m, k, n) if so.
fn dims(a: &Array, b: &Array) -> Result<(usize, usize, usize), ShapeError> {
    let bad = || ShapeError::Incompatible {
        a: a.shape().to_vec(),
        b: b.shape().to_vec(),
    };
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(bad());
    }
    let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
    if k != b.shape()[0] {
        return Err(bad());
    }
    Ok((m, k, n))
}

/// 2-d matrix product, `(m, k) x (k, n) → (m, n)`, textbook i-j-k loops.
///
/// The innermost loop walks `b` down a *column* — in C-order memory that's a
/// stride of `n` elements per step, so nearly every access misses cache.
///
/// Errors: `Incompatible` unless both are 2-d with matching inner dimension.
pub fn matmul_naive(a: &Array, b: &Array) -> Result<Array, ShapeError> {
    let (m, k, n) = dims(a, b)?;
    let mut out = Array::zeros(&[m, n]);
    // flat indexing by hand instead of get() — both are contiguous c-order so
    // row i starts at i*cols, and this keeps the loop free of Option unwraps.
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                // b[p][j] jumps n elements every step. that's the cache miss.
                acc += a.data[i * k + p] * b.data[p * n + j];
            }
            out.data[i * n + j] = acc;
        }
    }
    Ok(out)
}

/// Same product, i-k-j loop order: the innermost loop walks `b` and the
/// output along *rows* — contiguous memory, cache-friendly. Same O(m·k·n)
/// arithmetic, several times faster. Measure it, don't take it on faith.
pub fn matmul_ikj(a: &Array, b: &Array) -> Result<Array, ShapeError> {
    let (m, k, n) = dims(a, b)?;
    let mut out = Array::zeros(&[m, n]);
    // same multiplies, different order. j moved to the inside, so the inner
    // loop scans one row of b and one row of out straight through — every
    // cache line that gets pulled in is fully used before the next one.
    for i in 0..m {
        for p in 0..k {
            let aip = a.data[i * k + p];
            // hoisted so the inner loop is two adjacent walks and an fma
            let (row_b, row_c) = (p * n, i * n);
            for j in 0..n {
                out.data[row_c + j] += aip * b.data[row_b + j];
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_close;

    const EPS: f64 = 1e-9;

    fn arr(data: &[f64], shape: &[usize]) -> Array {
        Array::from_vec(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn matmul_known_product() {
        // [[1, 2, 3],    [[7,  8],      [[ 58,  64],
        //  [4, 5, 6]]  x  [9, 10],   =   [139, 154]]
        //                 [11, 12]]
        let a = arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = arr(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        for f in [matmul_naive, matmul_ikj] {
            let c = f(&a, &b).unwrap();
            assert_eq!(c.shape(), &[2, 2]);
            assert_close(c.get(&[0, 0]).unwrap(), 58.0, EPS);
            assert_close(c.get(&[0, 1]).unwrap(), 64.0, EPS);
            assert_close(c.get(&[1, 0]).unwrap(), 139.0, EPS);
            assert_close(c.get(&[1, 1]).unwrap(), 154.0, EPS);
        }
    }

    #[test]
    fn matmul_identity() {
        let a = arr(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mut i2 = Array::zeros(&[2, 2]);
        *i2.get_mut(&[0, 0]).unwrap() = 1.0;
        *i2.get_mut(&[1, 1]).unwrap() = 1.0;
        assert_eq!(matmul_naive(&a, &i2).unwrap(), a);
        assert_eq!(matmul_ikj(&i2, &a).unwrap(), a);
    }

    #[test]
    fn matmul_shape_mismatch() {
        let a = Array::zeros(&[2, 3]);
        let b = Array::zeros(&[4, 2]); // inner dims 3 vs 4
        assert!(matches!(
            matmul_naive(&a, &b),
            Err(ShapeError::Incompatible { .. })
        ));
        let v = Array::zeros(&[3]); // not 2-d
        assert!(matches!(
            matmul_naive(&a, &v),
            Err(ShapeError::Incompatible { .. })
        ));
    }

    #[test]
    fn naive_and_ikj_agree() {
        // Deterministic pseudo-random 16x16 (tiny LCG — no rand crate).
        let mut state: u64 = 42;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        };
        let data_a: Vec<f64> = (0..256).map(|_| next()).collect();
        let data_b: Vec<f64> = (0..256).map(|_| next()).collect();
        let a = arr(&data_a, &[16, 16]);
        let b = arr(&data_b, &[16, 16]);
        let c1 = matmul_naive(&a, &b).unwrap();
        let c2 = matmul_ikj(&a, &b).unwrap();
        for i in 0..16 {
            for j in 0..16 {
                assert_close(c1.get(&[i, j]).unwrap(), c2.get(&[i, j]).unwrap(), EPS);
            }
        }
    }
}

//! Zero-dependency benchmark runner. Methodology matches bench_numpy.py:
//! 2 warmup runs, median of N timed reps, CSV to stdout.
//!
//! Run with: cargo run --release --bin bench
//! Then:     python3 bench_numpy.py
//! and put both CSVs side by side in the README table.
//!
//! This file is provided complete (not a stub): timing methodology is
//! guidance, not the learning goal. Read it — black_box and median-of-reps
//! are the two ideas that make microbenchmarks honest.

use rumpy::{matmul_ikj, matmul_naive, Array};
use std::hint::black_box;
use std::time::Instant;

/// Median wall time of `reps` runs of `f`, in milliseconds, after 2 warmups.
/// Median (not mean) so one OS scheduling hiccup can't skew the number.
fn median_ms(mut f: impl FnMut(), reps: usize) -> f64 {
    f();
    f();
    let mut times: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    times[times.len() / 2]
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("ERROR: debug build. Benchmarks are meaningless without optimizations.");
        eprintln!("Run: cargo run --release --bin bench");
        std::process::exit(1);
    }

    println!("op,size,median_ms");

    // Elementwise + reduction: the fair fight vs NumPy (no BLAS involved).
    let n = 10_000_000;
    let a = Array::arange(0.0, n as f64, 1.0);
    let b = Array::ones(&[n]);
    println!(
        "add,{n},{:.3}",
        median_ms(|| { black_box(black_box(&a).add(black_box(&b)).unwrap()); }, 10)
    );
    println!(
        "sum,{n},{:.3}",
        median_ms(|| { black_box(black_box(&a).sum()); }, 10)
    );

    // Matmul: naive vs ikj is the story; NumPy (BLAS) is the reality check.
    let m = 512;
    let x = Array::arange(0.0, (m * m) as f64, 1.0)
        .reshape(&[m, m])
        .unwrap()
        .mul_scalar(1e-6);
    let y = x.clone();
    println!(
        "matmul_naive,{m}x{m},{:.3}",
        median_ms(|| { black_box(matmul_naive(black_box(&x), black_box(&y)).unwrap()); }, 5)
    );
    println!(
        "matmul_ikj,{m}x{m},{:.3}",
        median_ms(|| { black_box(matmul_ikj(black_box(&x), black_box(&y)).unwrap()); }, 5)
    );
}

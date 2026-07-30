#!/usr/bin/env python3
"""NumPy counterpart to `cargo run --release --bin bench`.

Same sizes, same methodology: 2 warmup runs, median of N timed reps,
CSV to stdout. Put both outputs side by side in the README table.

Fair-fight notes:
- add/sum: NumPy runs plain C loops here — this is the honest comparison.
- matmul: NumPy dispatches to BLAS (multi-threaded, SIMD, blocked). Pure-Rust
  loops losing by 10-100x is the expected teaching result. To single-thread
  BLAS for a slightly fairer number: OMP_NUM_THREADS=1 python3 bench_numpy.py

On this machine the system python3 has a broken NumPy (arch mismatch);
a known-good interpreter is:
  /opt/homebrew/Caskroom/miniconda/base/bin/python3 bench_numpy.py
"""

import time

import numpy as np


def median_ms(f, reps):
    f()
    f()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        times.append((time.perf_counter() - t) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main():
    print("op,size,median_ms")

    n = 10_000_000
    a = np.arange(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)
    print(f"add,{n},{median_ms(lambda: a + b, 10):.3f}")
    print(f"sum,{n},{median_ms(lambda: a.sum(), 10):.3f}")

    m = 512
    x = (np.arange(m * m, dtype=np.float64) * 1e-6).reshape(m, m)
    y = x.copy()
    print(f"matmul,{m}x{m},{median_ms(lambda: x @ y, 5):.3f}")


if __name__ == "__main__":
    main()

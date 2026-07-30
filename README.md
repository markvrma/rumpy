# rumpy

A bare-bones NumPy in pure Rust. Zero runtime dependencies — every array op,
broadcast rule, and matrix multiply is implemented from scratch against the
same design NumPy uses internally (flat buffer + shape + strides), then
benchmarked head-to-head against real NumPy.

**Status: work in progress.** 

## Why

Learning project with two goals: Rust fundamentals (ownership, traits,
`Result` error handling, testing, benchmarking) and NumPy fundamentals (what
an ndarray actually *is* under the hood, not just its API).

## How to work on it

```sh
cargo test                          # the spec — starts red, ends green
cargo test strides                  # run one milestone's tests by name filter
cargo run --release --bin bench     # Rust timings (refuses debug builds)
python3 bench_numpy.py              # NumPy timings, same methodology
```


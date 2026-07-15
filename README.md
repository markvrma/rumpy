# rumpy

A bare-bones NumPy in pure Rust. Zero runtime dependencies — every array op,
broadcast rule, and matrix multiply is implemented from scratch against the
same design NumPy uses internally (flat buffer + shape + strides), then
benchmarked head-to-head against real NumPy.


## Why

Learning project with two goals: Rust fundamentals (ownership, traits,
`Result` error handling, testing, benchmarking) and NumPy fundamentals (what
an ndarray actually *is* under the hood, not just its API).

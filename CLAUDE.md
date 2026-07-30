# rumpy — instructions for Claude agents

## What this project is

A bare-bones NumPy clone in pure Rust (zero runtime dependencies), built as a
**learning project** by Mark. The objective is not the library — it's Mark
getting fluent in Rust fundamentals (ownership, traits, `Result`, testing,
benchmarking) and NumPy internals (flat buffer + shape + strides, broadcasting,
cache-aware matmul). Endgame: CV-worthy repo with an honest NumPy-vs-Rust
benchmark table.

## The one rule that matters

**Do NOT implement the `todo!()` stub bodies. They are Mark's homework.**

The scaffold (struct definitions, stub signatures, doc comments, all tests,
bench harness, README) was generated deliberately; the implementations are the
learning exercise. When helping:

- Give hints, explain concepts, point to docs, review Mark's code, help debug.
- Never fill in a `todo!()` unless Mark explicitly asks for the implementation.
- Never "fix" a failing test by editing its expected values — tests are the
  spec. Fix code, not tests.
- Never add dependencies to Cargo.toml. Zero runtime deps is a project
  constraint (dev-deps also currently zero; `criterion` is a noted possible
  exception, ask first).

## Layout & roadmap

| File | Milestones | Contents |
|---|---|---|
| `src/lib.rs` | done | `ShapeError`, `assert_close`, API surface |
| `src/array.rs` | M1–M2 | `Array` struct, constructors, index math, reshape/transpose |
| `src/ops.rs` | M3–M5 | elementwise ops, broadcasting, reductions |
| `src/linalg.rs` | M6 | `matmul_naive` → `matmul_ikj` |
| `src/bin/bench.rs` | M7 | complete bench harness (not homework) |
| `bench_numpy.py` | M7 | NumPy counterpart, same methodology |

Milestone done = its tests green. Full roadmap, reading list, and locked
design decisions live in README.md — read it before advising.

## Locked design decisions (don't relitigate)

`f64` only · C-order only · strides count elements not bytes · copy-based
reshape/transpose (no-copy views = stretch goal) · `Result<_, ShapeError>`
everywhere, operators panic like `v[i]` · stride-0 on-the-fly broadcasting,
no materialized copies · no keepdims, no 0-d arrays, no fancy indexing.

## Commands

```sh
cargo test                        # the spec; starts red, goes green
cargo test <name-filter>          # one milestone's tests
cargo run --release --bin bench   # Rust timings (refuses debug builds)
python3 bench_numpy.py            # NumPy timings (needs working numpy)
```

Compiler warnings about unused params/dead code come from unimplemented stubs
and disappear as milestones land — leave them.

Machine note (Mark's Mac): system `python3` has arch-broken numpy; use
`/opt/homebrew/Caskroom/miniconda/base/bin/python3`.

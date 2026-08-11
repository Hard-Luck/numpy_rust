# numpy_rust

Targeted NumPy baseline: 2.5.1.

This crate ports the NumPy 2.5.1 `default_rng()` PCG64 path needed to reproduce the double-precision standard normal and Poisson distributions without invoking Python at runtime.

The low-level PCG64 engine is backed by `rand_pcg::Pcg64`, while NumPy-compatible seeding and distribution sampling remain implemented in this crate.

This project is distributed under the BSD 3-Clause License. It also includes material ported or adapted from NumPy and preserves the applicable upstream notices in `THIRD_PARTY_NOTICES.md`.

Source anchors used for the port:

- `numpy/random/bit_generator.pyx`: `SeedSequence`, `hashmix`, `mix`, and `generate_state`
- `numpy/random/_pcg64.pyx`: `PCG64.__init__` seeding path via `generate_state(4, np.uint64)`
- `numpy/random/src/distributions/distributions.c`: `random_standard_normal`, `random_poisson`, `random_poisson_mult`, `random_poisson_ptrs`, `random_loggam`
- `numpy/random/src/distributions/ziggurat_constants.h`: double-precision ziggurat tables and constants

`numpy/random/src/pcg64/pcg64.h` and `pcg64.c` remain useful upstream reference material, but the underlying PCG64 engine is now provided by `rand_pcg` rather than a direct Rust port of those files.

Current scope tracks the latest requirement narrowing to normal and Poisson. No intentional algorithmic deviations are introduced in these distribution paths.

## How to use

Add the crate to your `Cargo.toml`.

```toml
[dependencies]
numpy_rust = { path = "." }
```

Create a generator with the same integer-seed entry point as `numpy.random.default_rng(seed)`, then draw values from the supported distributions.

```rust
use numpy_rust::{Pcg64, target_numpy_version};

fn main() {
	let mut rng = Pcg64::from_seed(42);

	let uniform = rng.random();
	let standard_normal = rng.standard_normal();
	let shifted_normal = rng.normal_scalar(10.0, 2.5);
	let shifted_normal_batch = rng.normal(10.0, 2.5, 4);
	let poisson = rng.poisson(4.0);

	println!("NumPy baseline: {}", target_numpy_version());
	println!("uniform={uniform}");
	println!("standard_normal={standard_normal}");
	println!("shifted_normal={shifted_normal}");
	println!("shifted_normal_batch={shifted_normal_batch:?}");
	println!("poisson={poisson}");
}
```

Supported public entry points:

- `Pcg64::from_seed(u64)` seeds the generator using NumPy's `default_rng(integer_seed)` PCG64 path.
- `Pcg64::from_seed_slice(&[u64])` seeds the generator using NumPy's `default_rng(seed=[...])` `SeedSequence` path.
- `Pcg64::random()` returns a uniform `f64` in `[0.0, 1.0)`.
- `Pcg64::random_array(size)` returns `size` uniform `f64` samples in `[0.0, 1.0)`.
- `Pcg64::next_double()` returns a uniform `f64` in `[0.0, 1.0)`.
- `Pcg64::standard_normal()` returns a standard normal `f64`.
- `Pcg64::normal_scalar(loc, scale)` returns one normal `f64` with the given mean and standard deviation.
- `Pcg64::normal(loc, scale, size)` returns `size` normal samples as `Vec<f64>`.
- `Pcg64::normal_array(loc, scale, size)` is an alias for `normal(loc, scale, size)`.
- `Pcg64::poisson(lam)` returns a Poisson sample as `i64`.
- `Pcg64::poisson_array(lam, size)` returns `size` Poisson samples as `Vec<i64>`.
- `Pcg64::poisson_array_from_slice(&[f64], size)` returns one Poisson sample per lambda in the input slice and currently expects `size == lam.len()`.
- `target_numpy_version()` returns the NumPy version this port is matched against.

For repeated sampling, keep one `Pcg64` seeded the same way as Python's `np.random.default_rng(seed)` and consume it in the same call order as the Python code you are matching.

```rust
use numpy_rust::Pcg64;

fn sample_sequences(seed: u64) {
	let mut rng = Pcg64::from_seed(seed);

	let uniforms = rng.random_array(4);
	let normals = rng.normal(0.0, 1.0, 4);
	let poisson = rng.poisson_array(3.5, 4);

	println!("{uniforms:?}");
	println!("{normals:?}");
	println!("{poisson:?}");
}
```

For sequence seeding, use `from_seed_slice` with the same integer sequence you would pass to `np.random.default_rng(seed=[...])`:

```rust
use numpy_rust::Pcg64;

fn sample_from_sequence_seed(seed_words: &[u64]) -> Vec<i64> {
	let mut rng = Pcg64::from_seed_slice(seed_words);
	rng.poisson_array_from_slice(&[0.1, 0.5, 2.0, 10.0], 4)
}
```

If you are generating multiple arrays or mixing Poisson and normal draws from the same Python generator instance, keep a single `Pcg64` alive in Rust as well and consume it in the same order.

Unlike `rand_distr::Poisson::new`, this crate accepts `lambda == 0.0` and returns `0`.

If you need byte-for-byte compatibility with NumPy output, construct a fresh `Pcg64` with the same seed and call methods in the same order as the Python code you are matching.

For Poisson specifically, compatibility is defined by this crate's internal NumPy-derived sampler. Replacing `Pcg64::poisson(lambda)` with a different Poisson implementation such as `rand_distr::Poisson` will not preserve NumPy-matching output even if `lambda` is identical.

## Equivalence Testing

This repository verifies compatibility against Python by generating fixture data with NumPy 2.5.1 and checking that the Rust implementation reproduces the same outputs bit-for-bit.

The fixture generator in `scripts/generate_fixtures.py` does three important things:

- It pins the upstream baseline to one NumPy version so the tests are tied to a single known implementation.
- It creates fresh `np.random.default_rng(seed)` instances for each distribution path being compared, so the Rust tests validate the same call ordering and RNG state transitions as the Python reference.
- It stores `f64` outputs as raw IEEE-754 bit patterns rather than decimal text, which avoids false mismatches caused by formatting or parsing differences.

The generated fixture file is `tests/fixtures/generated_fixtures.rs`, and `cargo test` compares the Rust output against those saved Python results.

Intentional deviation:

- The public Rust API exposes the exact `default_rng(integer_seed)` path through `Pcg64::from_seed(u64)` and NumPy's integer-sequence path through `Pcg64::from_seed_slice(&[u64])`. It does not yet expose the broader Python seed surface such as arbitrary-size Python integers, `SeedSequence` objects, or existing bit generators.

Regenerate compatibility fixtures with:

- `/usr/local/bin/python3 scripts/generate_fixtures.py`

## Contributing

This crate currently exposes the API surface needed for the original downstream project that motivated the port.

If you would like to build on this further, broaden the NumPy coverage, or expose additional compatible APIs, pull requests are welcome.

## Disclaimer

AI tooling was used during the porting process.

The repository includes targeted equivalence tests against NumPy 2.5.1, but mistakes may still be present.

## Licensing

- This repository is licensed under the BSD 3-Clause License. See `LICENSE`.
- This repository includes code and constants ported or adapted from NumPy 2.5.1 random-generation sources.
- Upstream acknowledgements, attributions, and required license texts are reproduced in `THIRD_PARTY_NOTICES.md`.

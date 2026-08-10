from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


# The fixture data in this script is the Python source of truth for the Rust
# compatibility tests. We pin one NumPy version and serialize the exact output
# bits that Rust must reproduce.
TARGET_VERSION = "2.5.1"
SEEDS = [0, 1, 42, 123456789]
POISSON_LAMBDAS = [0.1, 0.5, 2.0, 10.0, 100.0]
COUNT = 100
RANDOM_EXAMPLE_SOURCE_SEED = 0x20260731
RANDOM_EXAMPLE_COUNT = 8
RANDOM_EXAMPLE_SAMPLE_COUNT = 12


def f64_bits(value: float) -> int:
    # Store floating-point fixtures as raw IEEE-754 bits so the Rust tests can
    # compare exact output without any decimal formatting ambiguity.
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def format_u64_array(values: list[int]) -> str:
    body = ",\n            ".join(f"0x{value:016x}" for value in values)
    return "[\n            " + body + "\n        ]"


def format_i64_array(values: list[int]) -> str:
    body = ",\n                ".join(str(value) for value in values)
    return "[\n                " + body + "\n            ]"


def format_f64_array(values: list[float]) -> str:
    body = ",\n            ".join(repr(value) for value in values)
    return "[\n            " + body + "\n        ]"


def main() -> None:
    if np.__version__ != TARGET_VERSION:
        raise SystemExit(
            f"Expected numpy {TARGET_VERSION}, found {np.__version__}. "
            "Regenerate fixtures with the pinned NumPy baseline only."
        )

    out_path = Path(__file__).resolve(
    ).parents[1] / "tests" / "fixtures" / "generated_fixtures.rs"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixture_blocks: list[str] = []
    for seed in SEEDS:
        # Use fresh generators per distribution family so the Rust tests can
        # compare each public API surface against the matching NumPy call path.
        doubles_rng = np.random.default_rng(seed)
        doubles = [f64_bits(value) for value in doubles_rng.random(COUNT)]

        normals_rng = np.random.default_rng(seed)
        normals = [f64_bits(value)
                   for value in normals_rng.standard_normal(COUNT)]

        poisson_rows = []
        for lam in POISSON_LAMBDAS:
            # Each lambda gets its own fresh generator, matching the way the
            # Rust compatibility tests isolate Poisson sampling by seed/lambda.
            poisson_rng = np.random.default_rng(seed)
            poisson_values = [int(value)
                              for value in poisson_rng.poisson(lam, COUNT)]
            poisson_rows.append(format_i64_array(poisson_values))

        poisson_block = "[\n            " + \
            ",\n            ".join(poisson_rows) + "\n        ]"
        fixture_blocks.append(
            "    Fixture {\n"
            f"        seed: {seed},\n"
            f"        doubles: {format_u64_array(doubles)},\n"
            f"        normals: {format_u64_array(normals)},\n"
            f"        poisson: {poisson_block},\n"
            "    }"
        )

    # These randomized examples widen coverage beyond a few hand-picked seeds
    # while staying deterministic because the source generator is fixed.
    random_source = np.random.default_rng(RANDOM_EXAMPLE_SOURCE_SEED)
    random_example_seeds = [
        int(value)
        for value in random_source.integers(
            0,
            np.iinfo(np.uint64).max,
            size=RANDOM_EXAMPLE_COUNT,
            dtype=np.uint64,
        )
    ]
    random_example_lambdas = [
        float(value)
        for value in random_source.uniform(0.01, 250.0, size=RANDOM_EXAMPLE_COUNT)
    ]

    random_example_blocks: list[str] = []
    for seed, lam in zip(random_example_seeds, random_example_lambdas):
        # As above, each distribution is sampled from a fresh NumPy generator
        # for the same seed so Rust can validate identical public entry points.
        doubles_rng = np.random.default_rng(seed)
        doubles = [f64_bits(value) for value in doubles_rng.random(RANDOM_EXAMPLE_SAMPLE_COUNT)]

        normals_rng = np.random.default_rng(seed)
        normals = [
            f64_bits(value)
            for value in normals_rng.standard_normal(RANDOM_EXAMPLE_SAMPLE_COUNT)
        ]

        poisson_rng = np.random.default_rng(seed)
        poisson = [
            int(value)
            for value in poisson_rng.poisson(lam, RANDOM_EXAMPLE_SAMPLE_COUNT)
        ]

        random_example_blocks.append(
            "    RandomExample {\n"
            f"        seed: {seed},\n"
            f"        lambda: {lam!r},\n"
            f"        doubles: {format_u64_array(doubles)},\n"
            f"        normals: {format_u64_array(normals)},\n"
            f"        poisson: {format_i64_array(poisson)},\n"
            "    }"
        )

    content = (
        "pub const TARGET_NUMPY_VERSION: &str = \"2.5.1\";\n"
        f"pub const POISSON_LAMBDAS: [f64; {len(POISSON_LAMBDAS)}] = {POISSON_LAMBDAS!r};\n\n"
        "#[derive(Clone, Copy)]\n"
        "pub struct Fixture {\n"
        "    pub seed: u64,\n"
        "    pub doubles: [u64; 100],\n"
        "    pub normals: [u64; 100],\n"
        "    pub poisson: [[i64; 100]; 5],\n"
        "}\n\n"
        f"pub const FIXTURES: [Fixture; {len(SEEDS)}] = [\n"
        + ",\n".join(fixture_blocks)
        + "\n];\n\n"
        + f"pub const RANDOM_EXAMPLE_SOURCE_SEED: u64 = {RANDOM_EXAMPLE_SOURCE_SEED};\n"
        + "#[derive(Clone, Copy)]\n"
        + "pub struct RandomExample {\n"
        + "    pub seed: u64,\n"
        + "    pub lambda: f64,\n"
        + f"    pub doubles: [u64; {RANDOM_EXAMPLE_SAMPLE_COUNT}],\n"
        + f"    pub normals: [u64; {RANDOM_EXAMPLE_SAMPLE_COUNT}],\n"
        + f"    pub poisson: [i64; {RANDOM_EXAMPLE_SAMPLE_COUNT}],\n"
        + "}\n\n"
        + f"pub const RANDOM_EXAMPLES: [RandomExample; {RANDOM_EXAMPLE_COUNT}] = [\n"
        + ",\n".join(random_example_blocks)
        + "\n];\n"
    )
    out_path.write_text(content)


if __name__ == "__main__":
    main()

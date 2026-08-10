mod constants;

use constants::{FI_DOUBLE, KI_DOUBLE, WI_DOUBLE, ZIGGURAT_NOR_INV_R, ZIGGURAT_NOR_R};
use rand_pcg::{Pcg64 as RandPcg64, rand_core::Rng};

// This file contains code and constants ported or adapted from NumPy 2.5.1
// random-generation sources, including numpy.random SeedSequence, PCG64,
// standard normal, Poisson, and ziggurat-derived tables. Applicable upstream
// license notices and acknowledgements are preserved in THIRD_PARTY_NOTICES.md.
//
// Ported directly from NumPy 2.5.1:
// - numpy/random/bit_generator.pyx
// - numpy/random/_pcg64.pyx
// - numpy/random/src/pcg64/pcg64.h
// - numpy/random/src/distributions/distributions.c
// - numpy/random/src/distributions/ziggurat_constants.h

const DEFAULT_POOL_SIZE: usize = 4;
const INIT_A: u32 = 0x43b0d7e5;
const MULT_A: u32 = 0x931e8875;
const INIT_B: u32 = 0x8b51f9dd;
const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd;
const MIX_MULT_R: u32 = 0x4973f715;
const XSHIFT: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeedState {
    pub words: [u64; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcg64 {
    core: RandPcg64,
}

impl Pcg64 {
    /// Constructs a generator from the same integer-seed path as
    /// `numpy.random.default_rng(seed)`.
    pub fn from_seed(seed: u64) -> Self {
        let seed_state = SeedSequence::from_u64(seed).generate_state_u64_4();
        Self::from_seed_state(seed_state)
    }

    /// Constructs a generator from the same integer-sequence seed path as
    /// `numpy.random.default_rng(seed=[...])`.
    pub fn from_seed_slice(seed: &[u64]) -> Self {
        let seed_state = SeedSequence::from_u64_slice(seed).generate_state_u64_4();
        Self::from_seed_state(seed_state)
    }

    /// Builds the generator from a NumPy-compatible four-word seed state.
    fn from_seed_state(seed_state: SeedState) -> Self {
        let initstate = ((seed_state.words[0] as u128) << 64) | seed_state.words[1] as u128;
        let initseq = ((seed_state.words[2] as u128) << 64) | seed_state.words[3] as u128;
        Self {
            core: RandPcg64::new(initstate, initseq),
        }
    }

    /// Returns the four generated `u64` words NumPy uses to initialize PCG64
    /// from an integer seed.
    pub fn seed_state(seed: u64) -> SeedState {
        SeedSequence::from_u64(seed).generate_state_u64_4()
    }

    /// Returns the four generated `u64` words NumPy uses to initialize PCG64
    /// from an integer-sequence seed.
    pub fn seed_state_from_slice(seed: &[u64]) -> SeedState {
        SeedSequence::from_u64_slice(seed).generate_state_u64_4()
    }

    /// Returns the next raw 64-bit value from the underlying PCG64 generator.
    pub fn next_u64(&mut self) -> u64 {
        self.core.next_u64()
    }

    /// Returns the next uniform `f64` in `[0.0, 1.0)` using NumPy's bit-to-double mapping.
    /// >> 11 is used to take the top 53 bits of the 64-bit integer, which is then scaled to [0.0, 1.0).
    /// 9007199254740992e-16 is 1.0 / 2^53, the smallest double-precision value that can be added to 1.0.
    pub fn next_double(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9007199254740992.0)
    }

    /// Returns one uniform `f64` in `[0.0, 1.0)`.
    pub fn random(&mut self) -> f64 {
        self.next_double()
    }

    /// Returns `size` uniform `f64` samples in `[0.0, 1.0)`.
    pub fn random_array(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.random()).collect()
    }

    /// Returns one standard normal sample using NumPy's double-precision ziggurat algorithm.
    pub fn standard_normal(&mut self) -> f64 {
        loop {
            /* r = e3n52sb8 */
            let mut r = self.next_u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = (r & 0x1) as i32;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = (rabs as f64) * WI_DOUBLE[idx];
            if (sign & 0x1) != 0 {
                x = -x;
            }
            if rabs < KI_DOUBLE[idx] {
                /* 99.3% of the time return here */
                return x;
            }
            if idx == 0 {
                loop {
                    /* Switch to 1.0 - U to avoid log(0.0), see GH 13361 */
                    let xx = -ZIGGURAT_NOR_INV_R * (-self.next_double()).ln_1p();
                    let yy = -(-self.next_double()).ln_1p();
                    if yy + yy > xx * xx {
                        return if ((rabs >> 8) & 0x1) != 0 {
                            -(ZIGGURAT_NOR_R + xx)
                        } else {
                            ZIGGURAT_NOR_R + xx
                        };
                    }
                }
            } else if ((FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]) * self.next_double() + FI_DOUBLE[idx])
                < (-0.5 * x * x).exp()
            {
                return x;
            }
        }
    }

    /// Returns one normal sample with the given location and scale.
    pub fn normal_scalar(&mut self, loc: f64, scale: f64) -> f64 {
        loc + scale * self.standard_normal()
    }

    /// Returns `size` normal samples with the given location and scale.
    pub fn normal(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.normal_scalar(loc, scale)).collect()
    }

    /// Alias for [`Pcg64::normal`] kept for call-site convenience.
    pub fn normal_array(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        self.normal(loc, scale, size)
    }

    /// Returns one Poisson sample with mean `lam` using NumPy-compatible logic.
    pub fn poisson(&mut self, lam: f64) -> i64 {
        if lam >= 10.0 {
            self.random_poisson_ptrs(lam)
        } else if lam == 0.0 {
            0
        } else {
            self.random_poisson_mult(lam)
        }
    }

    /// Returns `size` Poisson samples with a shared scalar `lam`.
    pub fn poisson_array(&mut self, lam: f64, size: usize) -> Vec<i64> {
        (0..size).map(|_| self.poisson(lam)).collect()
    }

    /// Returns one Poisson sample per element in `lam`.
    ///
    /// `size` must match `lam.len()` to mirror the explicit shape used by the
    /// current crate API.
    pub fn poisson_array_from_slice(&mut self, lam: &[f64], size: usize) -> Vec<i64> {
        assert_eq!(lam.len(), size, "size must match the lambda slice length");
        lam.iter().map(|&value| self.poisson(value)).collect()
    }

    /// Samples a Poisson variate using the multiplicative method used for small means.
    fn random_poisson_mult(&mut self, lam: f64) -> i64 {
        let enlam = (-lam).exp();
        let mut x = 0_i64;
        let mut prod = 1.0;
        loop {
            let u = self.next_double();
            prod *= u;
            if prod > enlam {
                x += 1;
            } else {
                return x;
            }
        }
    }

    /*
     * The transformed rejection method for generating Poisson random variables
     * W. Hoermann
     * Insurance: Mathematics and Economics 12, 39-45 (1993)
     */
    /// Samples a Poisson variate using the transformed rejection method used
    /// for larger means.
    fn random_poisson_ptrs(&mut self, lam: f64) -> i64 {
        let slam = lam.sqrt();
        let loglam = lam.ln();
        let b = 0.931 + 2.53 * slam;
        let a = -0.059 + 0.02483 * b;
        let invalpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);

        loop {
            let u = self.next_double() - 0.5;
            let v = self.next_double();
            let us = 0.5 - u.abs();
            let k = (((2.0 * a / us + b) * u) + lam + 0.43).floor() as i64;
            if us >= 0.07 && v <= vr {
                return k;
            }
            if k < 0 || (us < 0.013 && v > us) {
                continue;
            }
            /* log(V) == log(0.0) ok here */
            /* if U==0.0 so that us==0.0, log is ok since always returns */
            if v.ln() + invalpha.ln() - (a / (us * us) + b).ln()
                <= -lam + (k as f64) * loglam - random_loggam((k as f64) + 1.0)
            {
                return k;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SeedSequence {
    pool: [u32; DEFAULT_POOL_SIZE],
}

impl SeedSequence {
    /// Constructs a NumPy-compatible `SeedSequence` from one integer value.
    fn from_u64(entropy: u64) -> Self {
        let mut pool = [0_u32; DEFAULT_POOL_SIZE];
        let run_entropy = u64_entropy_words(&[entropy]);
        mix_entropy(&mut pool, &run_entropy);
        Self { pool }
    }

    /// Constructs a NumPy-compatible `SeedSequence` from an integer slice.
    fn from_u64_slice(entropy: &[u64]) -> Self {
        let mut pool = [0_u32; DEFAULT_POOL_SIZE];
        let run_entropy = u64_entropy_words(entropy);
        mix_entropy(&mut pool, &run_entropy);
        Self { pool }
    }

    /// Expands the mixed entropy pool into the four `u64` words used to seed PCG64.
    fn generate_state_u64_4(&self) -> SeedState {
        let mut state32 = [0_u32; 8];
        let mut hash_const = INIT_B;
        for (i, item) in state32.iter_mut().enumerate() {
            let mut data_val = self.pool[i % self.pool.len()];
            data_val ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            data_val = data_val.wrapping_mul(hash_const);
            data_val ^= data_val >> XSHIFT;
            *item = data_val;
        }

        let mut words = [0_u64; 4];
        for (idx, chunk) in state32.chunks_exact(2).enumerate() {
            words[idx] = (chunk[0] as u64) | ((chunk[1] as u64) << 32);
        }
        SeedState { words }
    }
}

fn u64_entropy_words(entropy: &[u64]) -> Vec<u32> {
    let mut words = Vec::with_capacity(entropy.len().max(1) * 2);
    for &value in entropy {
        if value == 0 {
            words.push(0);
            continue;
        }

        let mut remaining = value;
        while remaining > 0 {
            words.push((remaining & 0xffff_ffff) as u32);
            remaining >>= 32;
        }
    }

    if words.is_empty() {
        words.push(0);
    }

    words
}

fn mix_entropy(mixer: &mut [u32; DEFAULT_POOL_SIZE], entropy_array: &[u32]) {
    let mut hash_const = INIT_A;
    // Add in the entropy up to the pool size.
    for (idx, item) in mixer.iter_mut().enumerate() {
        let value = entropy_array.get(idx).copied().unwrap_or(0);
        *item = hashmix(value, &mut hash_const);
    }

    // Mix all bits together so late bits can affect earlier bits.
    for i_src in 0..mixer.len() {
        for i_dst in 0..mixer.len() {
            if i_src != i_dst {
                let mixed = hashmix(mixer[i_src], &mut hash_const);
                mixer[i_dst] = mix(mixer[i_dst], mixed);
            }
        }
    }

    // Add any remaining entropy, mixing each new entropy word with each pool word.
    for value in entropy_array.iter().skip(mixer.len()) {
        for item in mixer.iter_mut() {
            *item = mix(*item, hashmix(*value, &mut hash_const));
        }
    }
}

fn hashmix(mut value: u32, hash_const: &mut u32) -> u32 {
    // We are modifying the multiplier as we go along, so it is input-output.
    value ^= *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^= value >> XSHIFT;
    value
}

fn mix(x: u32, y: u32) -> u32 {
    let mut result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^= result >> XSHIFT;
    result
}

fn random_loggam(x: f64) -> f64 {
    // If random_loggam(k+1) is being used to compute log(k!) for an integer k,
    // NumPy's source notes that logfactorial(k) should be considered instead.
    let a = [
        8.333333333333333e-02,
        -2.777777777777778e-03,
        7.936507936507937e-04,
        -5.952380952380952e-04,
        8.417508417508418e-04,
        -1.917526917526918e-03,
        6.410256410256410e-03,
        -2.955065359477124e-02,
        1.796443723688307e-01,
        -1.39243221690590e+00,
    ];

    if x == 1.0 || x == 2.0 {
        return 0.0;
    }

    let n = if x < 7.0 { (7.0 - x) as i64 } else { 0 };
    let mut x0 = x + n as f64;
    let x2 = (1.0 / x0) * (1.0 / x0);
    let lg2pi = 1.8378770664093453e+00;
    let mut gl0 = a[9];
    for k in (0..=8).rev() {
        gl0 *= x2;
        gl0 += a[k];
    }
    let mut gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * x0.ln() - x0;
    if x < 7.0 {
        for _ in 1..=n {
            gl -= (x0 - 1.0).ln();
            x0 -= 1.0;
        }
    }
    gl
}

pub fn target_numpy_version() -> &'static str {
    "2.5.1"
}

pub fn target_numpy_notes() -> &'static str {
    "Ported against NumPy 2.5.1 source: SeedSequence in random/bit_generator.pyx, PCG64 in random/_pcg64.pyx and random/src/pcg64, standard normal and Poisson in random/src/distributions/distributions.c, double ziggurat tables in random/src/distributions/ziggurat_constants.h."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_sequence_matches_numpy_reference_for_42() {
        let state = Pcg64::seed_state(42);
        let observed = [
            state.words[0] as u32,
            (state.words[0] >> 32) as u32,
            state.words[1] as u32,
            (state.words[1] >> 32) as u32,
        ];
        assert_eq!(observed, [3444837047, 2669555309, 2046530742, 3581440988]);
    }

    #[test]
    fn seed_sequence_matches_numpy_reference_for_u64_slice() {
        let state = Pcg64::seed_state_from_slice(&[42, 43, 44]);
        let observed = [
            state.words[0] as u32,
            (state.words[0] >> 32) as u32,
            state.words[1] as u32,
            (state.words[1] >> 32) as u32,
            state.words[2] as u32,
            (state.words[2] >> 32) as u32,
            state.words[3] as u32,
            (state.words[3] >> 32) as u32,
        ];
        assert_eq!(
            observed,
            [
                167534246, 220497560, 2018543265, 1697036488, 2091501443, 447637570, 572053801,
                2514420289,
            ]
        );
    }
}

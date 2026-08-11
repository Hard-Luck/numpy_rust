// Ported or adapted from NumPy 2.5.1 random/_pcg64.pyx.

use rand_pcg::{Pcg64 as RandPcg64, rand_core::Rng};

use crate::bit_generator::{SeedSequence, SeedState};

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
}

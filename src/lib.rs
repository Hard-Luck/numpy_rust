mod _pcg64;
mod bit_generator;
mod distributions;
mod ziggurat_constants;

pub use _pcg64::Pcg64;
pub use bit_generator::SeedState;

// This crate contains code and constants ported or adapted from NumPy 2.5.1
// random-generation sources, split into Rust modules that mirror the upstream
// source anchors used by this crate. The low-level PCG64 core is delegated to
// rand_pcg. Applicable upstream license notices and acknowledgements are
// preserved in THIRD_PARTY_NOTICES.md.

pub fn target_numpy_version() -> &'static str {
    "2.5.1"
}

pub fn target_numpy_notes() -> &'static str {
    "Ported against NumPy 2.5.1 source: SeedSequence in random/bit_generator.pyx, the PCG64 seeding path in random/_pcg64.pyx, standard normal and Poisson in random/src/distributions/distributions.c, and double ziggurat tables in random/src/distributions/ziggurat_constants.h. The low-level PCG64 core is provided by rand_pcg."
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

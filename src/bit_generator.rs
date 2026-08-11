// Ported or adapted from NumPy 2.5.1 random/bit_generator.pyx.

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
pub(crate) struct SeedSequence {
    pool: [u32; DEFAULT_POOL_SIZE],
}

impl SeedSequence {
    /// Constructs a NumPy-compatible `SeedSequence` from one integer value.
    pub(crate) fn from_u64(entropy: u64) -> Self {
        let mut pool = [0_u32; DEFAULT_POOL_SIZE];
        let run_entropy = u64_entropy_words(&[entropy]);
        mix_entropy(&mut pool, &run_entropy);
        Self { pool }
    }

    /// Constructs a NumPy-compatible `SeedSequence` from an integer slice.
    pub(crate) fn from_u64_slice(entropy: &[u64]) -> Self {
        let mut pool = [0_u32; DEFAULT_POOL_SIZE];
        let run_entropy = u64_entropy_words(entropy);
        mix_entropy(&mut pool, &run_entropy);
        Self { pool }
    }

    /// Expands the mixed entropy pool into the four `u64` words used to seed PCG64.
    pub(crate) fn generate_state_u64_4(&self) -> SeedState {
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
    for (idx, item) in mixer.iter_mut().enumerate() {
        let value = entropy_array.get(idx).copied().unwrap_or(0);
        *item = hashmix(value, &mut hash_const);
    }

    for i_src in 0..mixer.len() {
        for i_dst in 0..mixer.len() {
            if i_src != i_dst {
                let mixed = hashmix(mixer[i_src], &mut hash_const);
                mixer[i_dst] = mix(mixer[i_dst], mixed);
            }
        }
    }

    for value in entropy_array.iter().skip(mixer.len()) {
        for item in mixer.iter_mut() {
            *item = mix(*item, hashmix(*value, &mut hash_const));
        }
    }
}

fn hashmix(mut value: u32, hash_const: &mut u32) -> u32 {
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

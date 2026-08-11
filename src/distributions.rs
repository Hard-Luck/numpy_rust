// Ported or adapted from NumPy 2.5.1 random/src/distributions/distributions.c.

use crate::_pcg64::Pcg64;
use crate::ziggurat_constants::{
    FI_DOUBLE, KI_DOUBLE, WI_DOUBLE, ZIGGURAT_NOR_INV_R, ZIGGURAT_NOR_R,
};

impl Pcg64 {
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
                return x;
            }
            if idx == 0 {
                loop {
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
        self.standard_normal().mul_add(scale, loc)
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
    pub fn poisson_array_from_slice(&mut self, lam: &[f64], size: usize) -> Vec<i64> {
        assert_eq!(lam.len(), size, "size must match the lambda slice length");
        lam.iter().map(|&value| self.poisson(value)).collect()
    }

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
            if v.ln() + invalpha.ln() - (a / (us * us) + b).ln()
                <= -lam + (k as f64) * loglam - random_loggam((k as f64) + 1.0)
            {
                return k;
            }
        }
    }
}

fn random_loggam(x: f64) -> f64 {
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

use std::mem::swap;
use std::sync::atomic::Ordering::{self, Relaxed};
use std::sync::atomic::{AtomicU8, AtomicU64};

use ndarray::prelude::*;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug, Default, PartialEq, serde::Deserialize)]
pub struct Weights {
    pub semantic: f64,
    pub matching: f64,
    pub alignment: f64,
    pub acceleration: f64,
}

#[derive(Debug)]
pub struct Backward {
    pub index: Array1<usize>,
    pub speed: Array1<usize>,
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "The values are small."
)]
#[expect(clippy::many_single_char_names, reason = "Mathematical notation.")]
pub fn forward_sequential(
    semantic: ArrayView1<f64>,
    matching: ArrayView2<f64>,
    alignment: ArrayView1<f64>,
    weights: &Weights,
    maximum_speed: usize,
) -> Array2<u8> {
    let evaluate = {
        let semantic = rankdata(semantic);
        let alignment = rankdata(alignment);

        let δ = maximum_speed as f64;
        let μ = semantic.len() as f64 / alignment.len() as f64;

        move |a: usize, v: usize, s: usize, p: usize| {
            let semantic = if (s as f64) < μ {
                1. - semantic[v]
            } else {
                1. - 0.05 * semantic[v]
            };
            let matching = matching[[v, s]];

            let s = s as f64;
            let p = p as f64;

            let τ = 1. + (δ - 1.) * alignment[a];

            let alignment = (s - τ).powi(2) / (δ - 1.).powi(2);
            let acceleration = (s - p).powi(2) / (δ - 1.).powi(2);

            weights.semantic * semantic
                + weights.matching * matching
                + weights.alignment * alignment
                + weights.acceleration * acceleration
        }
    };

    let mut forward = Array::zeros([alignment.len(), semantic.len()]);

    let mut current = Array::from_elem(semantic.len(), f64::INFINITY);
    let mut previous = Array::from_elem(semantic.len(), f64::INFINITY);

    let mu = (semantic.len() / alignment.len()) as u8;

    current[0] = 0.0;
    forward[[0, 0]] = mu;

    for a in 1..alignment.len() {
        swap(&mut previous, &mut current);
        current.fill(f64::INFINITY);

        let remaining = alignment.len() - a;

        let v_min = usize::max(
            // Moving forward at minimum speed.
            a,
            // Moving backward at maximum speed.
            semantic.len().saturating_sub(remaining * maximum_speed),
        );
        let v_max = usize::min(
            // Moving forward at maximum speed.
            a * maximum_speed,
            // Moving backward at minimum speed.
            semantic.len().saturating_sub(remaining),
        );

        for v in v_min..=v_max {
            let s_min = 1;
            let s_max = usize::min(maximum_speed, v);

            let mut min = f64::INFINITY;
            let mut argmin = 0;

            for s in s_min..=s_max {
                let sum = previous[v - s];
                if sum.is_finite() {
                    let p = forward[[a - 1, v - s]] as usize;
                    let value = sum + evaluate(a, v, s, p);

                    if value < min {
                        min = value;
                        argmin = s;
                    }
                }
            }

            current[v] = min;
            forward[[a, v]] = argmin as u8;
        }
    }

    forward
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "The values are small."
)]
#[expect(clippy::many_single_char_names, reason = "Mathematical notation.")]
pub fn forward_parallel(
    semantic: ArrayView1<f64>,
    matching: ArrayView2<f64>,
    alignment: ArrayView1<f64>,
    weights: &Weights,
    maximum_speed: usize,
) -> Array2<u8> {
    let evaluate = {
        let semantic = rankdata(semantic);
        let alignment = rankdata(alignment);

        let δ = maximum_speed as f64;
        let μ = semantic.len() as f64 / alignment.len() as f64;

        move |a: usize, v: usize, s: usize, p: usize| {
            let semantic = if (s as f64) < μ {
                1. - semantic[v]
            } else {
                1. - 0.05 * semantic[v]
            };
            let matching = matching[[v, s]];

            let s = s as f64;
            let p = p as f64;

            let τ = 1. + (δ - 1.) * alignment[a];

            let alignment = (s - τ).powi(2) / (δ - 1.).powi(2);
            let acceleration = (s - p).powi(2) / (δ - 1.).powi(2);

            weights.semantic * semantic
                + weights.matching * matching
                + weights.alignment * alignment
                + weights.acceleration * acceleration
        }
    };

    let forward =
        Array::from_shape_simple_fn([alignment.len(), semantic.len()], || {
            AtomicU8::new(0)
        });

    let mut current =
        Array::from_shape_simple_fn(semantic.len(), || AtomicF64::new(f64::INFINITY));
    let mut previous =
        Array::from_shape_simple_fn(semantic.len(), || AtomicF64::new(f64::INFINITY));

    let mu = (semantic.len() / alignment.len()) as u8;

    current[0].store(0.0, Relaxed);
    forward[[0, 0]].store(mu, Relaxed);

    for a in 1..alignment.len() {
        swap(&mut previous, &mut current);
        current.for_each(|c| c.store(f64::INFINITY, Relaxed));

        let remaining = alignment.len() - a;

        let v_min = usize::max(
            // Moving forward at minimum speed.
            a,
            // Moving backward at maximum speed.
            semantic.len().saturating_sub(remaining * maximum_speed),
        );
        let v_max = usize::min(
            // Moving forward at maximum speed.
            a * maximum_speed,
            // Moving backward at minimum speed.
            semantic.len().saturating_sub(remaining),
        );

        (v_min..=v_max).into_par_iter().for_each(|v| {
            let s_min = 1;
            let s_max = usize::min(maximum_speed, v);

            let mut min = f64::INFINITY;
            let mut argmin = 0;

            for s in s_min..=s_max {
                let sum = previous[v - s].load(Relaxed);
                if sum.is_finite() {
                    let p = forward[[a - 1, v - s]].load(Relaxed) as usize;
                    let value = evaluate(a, v, s, p) + sum;

                    if value < min {
                        min = value;
                        argmin = s;
                    }
                }
            }

            current[v].store(min, Relaxed);
            forward[[a, v]].store(argmin as u8, Relaxed);
        });
    }

    forward.map(|n| n.load(Relaxed))
}

#[must_use]
pub fn backward(
    forward: ArrayView2<u8>,
    input_length: usize,
    output_length: usize,
) -> Backward {
    let mut a = output_length - 1;
    let mut v = input_length - 1;

    let mut index = Array::zeros(output_length);
    let mut speed = Array::zeros(output_length);

    index[a] = v;
    speed[a] = forward[[a, v]] as usize;

    while a > 0 {
        v -= speed[a];
        a -= 1;

        index[a] = v;
        speed[a] = forward[[a, v]] as usize;
    }

    assert!(a == 0, "Expected to reach the beginning of the audio.");
    assert!(v == 0, "Expected to reach the beginning of the video.");

    Backward { index, speed }
}

/// Assign ranks to data, breaking ties with competition ranking.
///
/// See <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html>
/// and <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html>.
#[expect(clippy::cast_precision_loss, reason = "The values are small.")]
fn rankdata(scores: ArrayView1<f64>) -> Array1<f64> {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|left, right| f64::partial_cmp(left, right).unwrap());

    let n = scores.len() as f64;

    scores.mapv(|score| {
        let rank = sorted.partition_point(|&probe| probe < score);
        (rank as f64) / (n - 1.)
    })
}

/// Implements atomic floating-point numbers as a wrapper around [`AtomicU64`].
///
/// This is a workaround for the lack of a native `AtomicF64`.
#[repr(transparent)]
pub struct AtomicF64(AtomicU64);

impl AtomicF64 {
    /// See [`AtomicU64::new`].
    pub fn new(value: f64) -> Self {
        let bits = value.to_bits();
        Self(AtomicU64::new(bits))
    }

    /// See [`AtomicU64::store`].
    pub fn store(&self, value: f64, ordering: Ordering) {
        let bits = value.to_bits();
        self.0.store(bits, ordering);
    }

    /// See [`AtomicU64::load`].
    pub fn load(&self, ordering: Ordering) -> f64 {
        let bits = self.0.load(ordering);
        f64::from_bits(bits)
    }
}

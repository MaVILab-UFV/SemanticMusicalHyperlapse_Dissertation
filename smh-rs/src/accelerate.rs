use std::cmp::Ordering;
use std::mem::swap;

use ndarray::prelude::*;

#[derive(Copy, Clone, Debug, Default, PartialEq, serde::Deserialize)]
pub struct Weights {
    pub semantic: f64,
    pub matching: f64,
    pub alignment: f64,
    pub speed: f64,
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
pub fn forward(
    semantic: ArrayView1<f64>,
    matching: ArrayView2<f64>,
    alignment: ArrayView1<f64>,
    weights: &Weights,
    maximum_speed: usize,
) -> Array2<u8> {
    let evaluate = {
        let δ = maximum_speed as f64;
        let τ = 200.;

        let semantic = rankdata(semantic);
        let alignment = rankdata(alignment);

        move |a, v, s| {
            let semantic = -τ * semantic[v];
            let matching = τ * matching[[v, s]];

            let alignment = (s as f64 - δ * alignment[a]).powi(2).min(τ);
            let speed = (s as f64).powi(2).min(τ);

            weights.semantic * semantic
                + weights.matching * matching
                + weights.alignment * alignment
                + weights.speed * speed
        }
    };

    let mut forward = Array::zeros([alignment.len(), semantic.len()]);

    let mut current = Array::from_elem(semantic.len(), f64::INFINITY);
    let mut previous = Array::from_elem(semantic.len(), f64::INFINITY);

    // Assume an initial jump from `(-1, -1)` to `(0, 0)`.
    current[0] = evaluate(0, 0, 1);
    forward[[0, 0]] = 1;

    for a in 1..alignment.len() {
        swap(&mut previous, &mut current);
        current.fill(f64::INFINITY);

        let remaining = alignment.len() - a;

        let lower = usize::max(
            // Moving forward at minimum speed.
            a,
            // Moving backward at maximum speed.
            semantic.len().saturating_sub(remaining * maximum_speed),
        );
        let upper = usize::min(
            // Moving forward at maximum speed.
            a * maximum_speed,
            // Moving backward at minimum speed.
            semantic.len().saturating_sub(remaining),
        );

        for v in lower..=upper {
            let minimum_speed = 1;
            let maximum_speed = usize::min(maximum_speed, v);

            for s in minimum_speed..=maximum_speed {
                if previous[v - s].is_finite() {
                    let score = {
                        let score = evaluate(a, v, s);
                        let average = previous[v - s];

                        let n = (a + 1) as f64;
                        average + (score - average) / n
                    };

                    if score < current[v] {
                        current[v] = score;
                        forward[[a, v]] = s as u8;
                    }
                }
            }
        }
    }

    forward
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

/// Assign ranks to data.
///
/// See <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html>
/// and <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html>.
#[expect(clippy::cast_precision_loss, reason = "The values are small.")]
fn rankdata(scores: ArrayView1<f64>) -> Array1<f64> {
    let n = scores.len();
    let mut rank = Array1::zeros(n);

    for i in 0..n {
        for j in i..n {
            match f64::partial_cmp(&scores[i], &scores[j]) {
                Some(Ordering::Greater) => {
                    rank[i] += 1.0;
                }
                Some(Ordering::Less) => {
                    rank[j] += 1.0;
                }
                _ => continue,
            };
        }
    }

    let n = scores.len() as f64;
    rank / (n - 1.)
}

import json
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
from scipy.stats import rankdata
from tqdm import trange


@dataclass
class Weights:
    semantic: float
    matching: float
    alignment: float
    acceleration: float


@dataclass
class Backward:
    index: np.ndarray
    speed: np.ndarray


def forward(
    semantic: np.ndarray,
    matching: np.ndarray,
    alignment: np.ndarray,
    weights: Weights,
    maximum_speed: int,
) -> np.ndarray:
    δ = float(maximum_speed)
    μ = len(semantic) / len(alignment)

    semantic_rank = rankdata(semantic, method="min") / len(semantic)
    alignment_rank = rankdata(alignment, method="min") / len(alignment)

    n_semantic = len(semantic)
    n_alignment = len(alignment)

    forward_matrix = np.zeros((n_alignment, n_semantic), dtype=np.uint8)

    current = np.full(n_semantic, np.inf)
    previous = np.full(n_semantic, np.inf)

    def evaluate(a: int, v: int, s: int, p: int) -> float:
        if s < μ:
            semantic_score = 1.0 - semantic_rank[v]
        else:
            semantic_score = 1.0 - 0.05 * semantic_rank[v]

        matching_score = matching[v, s]

        τ = 1.0 + (δ - 1.0) * alignment_rank[a]

        alignment_score = (s - τ) ** 2 / (δ - 1.0) ** 2
        acceleration_score = (s - p) ** 2 / (δ - 1.0) ** 2

        return (
            weights.semantic * semantic_score
            + weights.matching * matching_score
            + weights.alignment * alignment_score
            + weights.acceleration * acceleration_score
        )

    current[0] = evaluate(0, 0, 1)
    forward_matrix[0, 0] = 1

    for a in trange(1, n_alignment):
        previous, current = current, previous
        current.fill(np.inf)

        remaining = n_alignment - a

        lower = max(a, n_semantic - remaining * maximum_speed)
        upper = min(a * maximum_speed, n_semantic - remaining)

        for v in range(lower, upper + 1):
            minimum_speed = 1
            maximum_speed_v = min(maximum_speed, v)

            for s in range(minimum_speed, maximum_speed_v + 1):
                if np.isfinite(previous[v - s]):
                    p = forward_matrix[a - 1, v - s]
                    value = previous[v - s] + evaluate(a, v, s, p)

                    if value < current[v]:
                        current[v] = value
                        forward_matrix[a, v] = s

    return forward_matrix


def backward(forward: np.ndarray, input_length: int, output_length: int) -> Backward:
    a = output_length - 1
    v = input_length - 1

    index = np.zeros(output_length, dtype=int)
    speed = np.zeros(output_length, dtype=int)

    index[a] = v
    speed[a] = forward[a, v]

    while a > 0:
        v -= speed[a]
        a -= 1

        index[a] = v
        speed[a] = forward[a, v]

    assert a == 0, "Expected to reach the beginning of the audio."
    assert v == 0, "Expected to reach the beginning of the video."

    return Backward(index=index, speed=speed)


@click.command()
@click.option(
    "--semantic",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the `.npy` file containing the video semantic features.",
)
@click.option(
    "--matching",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the `.npy` file containing the video transition matrix.",
)
@click.option(
    "--alignment",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the `.npy` file containing the audio features.",
)
@click.option(
    "--weights",
    default="weights.json",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the `.json` file containing the weights of each optimization term.",
)
@click.option(
    "--maximum_speed",
    default=20,
    show_default=True,
    type=int,
    help="The maximum momentary playback speed.",
)
@click.argument(
    "output", type=click.Path(writable=True, dir_okay=False, path_type=Path)
)
def main(semantic, matching, alignment, weights, maximum_speed, output):
    """
    Runs the “Multimodal Frame Sampling Algorithm for Semantic Hyperlapses with Musical Alignment”.
    """

    try:
        with open(weights) as f:
            weights_data = json.load(f)
            weights_data = Weights(**weights_data)
    except Exception as err:
        raise click.ClickException("Failed to read the `weights` file") from err

    try:
        semantic_data = np.load(semantic)
    except Exception as err:
        raise click.ClickException("Failed to read the `semantic` file") from err

    try:
        alignment_data = np.load(alignment)
    except Exception as err:
        raise click.ClickException("Failed to read the `alignment` file") from err

    try:
        matching_data = np.load(matching)
    except Exception as err:
        raise click.ClickException("Failed to read the `matching` file") from err

    forward_result = forward(
        semantic=semantic_data,
        matching=matching_data,
        alignment=alignment_data,
        weights=weights_data,
        maximum_speed=maximum_speed,
    )

    backward_result = backward(
        forward=forward_result,
        input_length=len(semantic_data),
        output_length=len(alignment_data),
    )

    try:
        outputs = {
            "index": backward_result.index.astype(np.uint64),
            "speed": backward_result.speed.astype(np.uint64),
            "audio": alignment_data,
            "video": semantic_data,
        }
        np.savez(output, **outputs)
    except Exception as err:
        raise click.ClickException("Failed to write the `output` file") from err


if __name__ == "__main__":
    main()

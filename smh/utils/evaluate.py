import json
import logging
from logging import warning
from pathlib import Path
from pprint import pprint

import click
import cv2
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(level=logging.DEBUG, format="[%(filename)s:%(lineno)s] %(message)s")


def instability(video: Path, index: np.ndarray, window: int = 7) -> float:
    """Calculates the instability index of the video according to Silva et al. (2016).

    The implementation is based on the MATLAB code provided by the authors [1].

    Parameters
    ----------
    video : Path
        The path to the video file.
    index : ndarray
        The indices of the selected frames.
    window : int, optional
        The size of the sliding window. Defaults to 7 [2].

    Returns
    -------
    instability : float
        The instability index of the video.

    References
    ----------
    [1]: <https://github.com/verlab/SemanticFastForward_ECCVW_2016/blob/b22bd3/Util/GetInstabilityIndex.m#L23>
    [2]: <https://github.com/verlab/2019-tpami-silva-multimodalSparseSampling-code/blob/bfeea7/scripts/run_instability.m#L9>
    """

    capture = cv2.VideoCapture(str(video))

    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    buffer = torch.zeros((window, height, width), device="cuda")
    scores = 0.0

    for t, cursor in enumerate(tqdm(index, "Instability", leave=False)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, cursor)

        is_ok, image = capture.read()
        if not is_ok:
            warning("Failed to read frame #%d", cursor)
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        buffer[t % window] = torch.from_numpy(image).cuda().double()

        if t >= window:
            scores += buffer.std(dim=0).mean().item()

    return scores / (t - window)


def semantic(video: np.ndarray, index: np.ndarray) -> float:
    """
    Computes the efficiency of the selection: the semantic scores collected compared to
    the maximum possible for the same number of frames.
    """

    expected = np.sum(np.sort(video)[-len(index) :])
    observed = np.sum(video[index])

    score = observed / expected
    return score


def correlation(audio: np.ndarray, speed: np.ndarray) -> float:
    """
    Computes the Spearman correlation between the speed-up and the loudness.
    """

    if len(audio) > len(speed):
        audio = audio[: len(speed)]
    else:
        speed = speed[: len(audio)]

    score, _ = stats.spearmanr(audio, speed)
    return score if np.isfinite(score) else 0.0


def discontinuity(video: np.ndarray, index: np.ndarray, speed: np.ndarray) -> float:
    """
    See <https://arxiv.org/pdf/2009.11063.pdf#page=5>.
    """

    ratio = len(video) / len(index)
    score = np.sqrt(
        np.mean(
            (speed - ratio) ** 2,
        ),
    )

    return score


def rmssd(speed: np.ndarray) -> float:
    """
    See <https://search.r-project.org/CRAN/refmans/psych/html/mssd.html>.
    """

    score = np.sqrt(
        np.mean(
            np.diff(speed) ** 2,
        ),
    )
    return score


@click.command()
@click.argument("video", type=Path)
@click.argument("input", type=Path)
@click.argument("output", type=Path, required=False)
@logging_redirect_tqdm()
def main(video: Path, input: Path, output: Path | None):
    result = np.load(input)

    metrics = {
        "Semantic": semantic(
            result["video"],
            result["index"],
        ),
        "Correlation": correlation(
            result["audio"],
            result["speed"],
        ),
        "Discontinuity": discontinuity(
            result["video"],
            result["index"],
            result["speed"],
        ),
        "Instability": instability(
            video,
            result["index"],
        ),
        "RMSSD": rmssd(
            result["speed"],
        ),
    }

    pprint(metrics)

    output = output or input.with_suffix(".json")
    with output.open("w") as file:
        json.dump(metrics, file)


if __name__ == "__main__":
    main()

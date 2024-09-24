import logging
from collections import deque
from itertools import count
from logging import debug, warning
from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(level=logging.DEBUG, format="[%(filename)s:%(lineno)s] %(message)s")


def find_homography(
    src_keypoints: list[cv2.KeyPoint],
    src_descriptors: np.ndarray,
    tgt_keypoints: list[cv2.KeyPoint],
    tgt_descriptors: np.ndarray,
) -> tuple[np.ndarray, float]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(src_descriptors, tgt_descriptors)

    if len(matches) < 10:
        raise ValueError("Pair only has %d matches.", len(matches))

    query_keypoints = np.array([src_keypoints[m.queryIdx].pt for m in matches])
    train_keypoints = np.array([tgt_keypoints[m.trainIdx].pt for m in matches])

    # See <https://opencv.org/blog/evaluating-opencvs-new-ransacs/>.
    homography, mask = cv2.findHomography(
        query_keypoints, train_keypoints, cv2.USAC_MAGSAC, 3.0
    )

    if homography is None:
        raise ValueError("Failed to find homography.")

    # region: Compute the reprojection error.

    train_keypoints = train_keypoints[mask]
    query_keypoints = query_keypoints[mask]

    error = train_keypoints - cv2.perspectiveTransform(query_keypoints, homography)
    error = np.linalg.norm(error, axis=2).mean()

    # endregion

    return homography, error


@click.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path, required=False)
@logging_redirect_tqdm()
def main(input: Path, output: Path | None):
    capture = cv2.VideoCapture(str(input))

    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    center = np.array([width, height]) / 2.0

    window = 32
    recent = deque(maxlen=window)

    errors = list()
    scores = list()
    homographies = list()

    orb = cv2.ORB_create()

    for t in tqdm(count(), desc="Processing", total=length):
        is_ok, image = capture.read()
        if not is_ok:
            warning("Failed to read frame #%d", t)
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        current_keypoints, current_descriptors = orb.detectAndCompute(image, None)

        current_errors = np.full((window + 1,), np.inf)
        current_scores = np.full((window + 1,), np.inf)
        current_homographies = np.full((window + 1, 3, 3), np.nan)

        for dt, (previous_keypoints, previous_descriptors) in enumerate(
            recent, start=1
        ):
            try:
                assert previous_descriptors is not None
                assert current_descriptors is not None

                homography, error = find_homography(
                    previous_keypoints,
                    previous_descriptors,
                    current_keypoints,
                    current_descriptors,
                )
            except (ValueError, AssertionError) as e:
                warning("Failed to find homography for frame #%d and offset #%d", t, dt)
                debug("Failure caused by an exception.", exc_info=e)
            else:
                u = center.reshape(1, 1, 2)
                v = cv2.perspectiveTransform(u, homography)

                current_errors[dt] = error
                current_scores[dt] = np.linalg.norm(u - v)
                current_homographies[dt] = homography

        errors.append(current_errors)
        scores.append(current_scores)
        homographies.append(current_homographies)

        recent.appendleft((current_keypoints, current_descriptors))

    errors = np.array(errors)
    scores = np.array(scores)
    homographies = np.array(homographies)

    diagonal = np.sqrt(width**2 + height**2)
    scores = np.where(errors < 0.1 * diagonal, scores / diagonal, 1.0)

    output = output or input.with_suffix(".homography.npy")
    np.save(output, scores)


if __name__ == "__main__":
    main()

import logging
from itertools import count
from logging import debug, info, warning
from pathlib import Path

import click
import cv2
import numpy as np
from scipy.linalg import fractional_matrix_power
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(level=logging.DEBUG, format="[%(filename)s:%(lineno)s] %(message)s")


SEGMENT_LENGTH: int = 5
"""Interval between master frames."""

COVERAGE_THRESHOLD: float = 0.999
"""Minimum coverage required to pass `test_image_coverage`."""

NUM_MAX_IMAGES_TO_RECONSTRUCT: int = 30
"""Number of neighboring frames used to reconstruct a image."""

INPAINT_RADIUS = 3
"""Size of the inpainting kernel."""

MINIMUM_KEYPOINTS: int = 50
"""Minimum number of keypoints required to compute a homography."""

CROP_PORTION: float = 0.05
DROP_PORTION: float = 0.2

# region: Computing keypoints and descriptors.


def find_homography(src_detections, tgt_detections) -> tuple[np.ndarray, int]:
    src_keypoints, src_descriptors = src_detections
    tgt_keypoints, tgt_descriptors = tgt_detections

    if src_descriptors is None or len(src_keypoints) < MINIMUM_KEYPOINTS:
        raise ValueError("Not enough keypoints in the source image.")

    if tgt_descriptors is None or len(tgt_keypoints) < MINIMUM_KEYPOINTS:
        raise ValueError("Not enough keypoints in the target image.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(src_descriptors, tgt_descriptors)

    if len(matches) < MINIMUM_KEYPOINTS:
        raise ValueError("Pair only has %d matches.", len(matches))

    query_keypoints = np.array([src_keypoints[m.queryIdx].pt for m in matches])
    train_keypoints = np.array([tgt_keypoints[m.trainIdx].pt for m in matches])

    # See <https://opencv.org/blog/evaluating-opencvs-new-ransacs/>.
    affine, mask = cv2.estimateAffinePartial2D(query_keypoints, train_keypoints)
    if affine is None:
        raise ValueError("Failed to find the transformation.")

    homography = np.eye(3)
    homography[:2] = affine

    if not 1e-4 <= np.linalg.det(homography) <= 1e4:
        raise ValueError("Bad homography matrix.")

    inliers = np.count_nonzero(mask)

    return homography, inliers


# endregion


def choose_master_frames(selected_frames, detections):
    length = len(selected_frames)

    scores = np.zeros(length)
    master = []

    for src in trange(length, desc="Master frames"):
        for tgt in range(src + 1, min(length, src + SEGMENT_LENGTH)):
            src_detections = detections[selected_frames[src]]
            tgt_detections = detections[selected_frames[tgt]]

            try:
                _, inliers = find_homography(src_detections, tgt_detections)
            except ValueError as e:
                warning("No homography for %d and %d.", src, tgt)
                debug("Failure caused by an exception.", exc_info=e)
            else:
                scores[src] += inliers
                scores[tgt] += inliers

    master.append(0)

    for start in range(1, length - 1, SEGMENT_LENGTH):
        argmax = start + np.argmax(scores[start : start + SEGMENT_LENGTH])
        master.append(argmax)

    master.append(-1)

    return np.array(master)


@click.command()
@click.argument("video", type=Path)
@click.argument("input", type=Path)
@click.argument("output", type=Path, required=False)
@logging_redirect_tqdm()
def main(video: Path, input: Path, output: Path) -> None:
    capture = cv2.VideoCapture(str(video))

    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_fps = int(capture.get(cv2.CAP_PROP_FPS))
    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # region: Compute keypoints and descriptors.

    orb = cv2.ORB_create()
    detections = list()

    for t in tqdm(count(), "Keypoints", video_length):
        is_ok, image = capture.read()
        if not is_ok:
            warning("Failed to read frame #%d", t)
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        current_detections = orb.detectAndCompute(image, None)
        detections.append(current_detections)

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # endregion

    selected_frames = np.load(input)["index"]
    master_indices = choose_master_frames(selected_frames, detections)

    crop_area = (
        cx0 := int(video_width * CROP_PORTION),
        cy0 := int(video_height * CROP_PORTION),
        cx1 := video_width - cx0,
        cy1 := video_height - cy0,
    )
    drop_area = (
        dx0 := int(video_width * DROP_PORTION),
        dy0 := int(video_height * DROP_PORTION),
        dx1 := video_width - dx0,
        dy1 := video_height - dy0,
    )

    output = output or input.with_suffix(".mp4")
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (cx1 - cx0, cy1 - cy0),
    )

    for src, tgt in tqdm(
        zip(master_indices[:-1], master_indices[1:]), "Processing", len(master_indices)
    ):
        # region: Process the source master frame.

        capture.set(cv2.CAP_PROP_POS_FRAMES, selected_frames[src])

        is_ok, image = capture.read()
        if is_ok:
            info("Master frame #%d.", selected_frames[src])
            writer.write(image[cy0:cy1, cx0:cx1])
        else:
            warning("Failed to read frame #%d.", selected_frames[src])
            break

        # endregion
        # region: Process the intermediate frames.

        src_detections = detections[selected_frames[src]]
        tgt_detections = detections[selected_frames[tgt]]

        for mid in trange(src + 1, tgt, leave=False):
            capture.set(cv2.CAP_PROP_POS_FRAMES, selected_frames[mid])

            is_ok, image = capture.read()
            if not is_ok:
                warning("Failed to read frame #%d.", selected_frames[src])
                break

            mid_detections = detections[selected_frames[mid]]

            # The formula for `w` in the paper is wrong.
            weight = (mid - src) / (tgt - src)

            try:
                # See <https://github.com/verlab/SemanticFastForward_JVCI_2018/blob/master/AcceleratedVideoStabilizer/src/sequence_processing.cpp#L186>.
                homography = find_intermediate_homography(
                    mid_detections, src_detections, tgt_detections, weight
                )
                reconstructed_frame = cv2.warpPerspective(
                    image, homography, (video_width, video_height)
                )

                mask = np.full(
                    shape=(video_height, video_width), dtype=np.uint8, fill_value=255
                )
                reconstructed_mask = cv2.warpPerspective(
                    mask, homography, (video_width, video_height)
                )

                # See <https://github.com/verlab/SemanticFastForward_JVCI_2018/blob/master/AcceleratedVideoStabilizer/src/main.cpp#L640>.
                if test_mask_coverage(reconstructed_mask, crop_area):
                    info("Keep frame #%d.", mid)
                elif test_mask_coverage(reconstructed_mask, drop_area):
                    info("Reconstruct frame #%d.", mid)
                    reconstructed_frame = reconstruct_image(
                        image,
                        homography,
                        selected_frames[mid],
                        capture,
                        crop_area,
                        detections,
                    )
                else:
                    info("Replace frame #%d.", mid)
                    raise ValueError("Frame #%d has no coverage.", mid)
            except ValueError as e:
                warning("Failed to process intermediate frame.")
                debug("Failure caused by an exception.", exc_info=e)

                reconstructed_frame = image

            writer.write(reconstructed_frame[cy0:cy1, cx0:cx1])

        # endregion

    # region: Process the last master frame.

    capture.set(cv2.CAP_PROP_POS_FRAMES, selected_frames[tgt])

    is_ok, image = capture.read()
    if is_ok:
        info("Master frame #%d.", selected_frames[src])
        writer.write(image[cy0:cy1, cx0:cx1])
    else:
        warning("Failed to read frame #%d.", selected_frames[src])

    # endregion

    writer.release()
    capture.release()


def find_intermediate_homography(
    current_detections: list,
    src_detections: list,
    tgt_detections: list,
    weight: float,
) -> np.ndarray:
    """
    See <https://arxiv.org/pdf/1711.03473.pdf#page=10>.
    """

    try:
        homography, _ = find_homography(current_detections, src_detections)
        src_homography = fractional_matrix_power(homography, 1 - weight)
    except ValueError as e:
        warning("Failed to find homography for `src` frame.")
        debug("Failure caused by an exception.", exc_info=e)

        src_homography = np.eye(3)
        has_src_homography = False
    else:
        has_src_homography = True

    try:
        homography, _ = find_homography(current_detections, tgt_detections)
        tgt_homography = fractional_matrix_power(homography, weight)
    except ValueError as e:
        warning("Failed to find homography for `tgt` frame.")
        debug("Failure caused by an exception.", exc_info=e)

        tgt_homography = np.eye(3)
        has_tgt_homography = False
    else:
        has_tgt_homography = True

    if not has_src_homography and not has_tgt_homography:
        raise ValueError("Failed to find homography for both `src` and `tgt` frames.")

    homography = src_homography @ tgt_homography
    homography = np.real(homography)

    return homography


def reconstruct_image(
    # See <https://github.com/verlab/SemanticFastForward_JVCI_2018/blob/master/AcceleratedVideoStabilizer/src/image_reconstruction.cpp#L500>.
    image: np.ndarray,
    homography: np.ndarray,
    index: int,
    capture: cv2.VideoCapture,
    crop_area: tuple[int, int, int, int],
    detections: list,
) -> np.ndarray:
    h, w, _ = image.shape
    mask = np.full(shape=(h, w), dtype=np.uint8, fill_value=255)

    result_image = cv2.warpPerspective(image, homography, (w, h))
    result_mask = cv2.warpPerspective(mask, homography, (w, h))

    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    index = int(index)

    min_index = max(index - NUM_MAX_IMAGES_TO_RECONSTRUCT, 0)
    max_index = min(index + NUM_MAX_IMAGES_TO_RECONSTRUCT, length)

    capture.set(cv2.CAP_PROP_POS_FRAMES, min_index)

    src_frame_buffer = []
    tgt_frame_buffer = []

    for idx in range(min_index, max_index):
        is_ok, frame = capture.read()
        if not is_ok:
            warning("Failed to read frame #%d.", idx)
            break

        if idx < index:
            src_frame_buffer.append((idx, frame))
        elif idx > index:
            tgt_frame_buffer.append((idx, frame))

    while src_frame_buffer or tgt_frame_buffer:
        try:
            src_frame_index, src_frame = src_frame_buffer.pop()
            result_image, result_mask = warp_mask_crop(
                result_image,
                result_mask,
                src_frame,
                src_frame_index,
                detections,
            )
        except IndexError:
            pass
        except ValueError as e:
            warning("Failed to warp previous frame.")
            debug("Failure caused by an exception.", exc_info=e)

        try:
            tgt_frame_index, tgt_frame = tgt_frame_buffer.pop(0)
            result_image, result_mask = warp_mask_crop(
                result_image,
                result_mask,
                tgt_frame,
                tgt_frame_index,
                detections,
            )
        except IndexError:
            pass
        except ValueError as e:
            warning("Failed to warp posterior frame.")
            debug("Failure caused by an exception.", exc_info=e)

        if test_mask_coverage(result_mask, crop_area):
            return result_image

    raise ValueError("Failed to cover the frame.")


def warp_mask_crop(
    partial_image: np.ndarray,
    partial_mask: np.ndarray,
    neighbor_image: np.ndarray,
    neighbor_index: int,
    detections: list,
) -> tuple[np.ndarray, np.ndarray]:
    orb = cv2.ORB_create()

    neighbor_detected = detections[neighbor_index]
    partial_detected = orb.detectAndCompute(partial_image, None)

    try:
        homography, _ = find_homography(neighbor_detected, partial_detected)
    except ValueError as e:
        warning("Failed to find homography.")
        debug("Failure caused by an exception.", exc_info=e)
        raise

    h, w, _ = neighbor_image.shape
    result_mask = np.full(shape=(h, w), dtype=np.uint8, fill_value=255)

    result_image = cv2.warpPerspective(neighbor_image, homography, (w, h))
    result_mask = cv2.warpPerspective(result_mask, homography, (w, h))

    # region: Inpainting mask.

    AB = result_mask & partial_mask
    BA = result_mask & ~AB

    AB = cv2.morphologyEx(
        AB,
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (INPAINT_RADIUS, INPAINT_RADIUS)),
    )
    BA = cv2.morphologyEx(
        BA,
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (INPAINT_RADIUS, INPAINT_RADIUS)),
    )

    inpaint_mask = AB & BA

    # endregion

    result_mask = partial_mask | result_mask
    result_image = np.where(partial_mask[..., np.newaxis], partial_image, result_image)

    result_image = cv2.inpaint(
        result_image, inpaint_mask, 2 * INPAINT_RADIUS, cv2.INPAINT_TELEA
    )

    return result_image, result_mask


def test_mask_coverage(
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    threshold: float = COVERAGE_THRESHOLD,
) -> bool:
    """
    Tests if the given `mask` covers a region of interest (`roi`, as XYXY coordinates)
    according to a `threshold`.
    """

    x0, y0, x1, y1 = roi

    coverage = np.count_nonzero(mask[y0:y1, x0:x1])
    required = threshold * (y1 - y0) * (x1 - x0)

    return coverage >= required


if __name__ == "__main__":
    main()

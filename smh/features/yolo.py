import logging
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from ultralytics import YOLO

logging.basicConfig(level=logging.DEBUG, format="[%(filename)s:%(lineno)s] %(message)s")

CLASSES = {"person"}


@click.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path, required=False)
@logging_redirect_tqdm()
def main(input: Path, output: Path | None) -> None:
    scores = []

    model = YOLO("yolov10x.pt")

    for prediction in tqdm(
        model(input, save=False, stream=True, verbose=False),
    ):
        score = 0.0

        width, height = prediction.orig_shape

        for cls, confidence, xywh in zip(
            prediction.boxes.cls,
            prediction.boxes.conf,
            prediction.boxes.xywh,
        ):
            cls = cls.item()
            confidence = confidence.item()
            x, y, w, h = xywh.tolist()

            if model.names[cls] not in CLASSES:
                continue

            cx = (x + w * 0.5) / width
            cy = (y + h * 0.5) / height

            area = w * h
            centrality = gaussian(cx, 0.5, 0.25) * gaussian(cy, 0.5, 0.25)

            score += area * confidence * centrality

        scores.append(score)

    scores = np.array(scores)

    output = output or input.with_suffix(".semantic.npy")
    np.save(output, scores)


def gaussian(x: np.ndarray, μ: float, σ: float) -> np.ndarray:
    z = (x - μ) / σ
    return np.exp(-0.5 * z**2)


if __name__ == "__main__":
    main()

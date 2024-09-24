import logging
from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(level=logging.DEBUG, format="[%(filename)s:%(lineno)s] %(message)s")


@click.command()
@click.argument("video", type=Path)
@click.argument("input", type=Path)
@click.argument("output", type=Path, required=False)
@logging_redirect_tqdm()
def main(video: Path, input: Path, output: Path):
    index = np.load(input)["index"]

    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened()

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output = output or input.with_suffix(".mp4")
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )

    for at in tqdm(index, "Rendering"):
        capture.set(cv2.CAP_PROP_POS_FRAMES, at)

        is_ok, image = capture.read()
        assert is_ok

        writer.write(image)

    writer.release()
    capture.release()


if __name__ == "__main__":
    main()

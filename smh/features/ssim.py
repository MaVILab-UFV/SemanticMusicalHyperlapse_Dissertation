import logging
from itertools import count
from logging import warning
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from kornia.metrics import ssim
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(level=logging.DEBUG, format="[%(filename)s:%(lineno)s] %(message)s")


@click.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path, required=False)
@logging_redirect_tqdm()
def main(input: Path, output: Path | None):
    capture = cv2.VideoCapture(str(input))

    width = 320
    height = 180

    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    window = 31

    recent = torch.zeros((window, 3, height, width), device="cuda")
    scores = np.zeros((length, window))

    for t in tqdm(count(), desc="Processing", total=length):
        is_ok, current = capture.read()
        if not is_ok:
            warning("Failed to read frame #%d", t)
            break

        current = cv2.resize(current, (width, height))
        current = torch.from_numpy(current / 255.0).permute(2, 0, 1).cuda()

        # Although comparing the current frame to itself is redundant, it is
        # more convenient to have the second dimension of `penalties` simply
        # be the time delta.
        recent[-t % window] = current

        current = current.expand(window, -1, -1, -1)

        scores[t] = (
            ssim(current, recent, window_size=11)
            .mean(dim=(1, 2, 3))
            .roll(t)
            .numpy(force=True)
        )

    transitions = 1.0 - scores

    output = output or input.with_suffix(".ssim.npy")
    np.save(output, transitions)


if __name__ == "__main__":
    main()

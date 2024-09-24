import io
from itertools import count, repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path
from subprocess import PIPE, Popen

import click
import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

sns.set_theme(context="paper", palette="Set2")


def plot(data) -> bytes:
    at, frame, index, speed, audio_scores, video_scores = data

    accelerated = video_scores[index]

    fig = plt.figure(figsize=(16, 9), dpi=120.0, layout="tight")
    grid = plt.GridSpec(3, 3, figure=fig)

    ax = fig.add_subplot(grid[0, 2:])
    ax.set_title("Speed")

    ax.plot(speed)
    ax.axvline(at, color="C1", label=f"{speed[at]:.0f}×")
    plt.legend(loc="lower right")

    ax = fig.add_subplot(grid[1, 2:])
    ax.set_title("Loudness")

    ax.plot(audio_scores)
    ax.axvline(at, color="C1", label=f"{audio_scores[at]:.1f}")
    plt.legend(loc="lower right")

    ax = fig.add_subplot(grid[2, 2:])
    ax.set_title("Semantic (Accelerated)")

    ax.plot(accelerated)
    ax.axvline(at, color="C1", label=f"{accelerated[at]:.1f}")
    plt.legend(loc="lower right")

    ax = fig.add_subplot(grid[2, :2])
    ax.set_title("Semantic (Original)")

    ax.plot(video_scores)
    ax.axvline(index[at], color="C1", label=f"{video_scores[index[at]]:.1f}")
    plt.legend(loc="lower right")

    ax = fig.add_subplot(grid[:2, :2])
    ax.set_title(f"Video @ {np.mean(speed):.1f}×")

    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    ax.set_xticks([])
    ax.set_yticks([])

    buf = io.BytesIO()

    # See <https://stackoverflow.com/a/5394042>.
    fig.savefig(buf, format="rgba", dpi=fig.dpi)
    plt.close(fig)

    return buf.getvalue()


@click.command()
@click.option("--video", type=Path)
@click.option("--audio", type=Path)
@click.argument("input", type=Path)
@click.argument("output", type=Path, required=False)
def main(
    video: Path,
    audio: Path,
    input: Path,
    output: Path,
):
    with np.load(input) as data:
        index: np.ndarray = data["index"]  # ndarray[uint64]
        speed: np.ndarray = data["speed"]  # ndarray[uint64]

        audio_scores: np.ndarray = data["audio"]  # ndarray[float64]
        video_scores: np.ndarray = data["video"]  # ndarray[float64]

    output = output or input.with_suffix(".mp4")
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        "1920x1080",
        "-r",
        "30",
        "-i",
        "-",
        "-i",
        audio,
        "-c:a",
        "copy",
        "-c:v",
        "hevc_nvenc",
        "-cq:v",
        "18",
        output,
    ]

    def capture() -> iter:
        cap = cv2.VideoCapture(str(video))

        for at in index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, at)

            is_ok, frame = cap.read()
            assert is_ok, "Failed to read frame."

            yield frame

    with (
        Pool(cpu_count()) as pool,
        Popen(args, stdin=PIPE) as writer,
    ):
        iterable = zip(
            count(),
            capture(),
            repeat(index),
            repeat(speed),
            repeat(audio_scores),
            repeat(video_scores),
        )

        for buf in tqdm(pool.imap(plot, iterable), leave=True, total=len(index)):
            writer.stdin.write(buf)

        writer.communicate()


if __name__ == "__main__":
    main()

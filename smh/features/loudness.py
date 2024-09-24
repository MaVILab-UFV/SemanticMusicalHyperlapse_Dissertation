from pathlib import Path
from typing import Optional

import click
import librosa
import numpy as np
from essentia import standard as es


@click.command()
@click.option("--fps", type=click.Choice(["30", "60"]), default="30")
@click.argument("input", type=Path)
@click.argument("output", type=Path, required=False)
def main(input: Path, output: Optional[Path], fps: str):
    match fps:
        case "30":
            hop = 1001 / 30000
        case "60":
            hop = 1001 / 60000

    y, sr = librosa.load(input, mono=False, sr=44100)

    if y.ndim == 1:
        y = np.vstack([y, y])

    ebu_r_128 = es.LoudnessEBUR128(hopSize=hop, sampleRate=sr, startAtZero=True)
    loudness, _, _, _ = ebu_r_128(y.T)

    output = output or input.with_suffix(".npy")
    np.save(output, loudness.astype(np.float64))


if __name__ == "__main__":
    main()

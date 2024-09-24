from pathlib import Path

import click
import h5py
import numpy as np
from scipy.io import loadmat


@click.command()
@click.argument("mat", type=Path)
@click.argument("npy", type=Path, required=False)
def main(mat: Path, npy: Path) -> None:
    try:
        data = loadmat(mat)
    except NotImplementedError:
        data = h5py.File(mat, "r")

    semantic = np.squeeze(data["total_values"])

    npy = npy or mat.with_suffix(".npy")
    np.save(npy, semantic.astype(np.float64))


if __name__ == "__main__":
    main()

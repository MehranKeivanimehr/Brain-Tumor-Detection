"""Extract all feature sets for the slice dataset.

Usage: python scripts/03_extract_features.py [--only classical,scattering,resnet50,densenet121]
Writes data/features/<name>.npy aligned with data/dataset/slices.csv rows.
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"


def _scatter_chunk(chunk):
    from btmri.features.scattering import batch_scattering

    return batch_scattering(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="classical,scattering,resnet50,densenet121")
    args = ap.parse_args()
    FEAT.mkdir(parents=True, exist_ok=True)
    images = np.load(ROOT / "data" / "dataset" / "images.npy", mmap_mode="r")
    print(f"{len(images)} slices")
    for name in args.only.split(","):
        out = FEAT / f"{name}.npy"
        if out.exists():
            print(f"{name}: exists, skipping")
            continue
        t = time.time()
        if name == "classical":
            from btmri.features.classical import batch_classical

            F = batch_classical(np.asarray(images))
        elif name == "scattering":
            chunks = np.array_split(np.asarray(images), 64)
            with mp.Pool(4) as pool:
                F = np.concatenate(pool.map(_scatter_chunk, chunks))
        else:
            from btmri.features.cnn import batch_cnn

            F = batch_cnn(np.asarray(images), name)
        assert len(F) == len(images) and np.isfinite(F).all()
        np.save(out, F.astype(np.float32))
        print(f"{name}: {F.shape} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()

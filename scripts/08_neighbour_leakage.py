"""Why do slice-level splits inflate accuracy? Look at nearest neighbours.

For each test slice, find its nearest training slice in feature space under
the slice-level and the patient-level split, and record whether it belongs to
the same patient and how far apart the two slices are. Also report the
accuracy of a 1-nearest-neighbour "classifier" that just copies that label.

Usage: python scripts/08_neighbour_leakage.py --features resnet50
Outputs: results/neighbour_leakage.json, data/outputs/neighbours_<features>.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from btmri.heads import Standardizer
from btmri.splits import patient_folds, slice_folds

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="resnet50")
    args = ap.parse_args()
    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    keep = t.label.values >= 0
    rows = np.flatnonzero(keep)
    t = t[keep].reset_index(drop=True)
    X = np.load(ROOT / "data" / "features" / f"{args.features}.npy")[rows].astype(np.float64)
    y, g, z = t.label.values.astype(int), t.patient.values, t.z_mm.values
    out, saved = {"features": args.features}, {}
    for protocol, folds in (("slice", slice_folds(y, 5, 0)), ("patient", patient_folds(y, g, 5, 0))):
        tr, te = folds[0]
        std = Standardizer().fit(X[tr])
        nn = NearestNeighbors(n_neighbors=1).fit(std.transform(X[tr]))
        _, j = nn.kneighbors(std.transform(X[te]))
        nb = tr[j[:, 0]]
        same = g[nb] == g[te]
        dz = np.abs(z[nb] - z[te])
        out[protocol] = {
            "test_slices": int(len(te)),
            "nn_same_patient": float(same.mean()),
            "nn_same_patient_within_6mm": float((same & (dz <= 6.01)).mean()),
            "median_dz_mm_if_same_patient": float(np.median(dz[same])) if same.any() else None,
            "one_nn_accuracy": float((y[nb] == y[te]).mean()),
        }
        saved[f"{protocol}_test"], saved[f"{protocol}_nn"] = rows[te], rows[nb]
    with open(ROOT / "results" / "neighbour_leakage.json", "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(ROOT / "data" / "outputs" / f"neighbours_{args.features}.npz", **saved)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

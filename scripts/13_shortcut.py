"""A controlled "Frankenstein dataset": what a class-dependent acquisition cue does.

In the public Kaggle tumor / no-tumor corpora the two classes come from
different sources (and file formats). Here the same effect is emulated on
UPENN-GBM, where it is otherwise absent: a fraction p of the *tumor-free*
slices is re-encoded as JPEG (quality 50), tumor slices are left untouched.
For each p, a model is trained and conformally calibrated with patient-level
splits in that biased world, then evaluated

  * in the biased world (what a benchmark built from such data reports), and
  * in the clean world (the same unseen patients, no JPEG cue).

Usage:
  python scripts/13_shortcut.py extract --features resnet50   (features of JPEG'd negatives)
  python scripts/13_shortcut.py run --features resnet50 --head elm
Output: results/shortcut.csv
"""

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from btmri import conformal as cp
from btmri.heads import RidgeHead, select_lambda_lopo
from btmri.metrics import binary_metrics
from btmri.splits import patient_folds

ROOT = Path(__file__).resolve().parents[1]
QUALITY = 50
LAMBDAS = np.logspace(0, 4, 5)


def jpeg(img: np.ndarray, quality: int = QUALITY) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())))


def extract(features: str):
    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    neg = np.flatnonzero(t.label.values == 0)
    imgs = np.load(ROOT / "data" / "dataset" / "images.npy", mmap_mode="r")
    J = np.stack([jpeg(np.asarray(imgs[i])) for i in neg])
    if features == "classical":
        from btmri.features.classical import batch_classical

        F = batch_classical(J)
    elif features == "scattering":
        from btmri.features.scattering import batch_scattering

        F = batch_scattering(J)
    else:
        from btmri.features.cnn import batch_cnn

        F = batch_cnn(J, features)
    np.savez_compressed(ROOT / "data" / "features" / f"{features}_jpeg{QUALITY}_negatives.npz", rows=neg, F=F)
    print(features, F.shape)


def run(features: str, head: str, alpha: float = 0.1):
    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    keep = t.label.values >= 0
    rows = np.flatnonzero(keep)
    t = t[keep].reset_index(drop=True)
    Xc = np.load(ROOT / "data" / "features" / f"{features}.npy")[rows].astype(np.float64)
    z = np.load(ROOT / "data" / "features" / f"{features}_jpeg{QUALITY}_negatives.npz")
    Xj = Xc.copy()
    pos_in_kept = np.searchsorted(rows, z["rows"])
    Xj[pos_in_kept] = z["F"]
    y, g = t.label.values.astype(int), t.patient.values
    width = 2048 if head == "elm" else 0
    rng = np.random.default_rng(0)
    u = rng.random(len(y))  # fixed per-slice draw: slice i is JPEG'd if u_i < p (nested across p)
    out = []
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        biased = np.where(((y == 0) & (u < p))[:, None], Xj, Xc)
        for k, (tr, te) in enumerate(patient_folds(y, g, 5, 0)):
            lam, _ = select_lambda_lopo(biased[tr], y[tr], g[tr], LAMBDAS, n_random=width)
            h = RidgeHead(lam=lam, n_random=width).fit(biased[tr], y[tr], g[tr])
            grp, D_b = h.lopo_decision_on(biased[te])
            _, D_c = h.lopo_decision_on(Xc[te])
            oof = h.lopo_decision()
            for world, X_eval, D in (("biased", biased[te], D_b), ("clean", Xc[te], D_c)):
                dec = h.decision_function(X_eval)
                m = binary_metrics(dec, y[te])
                ev = cp.evaluate_sets(cp.jackknife_plus(D, grp, oof, y[tr], g[tr], alpha, "pooled"), y[te], g[te])
                out.append({"features": features, "head": head, "p_jpeg_negatives": p, "fold": k, "world": world,
                            "auc": m["auc"], "balanced_accuracy": m["balanced_accuracy"],
                            "specificity": m["specificity"], "sensitivity": m["sensitivity"], **ev})
        print(f"p={p}: done", flush=True)
    res = pd.DataFrame(out)
    path = ROOT / "results" / "shortcut.csv"
    if path.exists():
        old = pd.read_csv(path)
        res = pd.concat([old[(old.features != features) | (old["head"] != head)], res], ignore_index=True)
    res.to_csv(path, index=False)
    print(res.groupby(["p_jpeg_negatives", "world"])[["auc", "balanced_accuracy", "coverage_patient",
                                                     "coverage_class0"]].mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "run"])
    ap.add_argument("--features", default="resnet50")
    ap.add_argument("--head", default="elm")
    a = ap.parse_args()
    extract(a.features) if a.mode == "extract" else run(a.features, a.head)

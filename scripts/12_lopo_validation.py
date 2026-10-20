"""Exactness and cost of closed-form leave-one-patient-out on the real features.

For growing numbers of patients, compute leave-one-patient-out decisions of
an ELM head (a) with the closed-form group-deletion identity and (b) by
refitting once per patient, and record the agreement and the wall-clock time.

Usage: python scripts/12_lopo_validation.py --features resnet50
Output: results/lopo_validation.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from btmri.heads import RidgeHead, brute_force_lopo

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="resnet50")
    ap.add_argument("--sizes", default="25,50,100,200,400")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--lam", type=float, default=100.0)
    args = ap.parse_args()
    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    keep = t.label.values >= 0
    t = t[keep].reset_index(drop=True)
    X = np.load(ROOT / "data" / "features" / f"{args.features}.npy")[np.flatnonzero(keep)].astype(np.float64)
    y, g = t.label.values.astype(int), t.patient.values
    rng = np.random.default_rng(0)
    order = rng.permutation(np.unique(g))
    factory = lambda: RidgeHead(lam=args.lam, n_random=args.width, random_state=0)  # noqa: E731
    out = {"features": args.features, "width": args.width, "lam": args.lam, "runs": []}
    for n in map(int, args.sizes.split(",")):
        sel = np.isin(g, order[:n])
        Xs, ys, gs = X[sel], y[sel], g[sel]
        t0 = time.time()
        fast = factory().fit(Xs, ys, gs).lopo_decision()
        t_fast = time.time() - t0
        t0 = time.time()
        slow = brute_force_lopo(factory, Xs, ys, gs)
        t_slow = time.time() - t0
        approx = brute_force_lopo(factory, Xs, ys, gs, refit_standardizer=True) if n <= 100 else None
        run = {"patients": n, "slices": int(sel.sum()), "closed_form_s": t_fast, "refit_s": t_slow,
               "speedup": t_slow / t_fast, "max_abs_diff": float(np.max(np.abs(fast - slow))),
               "max_abs_decision": float(np.max(np.abs(slow)))}
        if approx is not None:
            run["refit_standardizer_corr"] = float(np.corrcoef(fast, approx)[0, 1])
            run["refit_standardizer_sign_agreement"] = float(np.mean(np.sign(fast) == np.sign(approx)))
        out["runs"].append(run)
        print(run, flush=True)
        if n == 100:
            np.savez_compressed(ROOT / "data" / "outputs" / "lopo_pairs.npz", fast=fast, slow=slow,
                                approx=approx, y=ys)
    with open(ROOT / "results" / "lopo_validation.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

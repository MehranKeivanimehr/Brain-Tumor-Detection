"""How much do automatic labels change the conclusions?

For the patients whose AIMI segmentation was corrected by a radiologist,
compare automatic and corrected masks (voxel Dice, slice-label agreement) and
score the out-of-fold predictions of a benchmark configuration against both.

Usage: python scripts/07_label_audit.py --config resnet50|elm|patient|0
Output: results/label_audit.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from btmri.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="resnet50|elm|patient|0")
    args = ap.parse_args()
    qc = pd.read_csv(ROOT / "results" / "qc_patients.csv")
    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    keep = t.label.values >= 0
    t = t[keep].reset_index(drop=True)
    oof = np.load(ROOT / "data" / "outputs" / "benchmark_oof.npz")
    dec = oof[f"{args.config}|dec"]

    rad = qc[qc.ai_vs_rad_dice.notna() & (qc.status == "ok")]
    sel = (t.ref_label.values >= 0) & t.patient.isin(rad.PatientID).values
    ai_lab, ref_lab = t.label.values[sel], t.ref_label.values[sel]
    m_ai = binary_metrics(dec[sel], ai_lab)
    m_ref = binary_metrics(dec[sel], ref_lab)
    out = {
        "config": args.config,
        "patients_with_corrected_masks": int(len(rad)),
        "voxel_dice_ai_vs_corrected": {"median": float(rad.ai_vs_rad_dice.median()),
                                       "min": float(rad.ai_vs_rad_dice.min()),
                                       "mean": float(rad.ai_vs_rad_dice.mean())},
        "patients_with_identical_masks": int((rad.ai_vs_rad_dice > 0.999).sum()),
        "slices_compared": int(sel.sum()),
        "slice_label_disagreements": int((ai_lab != ref_lab).sum()),
        "metrics_vs_ai_labels": m_ai,
        "metrics_vs_corrected_labels": m_ref,
    }
    with open(ROOT / "results" / "label_audit.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

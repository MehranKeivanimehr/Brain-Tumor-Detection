"""Build the labelled 2D slice dataset from downloaded UPENN-GBM DICOM.

Outputs
  data/slices/<PatientID>.npz         per-patient slices and metadata
  data/dataset/images.npy             (N, 240, 240) uint8, all kept slices
  data/dataset/slices.csv             one row per slice (patient, label, areas, scanner)
  results/qc_patients.csv             per-patient geometry / label QC (committed)

Exclusions (reported in results/qc_patients.csv): non-axial FLAIR, an empty
or failed segmentation (no slice with >= 1 cm^2 of tumor), or a mask that is
not better aligned with FLAIR-hyperintense tissue than its own row- or
column-flipped copy (alignment margin <= 0, the signature of a misregistered
mask).
"""

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from btmri.data.dicom import is_axial, load_seg_on_grid, load_series
from btmri.data.slices import build_slices

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "upenn_gbm"
OUT = ROOT / "data" / "slices"


def process(row: dict) -> dict:
    pid = row["PatientID"]
    pdir = RAW / pid
    qc = {"PatientID": pid, "model": row["model"], "manufacturer": row["manufacturer"], "status": "ok"}
    try:
        vol = load_series(pdir / "flair")
        hdr = pydicom.dcmread(next((pdir / "flair").glob("*.dcm")), stop_before_pixels=True)
        qc["field_strength_T"] = float(getattr(hdr, "MagneticFieldStrength", None) or np.nan)
        qc.update(n_volume_slices=vol.pixels.shape[0], slice_spacing_mm=round(vol.slice_spacing, 3),
                  pixel_spacing_mm=round(vol.spacing[0], 4), axial=is_axial(vol))
        if not qc["axial"]:
            qc["status"] = "excluded_non_axial"
            return qc
        lab, info = load_seg_on_grid(next((pdir / "seg_ai").glob("*.dcm")), vol)
        qc.update(seg_flipped=";".join(info["flipped_axes"]) or "none", seg_uid_mismatch=info["uid_mismatch_frames"])
        ref = None
        if (pdir / "seg_rad").exists():
            ref, _ = load_seg_on_grid(next((pdir / "seg_rad").glob("*.dcm")), vol)
            a, b = lab > 0, ref > 0
            qc["ai_vs_rad_dice"] = float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
        ss = build_slices(vol, lab, ref)
        qc.update(lesion_z=ss.lesion_z, alignment_margin=ss.alignment_margin, n_kept=len(ss.label),
                  n_pos=int((ss.label == 1).sum()), n_neg=int((ss.label == 0).sum()), n_ambiguous=int((ss.label == -1).sum()),
                  max_tumor_area_mm2=float(ss.tumor_area_mm2.max()))
        if ss.ref_label is not None:
            both = (ss.label >= 0) & (ss.ref_label >= 0)
            qc["slice_label_agreement"] = float((ss.label[both] == ss.ref_label[both]).mean())
        if qc["n_pos"] == 0:
            qc["status"] = "excluded_no_tumor_slices"
        elif not np.isfinite(ss.alignment_margin) or ss.alignment_margin <= 0:
            qc["status"] = "excluded_mask_misaligned"
        if qc["status"] == "ok":
            np.savez_compressed(
                OUT / f"{pid}.npz", images=ss.images, slice_index=ss.slice_index, z_mm=ss.z_mm,
                head_area_mm2=ss.head_area_mm2, tumor_area_mm2=ss.tumor_area_mm2, enh_area_mm2=ss.enh_area_mm2,
                label=ss.label,
                ref_tumor_area_mm2=ss.ref_tumor_area_mm2 if ss.ref_label is not None else np.full(len(ss.label), np.nan),
                ref_label=ss.ref_label if ss.ref_label is not None else np.full(len(ss.label), -9, dtype=np.int8),
            )
    except Exception as e:  # keep going; the failure is recorded
        qc["status"] = f"error: {type(e).__name__}: {e}"[:200]
    return qc


def assemble(qc: pd.DataFrame, cohort: pd.DataFrame):
    rows, imgs = [], []
    meta = cohort.set_index("PatientID")
    fs = qc.set_index("PatientID")["field_strength_T"]
    for pid in qc.loc[qc.status == "ok", "PatientID"]:
        z = np.load(OUT / f"{pid}.npz")
        n = len(z["label"])
        imgs.append(z["images"])
        rows.append(pd.DataFrame({
            "patient": pid, "slice_index": z["slice_index"], "z_mm": z["z_mm"],
            "head_area_mm2": z["head_area_mm2"], "tumor_area_mm2": z["tumor_area_mm2"],
            "enh_area_mm2": z["enh_area_mm2"], "label": z["label"],
            "ref_tumor_area_mm2": z["ref_tumor_area_mm2"], "ref_label": z["ref_label"],
            "scanner_model": meta.loc[pid, "model"], "manufacturer": meta.loc[pid, "manufacturer"],
            "field_strength_T": fs.loc[pid],
            "age": meta.loc[pid, "age"], "sex": meta.loc[pid, "sex"],
        }))
        assert len(rows[-1]) == n
    ds = ROOT / "data" / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    table = pd.concat(rows, ignore_index=True)
    table.insert(0, "row", np.arange(len(table)))
    np.save(ds / "images.npy", np.concatenate(imgs))
    table.to_csv(ds / "slices.csv", index=False)
    return table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(ROOT / "results" / "cohort_manifest.csv")
    todo = [r for r in cohort.to_dict("records") if (RAW / r["PatientID"] / ".complete").exists()]
    print(f"processing {len(todo)} downloaded patients")
    with mp.Pool(4) as pool:
        qc = pd.DataFrame(pool.map(process, todo, chunksize=4))
    qc.to_csv(ROOT / "results" / "qc_patients.csv", index=False)
    print(qc.status.value_counts().to_string())
    table = assemble(qc, cohort)
    lab = table.label.value_counts().to_dict()
    print(f"slices: {len(table)} from {table.patient.nunique()} patients; labels {lab}")


if __name__ == "__main__":
    main()

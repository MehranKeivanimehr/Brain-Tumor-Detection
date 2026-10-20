"""Cohort selection and download of UPENN-GBM FLAIR series and AIMI segmentations.

The UPENN-GBM collection (Bakas et al., Sci Data 2022) is mirrored as DICOM by
the NCI Imaging Data Commons (Fedorov et al., RadioGraphics 2023). Tumor
segmentations come from the AIMI annotation effort (Murugesan et al., Sci Data
2025): an automatic nnU-Net segmentation for every series, plus a subset that
radiologists corrected.

Files are fetched over plain HTTPS from the public ``idc-open-data`` bucket
(Google Cloud Storage mirror of the AWS bucket listed in the IDC index).
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

COLLECTION = "upenn_gbm"
AI_SEG = "AIMI Brain MRI AI segmentation"
GCS_PREFIX = "https://storage.googleapis.com/idc-open-data/"


def _client():
    from idc_index import index  # imported lazily: heavy, only needed for downloads

    return index.IDCClient()


def select_cohort(client=None) -> pd.DataFrame:
    """One row per patient: the processed FLAIR series that has an AI segmentation.

    Columns: PatientID, StudyInstanceUID, flair_series, ai_seg_series,
    rad_seg_series (may be NaN), rad_desc, n_slices, manufacturer, model,
    field_strength_hint, idc_version.
    """
    client = client or _client()
    df = client.index
    u = df[df.collection_id == COLLECTION]
    flair = u[(u.Modality == "MR") & u.SeriesDescription.str.contains("flair", case=False, na=False)]
    seg = u[u.Modality == "SEG"]
    ai = seg[seg.SeriesDescription == AI_SEG][["StudyInstanceUID", "SeriesInstanceUID"]]
    rad = seg[seg.SeriesDescription.str.contains("radiologist", na=False)][
        ["StudyInstanceUID", "SeriesInstanceUID", "SeriesDescription"]
    ]
    m = flair.merge(ai.rename(columns={"SeriesInstanceUID": "ai_seg_series"}), on="StudyInstanceUID")
    m = m.merge(
        rad.rename(columns={"SeriesInstanceUID": "rad_seg_series", "SeriesDescription": "rad_desc"}),
        on="StudyInstanceUID",
        how="left",
    )
    m = m.sort_values(["PatientID", "StudyDate"]).groupby("PatientID", as_index=False).first()
    out = pd.DataFrame(
        {
            "PatientID": m.PatientID,
            "StudyInstanceUID": m.StudyInstanceUID,
            "flair_series": m.SeriesInstanceUID,
            "flair_description": m.SeriesDescription,
            "ai_seg_series": m.ai_seg_series,
            "rad_seg_series": m.rad_seg_series,
            "rad_desc": m.rad_desc,
            "n_slices": m.instanceCount.astype(int),
            "manufacturer": m.Manufacturer,
            "model": m.ManufacturerModelName,
            "age": m.PatientAge,
            "sex": m.PatientSex,
        }
    )
    out["idc_version"] = client.get_idc_version()
    return out.reset_index(drop=True)


def _fetch(url: str, dest: Path, retries: int = 4) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(f"{dest.suffix}.{os.getpid()}.part")  # unique per process
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                f.write(r.read())
            os.replace(tmp, dest)
            return
        except Exception:  # network hiccup: back off and retry
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def download_series(series_uid: str, dest_dir: Path, client=None, workers: int = 8) -> int:
    """Download every DICOM instance of a series into ``dest_dir``. Returns file count."""
    client = client or _client()
    dest_dir.mkdir(parents=True, exist_ok=True)
    urls = [u.replace("s3://idc-open-data/", GCS_PREFIX) for u in client.get_series_file_URLs(series_uid)]
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda u: _fetch(u, dest_dir / os.path.basename(u)), urls))
    return len(urls)


def download_cohort(cohort: pd.DataFrame, root: Path, client=None, workers: int = 8) -> None:
    """Download FLAIR + AI SEG (+ radiologist SEG when present) for each patient."""
    client = client or _client()
    for i, row in cohort.iterrows():
        pdir = root / row.PatientID
        done = pdir / ".complete"
        if done.exists():
            continue
        download_series(row.flair_series, pdir / "flair", client, workers)
        download_series(row.ai_seg_series, pdir / "seg_ai", client, workers)
        if isinstance(row.rad_seg_series, str):
            download_series(row.rad_seg_series, pdir / "seg_rad", client, workers)
        done.touch()
        if (i + 1) % 25 == 0:
            print(f"  downloaded {i + 1}/{len(cohort)} patients", flush=True)

"""Select the UPENN-GBM FLAIR cohort from the IDC index and download it.

Usage: python scripts/01_download_upenn_gbm.py [--limit N]

Writes results/cohort_manifest.csv (series UIDs only) and DICOM files under
data/raw/upenn_gbm/<PatientID>/{flair,seg_ai,seg_rad}.
"""

import argparse
from pathlib import Path

from btmri.data import idc

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reverse", action="store_true", help="walk the cohort backwards (run alongside a forward run)")
    args = ap.parse_args()

    client = idc._client()
    cohort = idc.select_cohort(client)
    (ROOT / "results").mkdir(exist_ok=True)
    cohort.to_csv(ROOT / "results" / "cohort_manifest.csv", index=False)
    print(f"cohort: {len(cohort)} patients, {cohort.rad_seg_series.notna().sum()} with radiologist SEG "
          f"(IDC {cohort.idc_version.iloc[0]})")
    if args.limit:
        cohort = cohort.head(args.limit)
    if args.reverse:
        cohort = cohort.iloc[::-1].reset_index(drop=True)
    idc.download_cohort(cohort, ROOT / "data" / "raw" / "upenn_gbm", client, args.workers)
    print("done")


if __name__ == "__main__":
    main()

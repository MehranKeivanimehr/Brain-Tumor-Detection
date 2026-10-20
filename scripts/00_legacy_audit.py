"""Chapter zero: re-evaluate the original pipeline from its saved feature files.

Usage: python scripts/00_legacy_audit.py   ->  results/legacy_audit.json
"""

import json
from pathlib import Path

from btmri.legacy_audit import run

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    res = run(ROOT / "legacy" / "data")
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "legacy_audit.json", "w") as f:
        json.dump(res, f, indent=1)
    d, p, g = res["duplicates"], res["original_protocol"], res["grouped_cv"]
    print(f"test images copied from training: {d['test_images_copied_from_train']}/{d['test_images']}; "
          f"identical pairs in training: {d['identical_pairs_within_train']}")
    print(f"ELM single run {p['elm_single_run_test_mean']:.3f} (sd {p['elm_single_run_test_sd']:.3f}); "
          f"best-of-10-on-test {p['best_of_10_on_test_mean']:.3f}; majority {p['majority_baseline_test']:.3f}")
    for k, v in g.items():
        print(f"  grouped CV {k}: {v}")

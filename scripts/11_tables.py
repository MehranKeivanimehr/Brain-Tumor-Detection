"""Generate markdown tables from results/ and inject them into README.md / docs.

Tables live between markers such as
    <!-- TABLE:benchmark -->  ...  <!-- /TABLE:benchmark -->
so every number in the documentation is produced from the result files.

Usage: python scripts/11_tables.py [--inject README.md docs/results.md docs/legacy_audit.md]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FEAT = {"classical": "Classical texture (DWT+GLCM+LBP)", "scattering": "Wavelet scattering",
        "resnet50": "ResNet-50 (ImageNet)", "densenet121": "DenseNet-121 (ImageNet)"}
HEAD = {"ridge": "Ridge", "elm": "ELM", "logistic": "Logistic", "knn": "15-NN"}
METHOD = {
    "naive_slice_split": "Naive: calibrate on training-patient slices",
    "patient_split": "Patient split, slices pooled",
    "patient_subsample": "Patient split, one slice per patient (Dunn et al.)",
    "patient_pooled_cdf": "Hierarchical CP (Lee et al.), patient split",
    "jackknife+_sub": "Jackknife+ over patients (exact LOPO), one slice each",
    "jackknife+_pooled": "**Hierarchical jackknife+ (exact LOPO)**",
    "cv+10_pooled": "Hierarchical CV+ (10 refitted models)",
    "naive_slice_split_mondrian": "Naive, class-conditional",
    "patient_pooled_cdf_mondrian": "Hierarchical CP, class-conditional",
    "jackknife+_pooled_mondrian": "**Hierarchical jackknife+, class-conditional**",
}


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def t_benchmark():
    b = pd.read_csv(RES / "benchmark.csv")
    b = b[b.seed == 0]
    rows = []
    for f in FEAT:
        for h in HEAD:
            s = b[(b.features == f) & (b["head"] == h)]
            if len(s) < 2:
                continue
            sl, pa = s[s.protocol == "slice"].iloc[0], s[s.protocol == "patient"].iloc[0]
            rows.append(f"| {FEAT[f]} | {HEAD[h]} | {sl.auc:.3f} | **{pa.auc:.3f}** [{pa.auc_lo:.3f}, {pa.auc_hi:.3f}] "
                        f"| {pct(sl.balanced_accuracy)} | **{pct(pa.balanced_accuracy)}** "
                        f"[{pct(pa.bacc_lo)}, {pct(pa.bacc_hi)}] | +{pct(sl.balanced_accuracy - pa.balanced_accuracy)} "
                        f"| {pa.ece:.3f} |")
    head = ("| Features | Head | AUC, slice split | AUC, patient split [95% CI] | bAcc %, slice split "
            "| bAcc %, patient split [95% CI] | Inflation (bAcc pts) | ECE, patient split |\n"
            "|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


def t_conformal(cfg="resnet50_elm", alpha=0.1):
    c = pd.read_csv(RES / "conformal.csv")
    c = c[(c.config == cfg) & np.isclose(c.alpha, alpha)]
    g = c.groupby("method")
    agg = g[["coverage_patient", "coverage_class0", "coverage_class1", "coverage_patient_p10", "mean_size",
             "frac_singleton"]].agg(["mean", "std"])
    lines = [f"| Calibration (target {pct(1 - alpha, 0)}% coverage) | Coverage on new patients | No-tumor class "
             "| Tumor class (sensitivity) | Worst-decile patient | Mean set size | Singletons |",
             "|---|---|---|---|---|---|---|"]
    for m in METHOD:
        if m not in agg.index:
            continue
        r = agg.loc[m]
        lines.append(f"| {METHOD[m]} | {pct(r[('coverage_patient', 'mean')])} ± {pct(r[('coverage_patient', 'std')])} "
                     f"| {pct(r[('coverage_class0', 'mean')])} | {pct(r[('coverage_class1', 'mean')])} "
                     f"| {pct(r[('coverage_patient_p10', 'mean')])} | {r[('mean_size', 'mean')]:.3f} "
                     f"| {pct(r[('frac_singleton', 'mean')])}% |")
    return "\n".join(lines)


def t_shift():
    s = pd.read_csv(RES / "scanner_shift.csv")
    g = s.groupby(["domain", "method"])[["n_patients", "auc", "balanced_accuracy", "coverage_patient",
                                         "coverage_class0", "coverage_class1", "coverage_patient_p10",
                                         "mean_size"]].mean()
    lines = ["| Test domain | Patients | AUC | bAcc % | Calibration | Coverage | No-tumor class | Tumor class "
             "| Worst-decile patient | Set size |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    domains = sorted(s.domain.unique(), key=lambda d: ("in-distribution" not in d, d))
    for dom in domains:
        for m in ("jackknife+_pooled", "jackknife+_pooled_mondrian"):
            if (dom, m) not in g.index:
                continue
            r = g.loc[(dom, m)]
            lines.append(f"| {dom} | {r.n_patients:.0f} | {r.auc:.3f} | {pct(r.balanced_accuracy)} "
                         f"| {METHOD.get(m, m).replace('**', '')} | {pct(r.coverage_patient)} | {pct(r.coverage_class0)} "
                         f"| {pct(r.coverage_class1)} | {pct(r.coverage_patient_p10)} | {r.mean_size:.2f} |")
    return "\n".join(lines)


def t_conformal_configs(alpha=0.1):
    c = pd.read_csv(RES / "conformal.csv")
    c = c[np.isclose(c.alpha, alpha)]
    methods = ("naive_slice_split", "patient_pooled_cdf", "jackknife+_pooled", "cv+10_pooled")
    short = {"naive_slice_split": "Naive", "patient_pooled_cdf": "Hierarchical CP",
             "jackknife+_pooled": "Hierarchical jackknife+", "cv+10_pooled": "CV+ (10 refits)"}
    lines = ["| Features | Head | " + " | ".join(f"{short[m]}: coverage / size" for m in methods)
             + " | Folds below target: naive / jackknife+ |",
             "|---|---|" + "---|" * (len(methods) + 1)]
    for f in FEAT:
        for h in HEAD:
            s = c[c.config == f"{f}_{h}"]
            if s.empty:
                continue
            cells = []
            for m in methods:
                q = s[s.method == m]
                cells.append(f"{pct(q.coverage_patient.mean())} / {q.mean_size.mean():.2f}")
            below = [int((s[s.method == m].coverage_patient < 1 - alpha).sum())
                     for m in ("naive_slice_split", "jackknife+_pooled")]
            n = int((s.method == "naive_slice_split").sum())
            lines.append(f"| {FEAT[f]} | {HEAD[h]} | " + " | ".join(cells) + f" | {below[0]} / {below[1]} of {n} |")
    return "\n".join(lines)


def t_shortcut(features="resnet50", head="elm"):
    s = pd.read_csv(RES / "shortcut.csv")
    s = s[(s.features == features) & (s["head"] == head)]
    g = s.groupby(["p_jpeg_negatives", "world"])[["auc", "balanced_accuracy", "specificity", "coverage_patient",
                                                  "coverage_class0"]].mean()
    lines = ["| Tumor-free slices saved as JPEG | Tested on | AUC | bAcc % | Specificity % | Set coverage % "
             "| Coverage, tumor-free % |",
             "|---|---|---|---|---|---|---|"]
    for (p, world), r in g.iterrows():
        lines.append(f"| {100 * p:.0f}% | {'biased data' if world == 'biased' else 'clean data'} | {r.auc:.3f} "
                     f"| {pct(r.balanced_accuracy)} | {pct(r.specificity)} | {pct(r.coverage_patient)} "
                     f"| {pct(r.coverage_class0)} |")
    return "\n".join(lines)


def t_legacy():
    r = json.loads((RES / "legacy_audit.json").read_text())
    p, g, d = r["original_protocol"], r["grouped_cv"], r["duplicates"]
    rows = [
        ("Accuracy reported by the original notebook", "90.0%", "image-level split, test images counted twice"),
        ("ELM, best of 10 seeds chosen on the test set (main.m)", f"{pct(p['best_of_10_on_test_mean'])}%",
         f"max {pct(p['best_of_10_on_test_max'])}% over 200 repetitions"),
        ("ELM, a single seed, original split", f"{pct(p['elm_single_run_test_mean'])}% ± {pct(p['elm_single_run_test_sd'])}",
         "2,000 seeds"),
        ("Majority class on the original test set", f"{pct(p['majority_baseline_test'])}%", "always 'tumor'"),
        ("ELM as configured, 5-fold CV grouped by duplicate image", f"{pct(g['elm_as_configured'][0])}%",
         f"{g['n_unique_images']} unique images"),
        ("ELM without the broken IDM feature, 20 hidden units", f"{pct(g['elm_without_idm_20_hidden'][0])}%", ""),
        ("Logistic regression, all 69 features", f"{pct(g['logistic_all_features'][0])}%", ""),
        ("Logistic regression, LBP only (no segmentation)", f"{pct(g['logistic_lbp_only'][0])}%", ""),
        ("Logistic regression, segmentation-derived features only", f"{pct(g['logistic_segmentation_features_only'][0])}%", ""),
    ]
    lines = ["| Protocol | Accuracy | Note |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines.append("")
    lines.append(f"{d['test_images_copied_from_train']} of {d['test_images']} test images are copies of training images "
                 f"(identical LBP feature vectors); the 13 tumor test images are only {d['distinct_tumor_test_images']} "
                 f"distinct images; the training set contains {d['identical_pairs_within_train']} identical pairs.")
    return "\n".join(lines)


TABLES = {"benchmark": t_benchmark, "conformal": t_conformal, "conformal_configs": t_conformal_configs,
          "shift": t_shift, "shortcut": t_shortcut, "legacy": t_legacy}


def inject(path: Path, tables: dict):
    text = path.read_text()
    for name, body in tables.items():
        pat = re.compile(rf"(<!-- TABLE:{name} -->\n)(?:.*?\n)?(<!-- /TABLE:{name} -->)", re.S)
        text = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
    path.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", nargs="*", default=[])
    args = ap.parse_args()
    built = {}
    for name, fn in TABLES.items():
        try:
            built[name] = fn()
        except FileNotFoundError:
            print(f"skip {name}: results missing")
    (RES / "tables.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in built.items()) + "\n")
    for p in args.inject:
        inject(ROOT / p, built)
    print("tables:", ", ".join(built))


if __name__ == "__main__":
    main()

"""Render every static figure from the saved results.

Usage: python scripts/09_figures.py [--only legacy,dataset,leakage,lopo,conformal,sensitivity,shift,shortcut,calibration]
Figures are written to figures/*.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from btmri.metrics import ece
from btmri.viz import HONEST, LEAKY, C, plt, save, setup

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FEAT_LABEL = {"classical": "Classical texture\n(DWT+GLCM+LBP)", "scattering": "Wavelet\nscattering",
              "resnet50": "ResNet-50\n(ImageNet)", "densenet121": "DenseNet-121\n(ImageNet)"}
HEAD_LABEL = {"ridge": "Ridge", "elm": "ELM (2048)", "logistic": "Logistic", "knn": "15-NN"}


# --------------------------------------------------------------------------- chapter zero
def fig_legacy():
    r = json.loads((RES / "legacy_audit.json").read_text())
    p, g, d = r["original_protocol"], r["grouped_cv"], r["duplicates"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 0.9]})

    ax = axes[0]
    labels = ["Claimed\n(notebook)", "Best-of-10\non test", "Single\nELM run", "Majority\nclass",
              "Logistic\n(honest CV)", "LBP only\n(honest CV)"]
    vals = [0.90, p["best_of_10_on_test_mean"], p["elm_single_run_test_mean"], p["majority_baseline_test"],
            g["logistic_all_features"][0], g["logistic_lbp_only"][0]]
    errs = [0, np.std(p["best_of_10_test_accuracies"]), p["elm_single_run_test_sd"], 0,
            g["logistic_all_features"][1], g["logistic_lbp_only"][1]]
    cols = [C["grey"], LEAKY, LEAKY, C["grey"], HONEST, HONEST]
    ax.bar(range(6), vals, yerr=errs, color=cols, capsize=3, width=0.7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02 + errs[i], f"{100 * v:.1f}%", ha="center", fontsize=9)
    ax.axhline(p["majority_baseline_test"], ls="--", c=C["grey"], lw=1)
    ax.set_xticks(range(6), labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Claimed vs measured accuracy")

    ax = axes[1]
    single = np.asarray(p["single_run_test_accuracies"])
    best = np.asarray(p["best_of_10_test_accuracies"])
    bins = np.arange(0.2, 0.95, 1 / 21)
    ax.hist(single, bins=bins, color=C["sky"], alpha=0.8, density=True, label="single random ELM")
    ax.hist(best, bins=bins, color=LEAKY, alpha=0.6, density=True, label="best of 10, picked on test")
    ax.axvline(p["majority_baseline_test"], ls="--", c=C["grey"], lw=1)
    ax.text(p["majority_baseline_test"] - 0.008, ax.get_ylim()[1] * 0.72, "majority\nclass", fontsize=8,
            color=C["grey"], ha="right")
    ax.set_xlabel("Test accuracy on the 21-image test set")
    ax.set_ylabel("Density over 2,000 seeds")
    ax.set_title("Picking the best seed on test: +14 points")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    tn = [f"No{i + 1}" for i in range(8)] + [f"Y{i + 1}" for i in range(13)]
    copied = [t in d["test_to_train"] for t in tn]
    ax.barh(range(21), [1] * 21, color=[LEAKY if c else HONEST for c in copied], height=0.8)
    for i, t in enumerate(tn):
        ax.text(0.03, i, t, va="center", fontsize=7.5, color="white", fontweight="bold")
        src = d["test_to_train"].get(t, "unseen")
        ax.text(0.97, i, f"= train {src}" if src != "unseen" else "unseen", va="center", ha="right",
                fontsize=7.5, color="white")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(f"{d['test_images_copied_from_train']}/{d['test_images']} test images are copies\nof training images")
    fig.suptitle("Chapter zero: audit of the original MATLAB pipeline (Kaggle, 253 slices)", y=1.03, fontsize=12)
    save(fig, FIG / "fig1_legacy_audit.png")


# --------------------------------------------------------------------------- dataset
def fig_dataset(example_patients=("UPENN-GBM-00002", "UPENN-GBM-00019")):
    from btmri.data.dicom import load_seg_on_grid, load_series
    from btmri.data.slices import build_slices

    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    qc = pd.read_csv(RES / "qc_patients.csv")
    fig = plt.figure(figsize=(16, 8.2))
    gs = fig.add_gridspec(3, 7, height_ratios=[1, 1, 1.05], hspace=0.45, wspace=0.12)
    raw = ROOT / "data" / "raw" / "upenn_gbm"
    for r, pid in enumerate(example_patients):
        vol = load_series(raw / pid / "flair")
        lab, _ = load_seg_on_grid(next((raw / pid / "seg_ai").glob("*.dcm")), vol)
        ss = build_slices(vol, lab, return_masks=True)
        pick = np.linspace(2, len(ss.label) - 3, 7).astype(int)
        for c, k in enumerate(pick):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(ss.images[k], cmap="gray", vmin=0, vmax=255)
            m = ss.masks[k]
            for val, col in ((2, C["yellow"]), (3, C["red"]), (1, C["sky"])):
                if (m == val).any():
                    ax.contour(m == val, levels=[0.5], colors=[col], linewidths=0.8)
            name = {1: "tumor", 0: "no tumor", -1: "excluded (<1 cm²)"}[int(ss.label[k])]
            col = {1: LEAKY, 0: HONEST, -1: C["grey"]}[int(ss.label[k])]
            ax.set_title(f"{name}", fontsize=9, color=col, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if c == 0:
                ax.set_ylabel(pid, fontsize=8)
    bottom = gs[2, :].subgridspec(1, 3, wspace=0.32, width_ratios=[1, 1, 1.2])
    ax = fig.add_subplot(bottom[0])
    pos = t.tumor_area_mm2[t.label == 1]
    amb = t.tumor_area_mm2[t.label == -1]
    bins = np.logspace(0, np.log10(t.tumor_area_mm2.max() + 1), 40)
    ax.hist(amb, bins=bins, color=C["grey"], label=f"excluded ({len(amb)})")
    ax.hist(pos, bins=bins, color=LEAKY, alpha=0.85, label=f"tumor slices ({len(pos)})")
    ax.axvline(100, c=C["ink"], ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("whole-tumor area per slice (mm²)")
    ax.set_ylabel("slices")
    ax.legend(fontsize=8)
    ax.set_title("Label definition: ≥ 1 cm² tumor vs none")
    ax = fig.add_subplot(bottom[1])
    per = t[t.label >= 0].groupby("patient").label.agg(["sum", "count"])
    ax.hist(per["count"], bins=25, color=C["sky"], label="labelled slices")
    ax.hist(per["sum"], bins=25, color=LEAKY, alpha=0.7, label="tumor slices")
    ax.set_xlabel("slices per patient")
    ax.set_ylabel("patients")
    ax.legend(fontsize=8)
    ax.set_title(f"{per.shape[0]} patients, {int(per['count'].sum()):,} slices")
    ax = fig.add_subplot(bottom[2])
    ok = qc[qc.status == "ok"].copy()
    ok["scanner"] = ok.model.where(ok.model.isin(ok.model.value_counts().index[:6]), "other")
    vc = ok.groupby("scanner").field_strength_T.agg(["size", "median"]).sort_values("size")
    ax.barh(vc.index, vc["size"], color=[HONEST if f >= 2.9 else C["orange"] for f in vc["median"].fillna(3)])
    for i, (n, f) in enumerate(zip(vc["size"], vc["median"])):
        ax.text(n + 3, i, f"{n}  ({f:.1f} T)" if np.isfinite(f) else f"{n}", va="center", fontsize=8)
    ax.set_xlabel("patients")
    ax.set_title("Scanners (blue 3 T, orange 1.5 T)")
    ax.set_xlim(0, vc["size"].max() * 1.25)
    fig.suptitle("UPENN-GBM FLAIR slice benchmark: tumor and tumor-free slices from the same patients and scanners "
                 "(yellow edema, red enhancing, blue necrosis)", fontsize=11, y=0.99)
    save(fig, FIG / "fig2_dataset.png")


# --------------------------------------------------------------------------- leakage
def fig_leakage():
    b = pd.read_csv(RES / "benchmark.csv")
    b = b[b.seed == 0]
    nn = json.loads((RES / "neighbour_leakage.json").read_text())
    feats = [f for f in FEAT_LABEL if f in set(b.features)]
    heads = [h for h in HEAD_LABEL if h in set(b["head"])]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.6, 1.6, 0.9]})
    for ax, metric, lo, hi, name in ((axes[0], "auc", "auc_lo", "auc_hi", "AUC"),
                                     (axes[1], "balanced_accuracy", "bacc_lo", "bacc_hi", "Balanced accuracy")):
        w = 0.8 / len(heads)
        for j, h in enumerate(heads):
            for i, f in enumerate(feats):
                s = b[(b.features == f) & (b["head"] == h)]
                if len(s) < 2:
                    continue
                sl, pa = s[s.protocol == "slice"].iloc[0], s[s.protocol == "patient"].iloc[0]
                x = i + (j - (len(heads) - 1) / 2) * w
                ax.bar(x, sl[metric], w * 0.92, color=LEAKY, alpha=0.35, edgecolor=LEAKY,
                       label="slice-level split (leaky)" if (i == 0 and j == 0) else None)
                ax.bar(x, pa[metric], w * 0.92, color=HONEST,
                       label="patient-level split" if (i == 0 and j == 0) else None)
                ax.errorbar(x, pa[metric], yerr=[[pa[metric] - pa[lo]], [pa[hi] - pa[metric]]], c="k", lw=0.8,
                            capsize=1.5)
                ax.text(x, 0.505, HEAD_LABEL[h].split(" ")[0], rotation=90, ha="center", va="bottom", fontsize=7,
                        color="white")
        ax.set_xticks(range(len(feats)), [FEAT_LABEL[f] for f in feats], fontsize=8.5)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel(name)
        ax.set_title(name, pad=24)
        ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        ax.grid(axis="x", visible=False)
    ax = axes[2]
    cats = ["nearest training slice\nfrom the same patient", "1-NN label copy\naccuracy"]
    sv = [nn["slice"]["nn_same_patient"], nn["slice"]["one_nn_accuracy"]]
    pv = [nn["patient"]["nn_same_patient"], nn["patient"]["one_nn_accuracy"]]
    xs = np.arange(2)
    ax.bar(xs - 0.2, sv, 0.38, color=LEAKY, alpha=0.8, label="slice-level split")
    ax.bar(xs + 0.2, pv, 0.38, color=HONEST, label="patient-level split")
    for x, v in zip(list(xs - 0.2) + list(xs + 0.2), sv + pv):
        ax.text(x, v + 0.01, f"{100 * v:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(xs, cats, fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Why: nearest neighbours ({FEAT_LABEL[nn['features']].splitlines()[0]})", pad=24)
    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, FIG / "fig3_leakage.png")


# --------------------------------------------------------------------------- exact LOPO
def fig_lopo():
    r = json.loads((RES / "lopo_validation.json").read_text())
    pairs = np.load(ROOT / "data" / "outputs" / "lopo_pairs.npz", allow_pickle=True)
    runs = pd.DataFrame(r["runs"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    ax.scatter(pairs["slow"], pairs["fast"], s=4, c=C["blue"], alpha=0.4)
    lim = [pairs["slow"].min(), pairs["slow"].max()]
    ax.plot(lim, lim, c=C["ink"], lw=0.8)
    ax.set_xlabel("refit without the patient")
    ax.set_ylabel("closed-form group deletion")
    d = runs.loc[runs.patients == 100, "max_abs_diff"].iloc[0]
    ax.set_title(f"Exact: max |difference| = {d:.1e}")
    ax = axes[1]
    ax.plot(runs.patients, runs.refit_s, "o-", c=LEAKY, label="refit once per patient")
    ax.plot(runs.patients, runs.closed_form_s, "o-", c=HONEST, label="closed form (one factorisation)")
    for _, rr in runs.iterrows():
        ax.text(rr.patients, rr.refit_s * 1.15, f"×{rr.speedup:.0f}", ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(runs.patients, [str(v) for v in runs.patients])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("training patients")
    ax.set_ylabel("seconds (4 CPU cores)")
    ax.set_title("Leave-one-patient-out cost")
    ax.legend(fontsize=8)
    ax = axes[2]
    if pairs["approx"].dtype != object and pairs["approx"].shape == pairs["fast"].shape:
        ax.scatter(pairs["approx"], pairs["fast"], s=4, c=C["orange"], alpha=0.4)
        lim = [pairs["approx"].min(), pairs["approx"].max()]
        ax.plot(lim, lim, c=C["ink"], lw=0.8)
        rr = runs.loc[runs.patients == 100].iloc[0]
        ax.set_title(f"Standardiser also refitted: r = {rr.refit_standardizer_corr:.5f}")
        ax.set_xlabel("refit of standardiser + head without the patient")
        ax.set_ylabel("closed form (standardiser fixed)")
    fig.suptitle(f"Closed-form leave-one-patient-out for an ELM head ({r['width']} random features, "
                 f"{FEAT_LABEL[r['features']].splitlines()[0]} inputs)", y=1.02)
    save(fig, FIG / "fig4_exact_lopo.png")


# --------------------------------------------------------------------------- conformal
CP_ORDER = ["naive_slice_split", "patient_split", "patient_subsample", "patient_pooled_cdf", "cv+10_pooled",
            "jackknife+_sub", "jackknife+_pooled"]
CP_LABEL = {"naive_slice_split": "naive: calibrate on training-patient slices",
            "patient_split": "patient split, slices pooled",
            "patient_subsample": "patient split, 1 slice per patient",
            "patient_pooled_cdf": "hierarchical CP (patient split)",
            "cv+10_pooled": "hierarchical CV+ (10 refits)",
            "jackknife+_sub": "jackknife+ over patients, 1 slice each",
            "jackknife+_pooled": "hierarchical jackknife+ (exact LOPO)"}


def fig_conformal(cfg="resnet50_elm"):
    c = pd.read_csv(RES / "conformal.csv")
    c = c[c.config == cfg]
    order = CP_ORDER[::-1]
    ys = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.25, 1, 0.8]})
    ax = axes[0]
    for k, (al, mk) in enumerate(((0.05, "s"), (0.10, "o"), (0.20, "^"))):
        s = c[np.isclose(c.alpha, al)]
        for i, m in enumerate(order):
            v = s[s.method == m].coverage_patient
            col = LEAKY if m.startswith("naive") else HONEST
            ax.errorbar(v.mean(), i + (1 - k) * 0.22, xerr=v.std(), fmt=mk, c=col, ms=5, capsize=2)
        ax.axvline(1 - al, c=C["grey"], ls="--", lw=0.8)
        ax.plot([], [], mk, c=C["ink"], ms=5, label=f"target {100 * (1 - al):.0f}%")
    ax.set_yticks(ys, [CP_LABEL[m] for m in order], fontsize=8.5)
    ax.set_xlabel("coverage on unseen patients (mean ± sd, 15 folds)")
    ax.set_title("Coverage at three targets (dashed lines)", pad=22)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    s = c[np.isclose(c.alpha, 0.10)]
    sizes = [s[s.method == m].mean_size.mean() for m in order]
    covs = [s[s.method == m].coverage_patient.mean() for m in order]
    cols = [LEAKY if cv < 0.895 else HONEST for cv in covs]
    ax.barh(ys, sizes, color=cols, height=0.65)
    for i, (sz, cv) in enumerate(zip(sizes, covs)):
        ax.text(sz + 0.002, i, f"{sz:.3f}  (coverage {100 * cv:.1f}%)", va="center", fontsize=8)
    ax.set_yticks(ys, [""] * len(order))
    ax.set_xlim(1.0, max(sizes) * 1.07)
    ax.set_xlabel("mean prediction-set size at 90% target (labels per slice)")
    ax.set_title("Mean set size at the 90% target", pad=22)
    ax.grid(axis="y", visible=False)
    ax = axes[2]
    data = [s[s.method == m].coverage_patient_p10.values for m in ("naive_slice_split", "patient_pooled_cdf",
                                                                   "jackknife+_pooled")]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": C["ink"]})
    for patch, col in zip(bp["boxes"], (LEAKY, C["sky"], HONEST)):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_xticks([1, 2, 3], ["naive", "hierarchical\nCP", "hierarchical\njackknife+"], fontsize=8.5)
    ax.axhline(0.9, c=C["grey"], ls="--", lw=0.8)
    ax.set_ylabel("coverage of the worst-covered 10% of patients")
    ax.set_title("Coverage is marginal, not per patient", pad=22)
    fig.suptitle("Patient-level conformal prediction on unseen patients (ResNet-50 features, ELM head; "
                 "orange = calibrated on slices of training patients)", y=1.06)
    save(fig, FIG / "fig5_conformal.png")


def fig_sensitivity(cfg="resnet50_elm", alpha="0.1"):
    """Class-conditional (Mondrian) vs marginal: coverage of tumor slices by tumor size."""
    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    t = t[t.label >= 0].reset_index(drop=True)
    ex = np.load(ROOT / "data" / "outputs" / f"conformal_sets_{cfg}.npz")
    te = ex["rows"]
    y = t.label.values[te]
    area = t.tumor_area_mm2.values[te]
    bins = [100, 300, 1000, 3000, 1e9]
    names = ["1–3 cm²", "3–10 cm²", "10–30 cm²", "> 30 cm²"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for k, (m, col, lab) in enumerate((("naive_slice_split", LEAKY, "naive"),
                                       ("jackknife+_pooled", C["sky"], "hierarchical jackknife+"),
                                       ("jackknife+_pooled_mondrian", HONEST, "… class-conditional"))):
        sets = ex[f"{m}|{alpha}"]
        cov = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (y == 1) & (area >= lo) & (area < hi)
            cov.append(sets[sel, 1].mean())
        ax.bar(np.arange(4) + (k - 1) * 0.27, cov, 0.26, color=col, label=lab)
    ax.axhline(1 - float(alpha), c=C["grey"], ls="--", lw=0.8)
    ax.set_xticks(range(4), names)
    ax.set_xlabel("tumor area on the slice")
    ax.set_ylabel("tumor slices whose set contains 'tumor'")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Sensitivity of the set by tumor size (one outer fold)")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    c = pd.read_csv(RES / "conformal.csv")
    c = c[(c.config == cfg) & np.isclose(c.alpha, float(alpha))]
    ms = ["naive_slice_split", "naive_slice_split_mondrian", "jackknife+_pooled", "jackknife+_pooled_mondrian",
          "patient_pooled_cdf_mondrian"]
    labs = ["naive", "naive,\nclass-cond.", "hier. JK+", "hier. JK+,\nclass-cond.", "hier. CP,\nclass-cond."]
    for k, (col_name, col) in enumerate((("coverage_class0", "#BBBBBB"), ("coverage_class1", C["purple"]))):
        v = [c[c.method == m][col_name].mean() for m in ms]
        e = [c[c.method == m][col_name].std() for m in ms]
        ax.bar(np.arange(len(ms)) + (k - 0.5) * 0.38, v, 0.36, yerr=e, capsize=2, color=col,
               label="no-tumor slices" if k == 0 else "tumor slices")
    ax.axhline(1 - float(alpha), c=C["grey"], ls="--", lw=0.8)
    ax.set_xticks(range(len(ms)), labs, fontsize=8)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("class-conditional coverage")
    ax.set_title("Per-class coverage (15 folds)", pad=22)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, FIG / "fig6_sensitivity.png")


def fig_shift():
    s = pd.read_csv(RES / "scanner_shift.csv")
    doms = ["TrioTim 3T (in-distribution)", "other 3T scanners", "1.5T scanners"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, col_name, title in ((axes[0], "auc", "AUC"), (axes[1], "coverage_patient", "Coverage (target 90%)"),
                                (axes[2], "coverage_class1", "Tumor-class coverage (target 90%)")):
        for k, (m, col) in enumerate((("jackknife+_pooled", C["sky"]), ("jackknife+_pooled_mondrian", HONEST))):
            v = [s[(s.domain == d) & (s.method == m)][col_name].mean() for d in doms]
            e = [s[(s.domain == d) & (s.method == m)][col_name].std() for d in doms]
            if col_name == "auc" and k == 1:
                continue
            ax.bar(np.arange(3) + (k - 0.5) * 0.38 * (col_name != "auc"), v, 0.36, yerr=e, capsize=2, color=col,
                   label=("hierarchical jackknife+" if k == 0 else "… class-conditional") if col_name != "auc" else None)
        n = [int(s[s.domain == d].n_patients.mean()) for d in doms]
        ax.set_xticks(range(3), [f"{d.split(' (')[0]}\n(n={k})" for d, k in zip(doms, n)], fontsize=8.5)
        if col_name != "auc":
            ax.axhline(0.9, c=C["grey"], ls="--", lw=0.8)
            ax.set_ylim(0.78, 0.96)
            ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        else:
            ax.set_ylim(0.85, 0.98)
        ax.set_title(title, pad=22)
    fig.suptitle("Train and calibrate on TrioTim patients only; test on unseen scanners", y=1.1)
    save(fig, FIG / "fig7_scanner_shift.png")


def fig_shortcut(features="resnet50", head="elm", quality=50):
    import io

    from PIL import Image

    s = pd.read_csv(RES / "shortcut.csv")
    s = s[(s.features == features) & (s["head"] == head)]
    g = s.groupby(["p_jpeg_negatives", "world"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), gridspec_kw={"width_ratios": [1.8, 1, 1, 1]})
    # the cue: one tumor-free slice, original vs JPEG
    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    imgs = np.load(ROOT / "data" / "dataset" / "images.npy", mmap_mode="r")
    neg = t[(t.label == 0) & (t.patient == "UPENN-GBM-00002")]
    img = np.asarray(imgs[neg.head_area_mm2.idxmax()])
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    jp = np.asarray(Image.open(io.BytesIO(buf.getvalue())))
    diff = np.clip(10 * np.abs(jp.astype(float) - img.astype(float)), 0, 255)
    ax = axes[0]
    gap = np.full((img.shape[0], 8), 255.0)
    ax.imshow(np.hstack([img, gap, jp, gap, diff]), cmap="gray", vmin=0, vmax=255)
    for k, lab in enumerate(("original", f"JPEG, quality {quality}", "|difference| × 10")):
        ax.text(k * (img.shape[1] + 8) + img.shape[1] / 2, img.shape[0] + 14, lab, ha="center", va="top",
                fontsize=9)
    ax.axis("off")
    ax.set_title("The cue: tumor-free slices re-encoded as JPEG")
    panels = ((axes[1], "balanced_accuracy", "Balanced accuracy", None),
              (axes[2], "specificity", "Specificity (tumor-free slices)", None),
              (axes[3], "coverage_class0", "Prediction sets: coverage of\ntumor-free slices (target 90%)", 0.9))
    for ax, col_name, title, target in panels:
        for world, col, lab in (("biased", LEAKY, "tested on the biased data (what gets reported)"),
                                ("clean", HONEST, "tested on the same new patients, no JPEG cue")):
            m = g[col_name].mean().xs(world, level="world")
            e = g[col_name].std().xs(world, level="world")
            ax.errorbar(m.index * 100, m.values, yerr=e.values, fmt="o-", c=col, ms=5, capsize=3, label=lab)
        if target is not None:
            ax.axhline(target, c=C["grey"], ls="--", lw=0.8)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("% of tumor-free slices saved as JPEG")
        ax.set_ylim(0, 1.03)
        ax.set_title(title)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.62, -0.08), ncol=2, fontsize=9)
    fig.suptitle("A Kaggle-style source cue: the benchmark approaches 99% while the model and its "
                 "guarantee fail on clean data (patient-level CV, ResNet-50 + ELM)", y=1.04)
    save(fig, FIG / "fig8_shortcut.png")


def fig_calibration(config="resnet50|elm|patient|0"):
    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    t = t[t.label >= 0].reset_index(drop=True)
    oof = np.load(ROOT / "data" / "outputs" / "benchmark_oof.npz")
    p, y = oof[f"{config}|prob"], t.label.values
    area = t.tumor_area_mm2.values
    dec = oof[f"{config}|dec"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    order = np.argsort(p)
    bins = np.array_split(order, 15)
    ax.plot([0, 1], [0, 1], c=C["grey"], ls="--", lw=0.8)
    ax.plot([p[b].mean() for b in bins], [y[b].mean() for b in bins], "o-", c=HONEST)
    ax.set_xlabel("predicted P(tumor)")
    ax.set_ylabel("observed fraction of tumor slices")
    ax.set_title(f"Reliability (Platt on out-of-patient decisions)\nECE = {ece(p, y):.3f}")
    ax = axes[1]
    edges = [100, 200, 400, 800, 1600, 3200, 1e9]
    sens = [((dec >= 0) & (y == 1) & (area >= lo) & (area < hi)).sum() / max(((y == 1) & (area >= lo) & (area < hi)).sum(), 1)
            for lo, hi in zip(edges[:-1], edges[1:])]
    ax.bar(range(len(sens)), sens, color=LEAKY)
    ax.set_xticks(range(len(sens)), ["1–2", "2–4", "4–8", "8–16", "16–32", ">32"])
    ax.set_xlabel("tumor area on slice (cm²)")
    ax.set_ylabel("sensitivity")
    ax.set_ylim(0, 1)
    ax.set_title("Sensitivity by tumor area on the slice\n(misses concentrate in small cross-sections)")
    feats, head = config.split("|")[:2]
    name = FEAT_LABEL.get(feats, feats).replace("\n", " ")
    fig.suptitle(f"{name} features, {HEAD_LABEL.get(head, head)} head, patient-level 5-fold CV", y=1.03)
    save(fig, FIG / "fig9_calibration_size.png")


FIGS = {"legacy": fig_legacy, "dataset": fig_dataset, "leakage": fig_leakage, "lopo": fig_lopo,
        "conformal": fig_conformal, "sensitivity": fig_sensitivity, "shift": fig_shift, "shortcut": fig_shortcut,
        "calibration": fig_calibration}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(FIGS))
    args = ap.parse_args()
    setup()
    for name in args.only.split(","):
        FIGS[name]()
        print("figure:", name)


if __name__ == "__main__":
    main()

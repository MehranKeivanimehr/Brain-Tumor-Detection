"""Animated GIFs for the README.

  figures/leakage_neighbours.gif   test slice vs its nearest training slice under a
                                   slice-level and a patient-level split
  figures/conformal_scroll.gif     scrolling through an unseen patient: ground truth,
                                   predicted probability and prediction sets
  figures/patient_fingerprint.gif  2D embedding of slice features coloured by class,
                                   then by patient: slices cluster by patient

Usage: python scripts/10_animations.py [--only leakage,scroll,fingerprint]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from btmri.viz import HONEST, LEAKY, C, plt, setup

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def _frames_to_gif(frames, path, ms=900):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print("gif:", path.name, len(frames), "frames")


def _fig_to_image(fig):
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return im.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)


def _table():
    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    return t


def gif_leakage(features="resnet50", n=10):
    t = _table()
    imgs = np.load(ROOT / "data" / "dataset" / "images.npy", mmap_mode="r")
    nb = np.load(ROOT / "data" / "outputs" / f"neighbours_{features}.npz")
    st, sn, pt, pn = nb["slice_test"], nb["slice_nn"], nb["patient_test"], nb["patient_nn"]
    # queries held out under BOTH splits, so both neighbours belong to the same test slice
    both = np.intersect1d(st, pt)
    pos_s = {r: i for i, r in enumerate(st)}
    pos_p = {r: i for i, r in enumerate(pt)}
    cand = [q for q in both if t.label.values[q] == 1 and t.patient.values[sn[pos_s[q]]] == t.patient.values[q]]
    rng = np.random.default_rng(3)
    pick = rng.choice(cand, n, replace=False)
    frames = []
    for q in pick:
        j_slice, j_pat = sn[pos_s[q]], pn[pos_p[q]]
        fig, ax = plt.subplots(1, 3, figsize=(9.6, 4.0))
        for a, r, title, col in ((ax[0], q, "Test slice", C["ink"]),
                                 (ax[1], j_slice, "Nearest training slice\nslice-level split", LEAKY),
                                 (ax[2], j_pat, "Nearest training slice\npatient-level split", HONEST)):
            a.imshow(imgs[r], cmap="gray", vmin=0, vmax=255)
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True)
                sp.set_color(col)
                sp.set_linewidth(3)
            a.grid(False)
            a.set_title(title, color=col, fontsize=10)
        dz = abs(t.z_mm.values[j_slice] - t.z_mm.values[q])
        ax[0].set_xlabel(f"{t.patient.values[q]}", fontsize=8)
        ax[1].set_xlabel(f"same patient, {dz:.0f} mm away", fontsize=8, color=LEAKY)
        ax[2].set_xlabel(f"{t.patient.values[j_pat]} (another patient)", fontsize=8, color=HONEST)
        fig.suptitle("Why slice-level splits leak: the 'unseen' test slice has a near copy in training "
                     "(ResNet-50 feature space)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        frames.append(_fig_to_image(fig))
    _frames_to_gif(frames, FIG / "leakage_neighbours.gif", 1500)


def gif_scroll(cfg="resnet50_elm", alpha="0.1"):
    from btmri.data.dicom import load_seg_on_grid, load_series
    from btmri.data.slices import build_slices

    t = _table()
    keep = t.label.values >= 0
    tk = t[keep].reset_index(drop=True)
    ex = np.load(ROOT / "data" / "outputs" / f"conformal_sets_{cfg}.npz")
    te = ex["rows"]
    naive, jk = ex[f"naive_slice_split|{alpha}"], ex[f"jackknife+_pooled|{alpha}"]
    pts = tk.patient.values[te]
    y_te = tk.label.values[te]
    cov_n = naive[np.arange(len(te)), y_te]
    cov_j = jk[np.arange(len(te)), y_te]
    # the unseen patient on which the two calibrations differ most
    best, best_pt = -1, None
    for p_ in np.unique(pts):
        sel = pts == p_
        gap = cov_j[sel].sum() - cov_n[sel].sum()
        if sel.sum() >= 20 and gap > best:
            best, best_pt = gap, p_
    sel = np.flatnonzero(pts == best_pt)
    sel = sel[np.argsort(tk.z_mm.values[te][sel])]
    raw = ROOT / "data" / "raw" / "upenn_gbm" / best_pt
    vol = load_series(raw / "flair")
    lab, _ = load_seg_on_grid(next((raw / "seg_ai").glob("*.dcm")), vol)
    ss = build_slices(vol, lab, return_masks=True)
    by_index = {int(k): i for i, k in enumerate(ss.slice_index)}
    names = {0: "no tumor", 1: "tumor"}

    def fmt(sv):
        return "{ }" if sv.sum() == 0 else "{" + ", ".join(names[c] for c in (1, 0) if sv[c]) + "}"

    frames, seen_n, seen_j = [], [], []
    for i in sel:
        r = te[i]
        y = int(tk.label.values[r])
        k = by_index[int(tk.slice_index.values[r])]
        seen_n.append(cov_n[i])
        seen_j.append(cov_j[i])
        fig = plt.figure(figsize=(9.2, 4.4))
        ax = fig.add_axes([0.01, 0.04, 0.43, 0.84])
        ax.imshow(ss.images[k], cmap="gray", vmin=0, vmax=255)
        if (ss.masks[k] > 0).any():
            ax.contour(ss.masks[k] > 0, levels=[0.5], colors=[C["yellow"]], linewidths=1.0)
        ax.axis("off")
        ax.set_title(f"unseen patient {best_pt}   z = {tk.z_mm.values[r]:.0f} mm", fontsize=10)
        tx = fig.add_axes([0.47, 0.04, 0.52, 0.84])
        tx.axis("off")
        tx.text(0, 0.93, f"Reference: {names[y]}", fontsize=12, fontweight="bold")
        tx.text(0, 0.84, f"tumor area {tk.tumor_area_mm2.values[r]:.0f} mm² (yellow outline)", fontsize=9,
                color=C["grey"])
        for yy, (title, sv, col, seen) in zip((0.62, 0.32), (("Naive slice-level calibration", naive[i], LEAKY, seen_n),
                                                              ("Hierarchical jackknife+ (patient level)", jk[i], HONEST,
                                                               seen_j))):
            ok = bool(sv[y])
            tx.text(0, yy + 0.09, title, fontsize=10, color=col, fontweight="bold")
            tx.text(0, yy, f"set {fmt(sv)}", fontsize=11)
            tx.text(0.62, yy, "covers" if ok else "misses", fontsize=10, fontweight="bold",
                    color=C["green"] if ok else C["red"])
            tx.text(0, yy - 0.09, f"coverage so far on this patient: {100 * np.mean(seen):.0f}% "
                                  f"({int(np.sum(seen))}/{len(seen)})", fontsize=9, color=C["grey"])
        tx.text(0, 0.02, f"target: {100 * (1 - float(alpha)):.0f}% of a new patient's slices", fontsize=9,
                color=C["grey"])
        frames.append(_fig_to_image(fig))
    _frames_to_gif(frames, FIG / "conformal_scroll.gif", 750)


def gif_fingerprint(features="resnet50", n_patients=40, n_show=6, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    t = _table()
    keep = t.label.values >= 0
    tk = t[keep].reset_index(drop=True)
    X = np.load(ROOT / "data" / "features" / f"{features}.npy")[np.flatnonzero(keep)]
    rng = np.random.default_rng(seed)
    pts = np.sort(rng.choice(np.unique(tk.patient), n_patients, replace=False))
    sel = np.flatnonzero(tk.patient.isin(pts).values)
    Z = PCA(50, random_state=seed).fit_transform((X[sel] - X[sel].mean(0)) / (X[sel].std(0) + 1e-6))
    E = TSNE(2, perplexity=30, random_state=seed, init="pca").fit_transform(Z)
    y = tk.label.values[sel]
    pat = tk.patient.values[sel]
    z = tk.z_mm.values[sel]
    col = np.array([HONEST, LEAKY])[y]
    show = rng.choice(pts, n_show, replace=False)
    frames = []

    def base(title):
        fig, ax = plt.subplots(figsize=(6.4, 5.8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, fontsize=10)
        return fig, ax

    fig, ax = base(f"ResNet-50 features of {len(sel)} slices from {n_patients} patients (t-SNE)")
    ax.scatter(E[:, 0], E[:, 1], c=col, s=12, linewidths=0)
    ax.scatter([], [], c=LEAKY, label="tumor")
    ax.scatter([], [], c=HONEST, label="no tumor")
    ax.legend(loc="lower right", fontsize=8)
    frames.append(_fig_to_image(fig))
    for p_ in show:
        idx = np.flatnonzero(pat == p_)
        idx = idx[np.argsort(z[idx])]
        fig, ax = base(f"{p_}: its slices form one continuous path\n"
                       "(neighbouring slices sit next to each other in feature space)")
        ax.scatter(E[:, 0], E[:, 1], c="#d0d0d0", s=10, linewidths=0)
        ax.plot(E[idx, 0], E[idx, 1], "-", c=C["ink"], lw=1.2, zorder=2)
        ax.scatter(E[idx, 0], E[idx, 1], c=col[idx], s=34, edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate("top of head", E[idx[-1]], fontsize=7, xytext=(6, 6), textcoords="offset points")
        ax.annotate("skull base", E[idx[0]], fontsize=7, xytext=(6, -10), textcoords="offset points")
        frames.append(_fig_to_image(fig))
    _frames_to_gif(frames, FIG / "patient_fingerprint.gif", 1600)


GIFS = {"leakage": gif_leakage, "scroll": gif_scroll, "fingerprint": gif_fingerprint}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(GIFS))
    args = ap.parse_args()
    setup()
    for name in args.only.split(","):
        GIFS[name]()


if __name__ == "__main__":
    main()

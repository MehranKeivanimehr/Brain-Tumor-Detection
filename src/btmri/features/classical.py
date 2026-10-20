"""Corrected version of the original repository's handcrafted descriptors.

The legacy MATLAB pipeline combined a 3-level Haar DWT, GLCM statistics and a
59-bin LBP histogram. Its audit (docs/legacy_audit.md) found that the
GLCM/IDM implementation was wrong (IDM computed over pixel coordinates), that
"smoothness" was constant, and that the Gabor+GMM segmentation front-end hurt.
This module keeps the same descriptor families, computed on the whole slice:

* first-order intensity statistics inside the head (9)
* Haar DWT level 3: mean |c|, std and log-energy of the 10 sub-bands (30)
* GLCM on 32 grey levels, distances {1, 3}, averaged over 4 angles:
  contrast, dissimilarity, homogeneity (= inverse difference moment),
  energy, correlation, ASM (12)
* rotation-invariant uniform LBP histograms, (P, R) = (8, 1) and (16, 2) (28)
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def feature_names() -> list[str]:
    names = [f"fo_{n}" for n in ("mean", "std", "skew", "kurt", "entropy", "p10", "p50", "p90", "p99")]
    bands = ["cA3"] + [f"c{o}{lvl}" for lvl in (3, 2, 1) for o in ("H", "V", "D")]
    names += [f"dwt_{b}_{s}" for b in bands for s in ("absmean", "std", "logenergy")]
    names += [f"glcm_d{d}_{p}" for d in (1, 3) for p in GLCM_PROPS]
    names += [f"lbp8_{i}" for i in range(10)] + [f"lbp16_{i}" for i in range(18)]
    return names


def classical_features(img: np.ndarray) -> np.ndarray:
    """79 descriptors from a 2D uint8 slice."""
    x = img.astype(np.float64) / 255.0
    head = img > 8
    v = x[head] if head.sum() > 50 else x.ravel()
    hist, _ = np.histogram(v, bins=64, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    fo = [v.mean(), v.std(), stats.skew(v), stats.kurtosis(v), -np.sum(p[p > 0] * np.log2(p[p > 0])),
          *np.percentile(v, [10, 50, 90, 99])]

    coeffs = pywt.wavedec2(x, "haar", level=3)
    bands = [coeffs[0]] + [c for detail in coeffs[1:] for c in detail]
    dwt = []
    for b in bands:
        dwt += [np.abs(b).mean(), b.std(), np.log1p(np.sum(b * b))]

    q = (img // 8).astype(np.uint8)  # 32 grey levels
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32,
                        symmetric=True, normed=True)
    gl = []
    for di in range(2):
        for prop in GLCM_PROPS:
            gl.append(graycoprops(glcm, prop)[di].mean())

    lbp = []
    for P, R in ((8, 1), (16, 2)):
        codes = local_binary_pattern(img, P, R, method="uniform")
        h, _ = np.histogram(codes, bins=np.arange(P + 3) - 0.5)
        lbp += list(h / h.sum())
    return np.asarray(fo + dwt + gl + lbp, dtype=np.float32)


def batch_classical(images: np.ndarray, workers: int = 4) -> np.ndarray:
    import multiprocessing as mp

    with mp.Pool(workers) as pool:
        feats = pool.map(classical_features, list(images), chunksize=64)
    return np.stack(feats)

"""Turn a FLAIR volume + tumor segmentation into a labelled 2D slice set.

Design choices (all label-independent except the label itself):

* in-plane resampling to 1 mm isotropic pixels and a 240 x 240 mm field of
  view centred on the head, so pixel size cannot act as a scanner shortcut;
* intensities scaled per volume to the 1st-99.5th percentile of head voxels;
* thin-slice volumes are subsampled to >= 3 mm spacing so every patient
  contributes a comparable number of near-independent slices;
* slices covering < 30% of the largest head cross-section are dropped
  (skull vertex / skull base);
* label 1 if the whole-tumor area (necrosis + edema + enhancing) is at least
  ``min_tumor_mm2``, label 0 if the slice has no tumor voxel at all, and -1
  (excluded from training/evaluation) for the thin band in between.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.transform import resize

from .dicom import Volume

FOV_MM = 240
MIN_TUMOR_MM2 = 100.0
MIN_HEAD_FRACTION = 0.30
TARGET_SLICE_SPACING_MM = 3.0


@dataclass
class SliceSet:
    images: np.ndarray  # (n, 240, 240) uint8
    slice_index: np.ndarray  # index into the source volume
    z_mm: np.ndarray  # position along the slice normal
    head_area_mm2: np.ndarray
    tumor_area_mm2: np.ndarray  # whole tumor, primary (AI) segmentation
    enh_area_mm2: np.ndarray  # enhancing lesion
    label: np.ndarray  # 1 tumor, 0 no tumor, -1 ambiguous
    ref_tumor_area_mm2: np.ndarray | None = None  # reference (radiologist) segmentation
    ref_label: np.ndarray | None = None
    lesion_z: float = float("nan")  # FLAIR z-score inside the tumor (sanity check)
    alignment_margin: float = float("nan")  # lesion_z minus best flipped-mask lesion_z
    masks: np.ndarray | None = None  # (n, 240, 240) uint8 label maps, only if requested (figures)


def head_mask(vol: np.ndarray) -> np.ndarray:
    """Foreground (head) mask: Otsu threshold, slice-wise hole filling, largest component."""
    v = vol[vol > 0]
    thr = threshold_otsu(v) if v.size else 0.0
    m = vol > 0.5 * thr
    out = np.zeros_like(m)
    for k in range(m.shape[0]):
        s = ndimage.binary_opening(m[k], iterations=2)
        s = ndimage.binary_fill_holes(s)
        lab, n = ndimage.label(s)
        if n:
            sizes = ndimage.sum(s, lab, range(1, n + 1))
            s = lab == (1 + int(np.argmax(sizes)))
        out[k] = s
    return out


def lesion_contrast(vol: np.ndarray, tumor: np.ndarray, head: np.ndarray) -> float:
    """Mean FLAIR z-score (within the head) of tumor voxels. Edema is FLAIR-hyperintense,
    so a correctly registered mask gives clearly positive values."""
    hv = vol[head]
    if not tumor.any() or hv.size == 0:
        return float("nan")
    return float((vol[tumor].mean() - hv.mean()) / (hv.std() + 1e-6))


def alignment_margin(vol: np.ndarray, tumor: np.ndarray, head: np.ndarray) -> float:
    """Lesion contrast of the decoded mask minus the best contrast obtained by
    flipping it along rows or columns. A correctly registered mask sits on
    FLAIR-hyperintense tissue, so the margin is positive; a mask decoded with
    the wrong orientation gives a negative margin."""
    z = lesion_contrast(vol, tumor, head)
    alt = max(lesion_contrast(vol, tumor[:, ::-1, :], head), lesion_contrast(vol, tumor[:, :, ::-1], head))
    return float(z - alt)


def _crop_center(img: np.ndarray, center: tuple[float, float], size: int) -> np.ndarray:
    out = np.zeros((size, size), dtype=img.dtype)
    r0 = int(round(center[0])) - size // 2
    c0 = int(round(center[1])) - size // 2
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + size, img.shape[0]), min(c0 + size, img.shape[1])
    out[rs - r0 : re - r0, cs - c0 : ce - c0] = img[rs:re, cs:ce]
    return out


def build_slices(
    vol: Volume,
    labels: np.ndarray,
    ref_labels: np.ndarray | None = None,
    min_tumor_mm2: float = MIN_TUMOR_MM2,
    return_masks: bool = False,
) -> SliceSet:
    px_area = vol.spacing[0] * vol.spacing[1]
    pix = vol.pixels
    head = head_mask(pix)
    tumor = labels > 0
    contrast = lesion_contrast(pix, tumor, head)

    # z subsampling for thin-slice acquisitions
    stride = max(1, int(round(TARGET_SLICE_SPACING_MM / max(vol.slice_spacing, 1e-3))))
    keep = np.arange(0, pix.shape[0], stride)
    head_area = head.reshape(head.shape[0], -1).sum(1) * px_area
    keep = keep[head_area[keep] >= MIN_HEAD_FRACTION * head_area.max()]

    hv = pix[head]
    lo, hi = np.percentile(hv, [1, 99.5])
    norm = np.clip((pix - lo) / (hi - lo + 1e-6), 0, 1)

    out_shape = (int(round(pix.shape[1] * vol.spacing[0])), int(round(pix.shape[2] * vol.spacing[1])))
    scale = (out_shape[0] / pix.shape[1], out_shape[1] / pix.shape[2])
    com = ndimage.center_of_mass(head[keep].any(0))
    center = (com[0] * scale[0], com[1] * scale[1])

    imgs = []
    for k in keep:
        r = resize(norm[k], out_shape, order=1, anti_aliasing=True, preserve_range=True)
        imgs.append(_crop_center(r, center, FOV_MM))
    images = (np.stack(imgs) * 255).round().astype(np.uint8)

    def areas(lbl):
        t = (lbl > 0).reshape(lbl.shape[0], -1).sum(1) * px_area
        return t[keep]

    def to_label(area):
        y = np.full(area.shape, -1, dtype=np.int8)
        y[area >= min_tumor_mm2] = 1
        y[area == 0] = 0
        return y

    t_area = areas(labels)
    enh = ((labels == 3).reshape(labels.shape[0], -1).sum(1) * px_area)[keep]
    ss = SliceSet(
        images=images,
        slice_index=keep,
        z_mm=(vol.origins @ vol.normal)[keep],
        head_area_mm2=head_area[keep],
        tumor_area_mm2=t_area,
        enh_area_mm2=enh,
        label=to_label(t_area),
        lesion_z=contrast,
        alignment_margin=alignment_margin(pix, tumor, head) if tumor.any() else float("nan"),
    )
    if return_masks:
        ss.masks = np.stack([_crop_center(resize(labels[k], out_shape, order=0, preserve_range=True,
                                                 anti_aliasing=False).astype(np.uint8), center, FOV_MM)
                             for k in keep])
    if ref_labels is not None:
        ra = areas(ref_labels)
        ss.ref_tumor_area_mm2 = ra
        ss.ref_label = to_label(ra)
    return ss

"""Geometry-aware loading of MR series and DICOM SEG objects.

Why geometry matters: the AIMI segmentations in IDC store their frames with the
column direction cosine reversed relative to the source FLAIR images (row order
flipped, origin at the opposite edge). Decoding frames by referenced SOP
Instance UID alone therefore returns masks that are upside down. Here every
segmentation pixel is mapped into the source image grid through patient
coordinates, so flips/transpositions are handled exactly and anything that is
not an exact grid correspondence raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom


@dataclass
class Volume:
    """A stack of parallel 2D slices sorted along the slice normal."""

    pixels: np.ndarray  # (n_slices, rows, cols) float32
    origins: np.ndarray  # (n_slices, 3) ImagePositionPatient per slice
    row_cos: np.ndarray  # direction of increasing column index (3,)
    col_cos: np.ndarray  # direction of increasing row index (3,)
    spacing: tuple[float, float]  # (row spacing, column spacing) in mm
    slice_spacing: float
    sop_uids: list[str]

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.row_cos, self.col_cos)


def load_series(directory: Path) -> Volume:
    """Load a single-frame MR series into a :class:`Volume`."""
    files = sorted(Path(directory).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no DICOM files in {directory}")
    ds = [pydicom.dcmread(f) for f in files]
    iop = np.asarray(ds[0].ImageOrientationPatient, dtype=float)
    row_cos, col_cos = iop[:3], iop[3:]
    normal = np.cross(row_cos, col_cos)
    ds.sort(key=lambda d: float(np.dot(np.asarray(d.ImagePositionPatient, float), normal)))
    pix = []
    for d in ds:
        a = d.pixel_array.astype(np.float32)
        slope = float(getattr(d, "RescaleSlope", 1.0))
        inter = float(getattr(d, "RescaleIntercept", 0.0))
        pix.append(a * slope + inter)
    origins = np.array([np.asarray(d.ImagePositionPatient, float) for d in ds])
    proj = origins @ normal
    slice_spacing = float(np.median(np.diff(proj))) if len(ds) > 1 else float(ds[0].SliceThickness)
    ps = [float(x) for x in ds[0].PixelSpacing]
    return Volume(
        pixels=np.stack(pix),
        origins=origins,
        row_cos=row_cos,
        col_cos=col_cos,
        spacing=(ps[0], ps[1]),
        slice_spacing=slice_spacing,
        sop_uids=[d.SOPInstanceUID for d in ds],
    )


def is_axial(vol: Volume, tol: float = 0.9) -> bool:
    """True if the slice normal is within ~25 degrees of the head-foot axis."""
    return abs(vol.normal[2]) >= tol


def _frame_geometry(seg: pydicom.Dataset, i: int):
    pf = seg.PerFrameFunctionalGroupsSequence[i]
    sh = seg.SharedFunctionalGroupsSequence[0]

    def get(name):
        return getattr(pf, name, None) or getattr(sh, name)

    iop = np.asarray(get("PlaneOrientationSequence")[0].ImageOrientationPatient, float)
    pos = np.asarray(pf.PlanePositionSequence[0].ImagePositionPatient, float)
    ps = [float(x) for x in get("PixelMeasuresSequence")[0].PixelSpacing]
    seg_no = int(pf.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
    src_uid = None
    if "DerivationImageSequence" in pf:
        src = pf.DerivationImageSequence[0].SourceImageSequence[0]
        src_uid = src.ReferencedSOPInstanceUID
    return iop, pos, ps, seg_no, src_uid


def load_seg_on_grid(seg_path: Path, vol: Volume, atol_mm: float = 0.3) -> tuple[np.ndarray, dict]:
    """Decode a BINARY DICOM SEG into a label volume on ``vol``'s voxel grid.

    Returns ``(labels, info)`` where ``labels`` has the shape of ``vol.pixels``
    and holds the segment number (0 = background; later segments overwrite
    earlier ones where they overlap). ``info`` reports segment names and
    geometry checks (number of frames, UID agreement, detected flips).
    """
    seg = pydicom.dcmread(seg_path)
    if str(getattr(seg, "SegmentationType", "BINARY")).upper() != "BINARY":
        raise ValueError("only BINARY segmentations are supported")
    frames = seg.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    names = {int(s.SegmentNumber): str(s.SegmentLabel) for s in seg.SegmentSequence}
    labels = np.zeros(vol.pixels.shape, dtype=np.uint8)
    n_slices, rows, cols = vol.pixels.shape
    normal = vol.normal
    slice_proj = vol.origins @ normal
    uid_to_slice = {u: k for k, u in enumerate(vol.sop_uids)}
    uid_mismatch = 0
    flips = set()
    for i in range(frames.shape[0]):
        iop, pos, ps, seg_no, src_uid = _frame_geometry(seg, i)
        rc, cc = iop[:3], iop[3:]
        # slice index from the frame's position along the volume normal
        k = int(np.argmin(np.abs(slice_proj - pos @ normal)))
        if abs(slice_proj[k] - pos @ normal) > atol_mm:
            raise ValueError(f"frame {i} does not lie on any source slice")
        if src_uid is not None and uid_to_slice.get(src_uid, k) != k:
            uid_mismatch += 1
        fr = frames[i].astype(bool)
        r_idx, c_idx = np.nonzero(fr)
        if r_idx.size == 0:
            continue
        # patient coordinates of segmentation pixels -> source grid indices
        pts = pos[None, :] + c_idx[:, None] * ps[1] * rc[None, :] + r_idx[:, None] * ps[0] * cc[None, :]
        d = pts - vol.origins[k][None, :]
        src_c = d @ vol.row_cos / vol.spacing[1]
        src_r = d @ vol.col_cos / vol.spacing[0]
        rr, cci = np.rint(src_r).astype(int), np.rint(src_c).astype(int)
        if np.max(np.abs(src_r - rr)) > 0.05 or np.max(np.abs(src_c - cci)) > 0.05:
            raise ValueError("segmentation grid is not aligned with the source grid")
        if rr.min() < 0 or cci.min() < 0 or rr.max() >= rows or cci.max() >= cols:
            raise ValueError("segmentation pixels fall outside the source image")
        labels[k, rr, cci] = seg_no
        if np.dot(cc, vol.col_cos) < 0:
            flips.add("rows")
        if np.dot(rc, vol.row_cos) < 0:
            flips.add("cols")
    info = {
        "segments": names,
        "n_frames": int(frames.shape[0]),
        "uid_mismatch_frames": uid_mismatch,
        "flipped_axes": sorted(flips),
    }
    return labels, info

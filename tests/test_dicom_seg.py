"""Synthetic DICOM SEG whose frames are stored row-flipped relative to the source
images (the geometry found in the AIMI segmentations): decoding must follow the
patient-coordinate geometry, not just the referenced SOP Instance UIDs."""

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.pixels import pack_bits
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from btmri.data.dicom import Volume, load_seg_on_grid


def _volume(n=3, rows=6, cols=5, sp=0.9):
    origins = np.array([[-10.0, -20.0, 5.0 + 3 * k] for k in range(n)])
    return Volume(pixels=np.zeros((n, rows, cols), np.float32), origins=origins,
                  row_cos=np.array([1.0, 0, 0]), col_cos=np.array([0, 1.0, 0]), spacing=(sp, sp),
                  slice_spacing=3.0, sop_uids=[generate_uid() for _ in range(n)])


def _seg_file(tmp_path, vol, mask):
    """Write a SEG storing ``mask`` with rows flipped (column cosine negated)."""
    n, rows, cols = mask.shape
    sp = vol.spacing[0]
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "SEG"
    ds.SegmentationType = "BINARY"
    ds.Rows, ds.Columns, ds.NumberOfFrames = rows, cols, n
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 1, 1, 0, 0
    seg = Dataset()
    seg.SegmentNumber, seg.SegmentLabel = 1, "Tumor"
    ds.SegmentSequence = Sequence([seg])
    shared = Dataset()
    po = Dataset()
    po.ImageOrientationPatient = [1, 0, 0, 0, -1, 0]  # rows run the other way
    pm = Dataset()
    pm.PixelSpacing, pm.SliceThickness = [sp, sp], 3.0
    shared.PlaneOrientationSequence, shared.PixelMeasuresSequence = Sequence([po]), Sequence([pm])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])
    frames, per = [], []
    for k in range(n):
        f = Dataset()
        pp = Dataset()
        top = vol.origins[k] + (rows - 1) * sp * vol.col_cos  # origin at the opposite edge
        pp.ImagePositionPatient = [float(v) for v in top]
        si = Dataset()
        si.ReferencedSegmentNumber = 1
        src = Dataset()
        src.ReferencedSOPInstanceUID = vol.sop_uids[k]
        der = Dataset()
        der.SourceImageSequence = Sequence([src])
        f.PlanePositionSequence, f.SegmentIdentificationSequence = Sequence([pp]), Sequence([si])
        f.DerivationImageSequence = Sequence([der])
        per.append(f)
        frames.append(mask[k][::-1, :])
    ds.PerFrameFunctionalGroupsSequence = Sequence(per)
    ds.PixelData = pack_bits(np.stack(frames).astype(np.uint8))
    path = tmp_path / "seg.dcm"
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    return path


def test_flipped_seg_is_decoded_onto_the_source_grid(tmp_path):
    vol = _volume()
    mask = np.zeros(vol.pixels.shape, np.uint8)
    mask[1, 0:2, 1:4] = 1  # asymmetric blob near the top rows
    labels, info = load_seg_on_grid(_seg_file(tmp_path, vol, mask), vol)
    assert info["flipped_axes"] == ["rows"]
    assert info["uid_mismatch_frames"] == 0
    np.testing.assert_array_equal(labels > 0, mask > 0)
    # decoding by UID alone would have put the blob at the bottom of the slice
    assert not np.array_equal(pydicom.dcmread(_seg_file(tmp_path, vol, mask)).pixel_array[1] > 0, mask[1] > 0)

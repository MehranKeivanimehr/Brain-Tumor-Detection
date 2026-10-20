# Data

## Source

| | |
|---|---|
| Images | UPENN-GBM (Bakas et al., *Sci Data* 2022), TCIA DOI [10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49), CC BY 4.0 |
| Segmentations | AIMI annotations (Murugesan et al., *Sci Data* 2025), DOI [10.5281/zenodo.8345959](https://doi.org/10.5281/zenodo.8345959), CC BY 4.0 |
| Access | NCI Imaging Data Commons, index v25, via `idc-index`; files fetched over HTTPS from the public `idc-open-data` bucket |
| Series used | the processed axial FLAIR (`… : Processed_CaPTk`) of the pre-operative study, one per patient |
| Labels | AIMI automatic nnU-Net segmentation of that FLAIR study (necrosis, edema, enhancing lesion); for 52 patients also the radiologist-corrected version |

`results/cohort_manifest.csv` lists the exact Study/Series Instance UIDs of every
series used, so the cohort can be re-downloaded bit for bit
(`python scripts/01_download_upenn_gbm.py`). No image data are stored in git.

## Geometry-aware segmentation decoding

The AIMI DICOM-SEG objects reference each FLAIR slice by SOP Instance UID, but
store their frames with the column direction cosine reversed
(`ImageOrientationPatient = [1,0,0, 0,-1,0]` against `[1,0,0, 0,1,0]` in the
FLAIR) and with the frame origin at the opposite edge. Decoding frames by UID
alone therefore returns masks that are upside down. **All 540 segmentations
decoded in this cohort have this property** (`results/qc_patients.csv`,
column `seg_flipped`).

`btmri.data.dicom.load_seg_on_grid` maps every segmentation pixel through
patient coordinates into the FLAIR grid and raises if the mapping is not an
exact grid correspondence. Two independent checks guard it:

* a unit test that writes a synthetic row-flipped SEG and checks that decoding
  by geometry recovers the mask while decoding by UID does not
  (`tests/test_dicom_seg.py`);
* a per-patient image check: the decoded mask must lie on more
  FLAIR-hyperintense tissue than the same mask flipped along rows or columns
  ("alignment margin" > 0). Minimum margin over the cohort: 0.10; median 1.03
  head-normalised standard deviations.

## Slice dataset

| step | rule |
|---|---|
| in-plane resampling | 1 mm pixels, 240 × 240 mm field of view centred on the head (pixel size cannot act as a scanner shortcut) |
| intensity | per volume, 1st–99.5th percentile of head voxels mapped to [0, 1] |
| through-plane | thin-slice volumes subsampled to ≥ 3 mm spacing |
| slice selection | head cross-section ≥ 30% of the largest one in the volume |
| positive | whole-tumor area ≥ 1 cm² (≈ 114 native pixels) |
| negative | no tumor voxel at all |
| excluded | 0 < area < 1 cm² (795 slices) |

## Cohort flow

| | patients |
|---|---|
| UPENN-GBM patients with an AIMI-segmented FLAIR in IDC v25 | 541 |
| excluded: FLAIR not axial | 1 |
| excluded: automatic segmentation empty or < 1 cm² on every slice | 21 |
| **analysed** | **519** (17,147 labelled slices: 8,524 tumor, 8,623 no tumor) |
| with a radiologist-corrected segmentation | 52 |

Scanners: 400 Siemens TrioTim (3 T), 40 Verio (3 T), 29 Espree (1.5 T),
17 Avanto (1.5 T), 9 Aera (1.5 T), 7 Symphony Tim (1.5 T) and 17 others
(GE Discovery MR750w ×5 and Optima MR450w ×2, Siemens MAGNETOM Vida ×4 and
Avanto fit ×2, and four with missing or software-only model names). Field
strength from the DICOM headers: 446 patients at 3 T, 66 at 1.5 T, 7 not
recorded (4 TrioTim, 2 Verio, 1 Vida, all 3 T systems).

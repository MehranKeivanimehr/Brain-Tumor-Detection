# Brain Tumor Detection from MRI (2023 version)

This branch keeps the original 2023 version of the project, unchanged: a
classical image-processing and machine-learning pipeline that classifies 2D
brain MRI images as **tumor** or **no tumor**. It is implemented twice, once in
MATLAB and once in a Python notebook.

> The current version of the project is on the
> [`main`](https://github.com/MehranKeivanimehr/Brain-Tumor-Detection) branch.
> It audits this pipeline and rebuilds the study on 519 glioblastoma patients
> from the UPENN-GBM collection.

## Dataset

[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
(Chakrabarty, Kaggle 2019): 253 JPEG images, 98 without and 155 with a tumor.
The images are not included in this repository. The code expects them in the
following layout:

```
DataSet/
  Train/no/   No1.jpg ... No95.jpg
  Train/yes/  Y1.jpg  ... Y155.jpg
  Test/       No1.jpg ... No8.jpg, Y1.jpg ... Y13.jpg
```

## MATLAB pipeline

`main.m` runs the whole pipeline on the training and test folders.

1. **Pre-processing** (`Pre_Processing.m`): resize to 256 × 256, convert to
   grayscale, 3 × 3 median filter.
2. **Segmentation** (`gabor_filters.m`): Gabor filter bank (4 orientations,
   several wavelengths), Gaussian smoothing of the magnitude responses, then a
   2-component Gaussian mixture model clusters the pixels.
3. **Mask clean-up** (`importdata.m`): binarise, fill holes, erode (diamond,
   radius 3) and dilate (5 × 5 square).
4. **Features** (`Feature_extraction.m`, `importdata.m`), 69 per image:
   * 10 features of the mask after a 3-level Haar wavelet transform: GLCM
     contrast, energy and homogeneity, plus mean, standard deviation, entropy,
     RMS, variance, smoothness and inverse difference moment;
   * 59 local binary pattern (LBP) features of the pre-processed image.
5. **Classifier** (`elm.m`): extreme learning machine with 160 hidden neurons
   and a sine activation. `main.m` trains it 10 times with different random
   weights and reports one run.

The extracted features are saved in `Trainsetnew0.5.mat` (250 × 69, with
labels) and `Testsetnew0.5.mat` (21 × 69, with labels), so the classifier can
be run without the images.

## Python notebook

`BrainTumorClassification.ipynb` (Google Colab) implements a lighter variant:

1. grayscale, resize to 256 × 256, 5 × 5 median blur;
2. k-means (K = 8) segmentation, contour filtering by area and solidity, and
   the largest connected component as the tumor region;
3. 12 features: GLCM contrast, homogeneity, ASM, energy and correlation of the
   segmented region, plus 7 features from a uniform LBP histogram (radius 2,
   16 points);
4. two classifiers: a linear SVM and an extreme learning machine with 82
   tanh hidden neurons.

## Files

| File | Purpose |
|---|---|
| `main.m` | Runs feature extraction on the training and test folders, then the ELM |
| `Pre_Processing.m` | Resize, grayscale and median filter |
| `gabor_filters.m` | Gabor features and Gaussian-mixture segmentation |
| `importdata.m` | Reads a folder of images, cleans the mask and builds the 69 features |
| `Feature_extraction.m` | Wavelet, GLCM and statistical features of the mask |
| `elm.m` | Extreme learning machine (training and testing) |
| `Trainsetnew0.5.mat`, `Testsetnew0.5.mat` | Saved training and test features with labels |
| `BrainTumorClassification.ipynb` | Python version with an SVM and an ELM |

## Requirements and usage

**MATLAB** with the Image Processing, Statistics and Machine Learning, Wavelet
and Computer Vision toolboxes. The dataset paths in `main.m` are hard-coded;
point the four `addr_folder` variables to your copy of the dataset, then run
`main.m`.

**Python** (the notebook was run on Google Colab): `numpy`, `pandas`,
`opencv-python`, `scikit-image`, `scikit-learn`, `imutils` and `matplotlib`.
The notebook reads the dataset from
`/content/drive/MyDrive/BrainTumurDiagnosis/Dataset/` on Google Drive.

## Results reported in 2023

The notebook reports 90.0% test accuracy for both the linear SVM and the ELM
(ELM training accuracy 94.6%).

**Note on these numbers.** A later audit, documented in
[`docs/legacy_audit.md`](https://github.com/MehranKeivanimehr/Brain-Tumor-Detection/blob/main/docs/legacy_audit.md)
on the `main` branch, showed that this evaluation does not measure performance
on new images. 18 of the 21 test images are copies of training images,
`main.m` chooses its run by test accuracy, and the notebook counts each test
image twice. Under cross-validation that keeps
duplicate images together, the MATLAB ELM reaches about 54% and logistic
regression on the same features about 73%. A classifier that always answers
"tumor" scores 62% on this data.

## Acknowledgements

`elm.m` follows the reference extreme learning machine code of G.-B. Huang
(Huang, Zhu & Siew, *Extreme learning machine: theory and applications*,
Neurocomputing 70, 2006).

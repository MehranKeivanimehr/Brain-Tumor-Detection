# Methods

This page gives the full definitions behind the results in the README. The code
paths are `src/btmri/heads.py`, `src/btmri/conformal.py`, `src/btmri/data/`.

## 1. Task and unit of analysis

Binary slice classification on axial FLAIR MRI: does this slice contain at least
1 cm² of tumor (necrosis + edema + enhancing lesion)? Slices with no tumor voxel
are negatives; slices with 0–1 cm² are excluded as ambiguous. Positives and
negatives come from **the same patients, scanners and acquisitions**, so the
classes cannot be separated by source (the defect of the Kaggle tumor/no-tumor
datasets, see `docs/legacy_audit.md`).

The statistical unit is the **patient**. Every split, every tuning step, every
calibration step and every confidence interval respects it:

| step | unit |
|---|---|
| outer cross-validation (5 folds) | patients (`StratifiedGroupKFold`) |
| hyper-parameter selection | exact leave-one-patient-out on the training fold |
| probability calibration (Platt) | out-of-patient decision values of the training fold |
| conformal calibration | patients (Section 4) |
| 95% confidence intervals | patient-level cluster bootstrap |

The slice-level protocol (random K-fold over slices) is run only to measure how
much it leaks.

## 2. Closed-form heads

For a feature vector `x` (standardised with training-fold statistics) the head
uses a fixed feature map

* **ridge:** `φ(x) = [1, x]`
* **extreme learning machine (ELM) / random features:** `φ(x) = [1, tanh(Wx + b)]`,
  `W_ij ~ N(0, 1/d)`, `b_j ~ U(-1, 1)`, 2048 hidden units, fixed seed
  (Huang et al. 2006; Rahimi & Recht 2007)

and generalised ridge regression on targets `t ∈ {−1, +1}` with an unpenalised
intercept:

```
A = Φᵀ Φ + diag(0, λ, …, λ),        B = A⁻¹ Φᵀ T,        f(x) = φ(x)ᵀ B
```

### Exact leave-one-patient-out (LOPO)

Let `G` be the rows of one patient, `H_GG = Φ_G A⁻¹ Φ_Gᵀ` and `E_G = T_G − Φ_G B`.
By the Woodbury identity (subset-deletion formulas, Cook & Weisberg 1982):

```
B₋G             = B − A⁻¹ Φ_Gᵀ (I − H_GG)⁻¹ E_G
T_G − Φ_G B₋G   = (I − H_GG)⁻¹ E_G
```

Derivation: `A₋G⁻¹ = A⁻¹ + A⁻¹Φ_Gᵀ M Φ_G A⁻¹` with `M = (I − H_GG)⁻¹`, so
`B₋G = A₋G⁻¹(ΦᵀT − Φ_GᵀT_G) = B + A⁻¹Φ_Gᵀ[M Φ_G B − (I + M H_GG) T_G]`, and
`I + M H_GG = M`.

One Cholesky factorisation of `A` plus one `n_G × n_G` solve per patient
therefore yields (i) the out-of-patient prediction for every training slice and
(ii) the model trained without each patient, whose prediction on any new slice
is `f(x) + φ(x)ᵀ ΔB_G`. With ~430 training patients and ~40 slices each this
replaces 430 refits. `tests/test_heads.py` checks both quantities against
brute-force refits to 1e-8.

The input standardisation is estimated on the whole training fold and then held
fixed; the identities are exact for that fixed transform. Re-estimating it
without each patient changes the out-of-patient decisions negligibly: on the
real ResNet-50 features (`scripts/12_lopo_validation.py`) the correlation with
full refits is 0.9995–0.99995 and the sign agreement 99.1–99.8% for 25–100
training patients, growing with cohort size. The
penalty `λ ∈ {1, 10, 100, 10³, 10⁴}` is chosen by exact LOPO AUC, ties going to
the stronger penalty.

## 3. Baselines

* logistic regression on standardised features, `C ∈ {0.001, 0.01, 0.1}`
  chosen by inner 3-fold CV of the same kind as the outer protocol;
* 15-nearest-neighbours (Euclidean, standardised features): no training, so it
  exposes leakage most directly.

## 4. Conformal prediction with patients as the exchangeable unit

Score for decision value `f` and label `y ∈ {0, 1}`: `s(x, y) = −(2y − 1) f(x)`.
A prediction set contains every label whose score is below a calibrated
threshold. Slices from the same patient are correlated, so exchangeability holds
between patients, not between slices (Dunn, Wasserman & Ramdas 2023; Lee, Barber
& Willett 2026). The target is coverage for **a random slice of a new patient**,
averaged over patients (no method here promises `1 − α` for every individual
patient; the per-patient spread is reported separately).

| method | calibration data | guarantee for a new patient |
|---|---|---|
| naive slice split | 25% of training *slices* (their patients are also in the training 75%) | none |
| patient split | all slices of 25% held-out training patients, ordinary quantile | approximate |
| patient subsample ("subsampling once", Dunn et al. 2023) | one random slice per held-out patient, ordinary quantile | `≥ 1 − α` (finite sample) |
| hierarchical CP (Lee, Barber & Willett 2026) | `F(t) = (1/m) Σ_g F̂_g(t)`, threshold `inf{t : F(t) ≥ (1−α)(1 + 1/m)}` | `≥ 1 − α` (finite sample; `≤ 1 − α + 2/(m+1)` without ties) |
| jackknife+ (subsample) | LOPO models of *all* training patients, one random slice each | `≥ 1 − 2α` (Barber et al. 2021, patients as units) |
| **hierarchical jackknife+** (pooled) | LOPO models, all slices, each patient weighted equally | `≥ 1 − 2α` (Lee, Barber & Willett 2026, Thm 5, stated for residual scores; the classification-score form follows the same comparison argument, cf. Romano et al. 2020); ≈ `1 − α` in practice |
| CV+ (10 folds, pooled) | 10 refitted patient-fold models | approximate |

Patient-level jackknife+ includes label `y` for a test slice `x` iff

```
Σ_i  w_i(x, y)  <  (1 − α)(m + 1),
w_i(x, y) = 1[ s(x, y; f₋ᵢ) > s(xᵢ, yᵢ; f₋ᵢ) ]              (subsample: one slice of patient i)
w_i(x, y) = (1/nᵢ) Σⱼ 1[ s(x, y; f₋ᵢ) > s(xᵢⱼ, yᵢⱼ; f₋ᵢ) ]   (pooled: all slices of patient i)
```

where `f₋ᵢ` is the exact leave-patient-`i`-out model of Section 2. This uses
every training patient for both fitting and calibration, at the cost of one fit.
The pooled form is the hierarchical jackknife+ of Lee, Barber & Willett (2026,
eq. 20) with the regression residual replaced by the classification score.

Two caveats on the guarantees. (i) They require the learning algorithm to be
symmetric in patients and the LOPO models to be the algorithm applied without
patient `i`; here the input standardisation and the penalty `λ` are chosen once
on all training patients and then held fixed, so the LOPO models are exact for
that fixed pipeline but not refits of the whole pipeline. (ii) All guarantees are
marginal over new patients; per-patient coverage can be lower and is reported.

**Mondrian** variants calibrate each class separately, giving class-conditional
coverage; for the tumor class this is a guarantee on the sensitivity of the set.

Within each outer fold the head (features, ELM width, penalty `λ`) is fixed
first: `λ` is chosen once by exact LOPO on the training patients and shared by
every calibration method, so the methods differ only in which data they
calibrate on.

Reported per outer fold and averaged over 5 folds × 3 repeats: patient-weighted
coverage, class-conditional coverage, the 10th percentile of per-patient
coverage, mean set size and the fraction of singleton / two-label / empty sets.

## 5. Metrics

AUC, accuracy, balanced accuracy, sensitivity and specificity at the decision
threshold 0 (ridge/ELM) or 0.5 (logistic, kNN); Brier score and expected
calibration error (15 equal-mass bins) of Platt-calibrated probabilities. All
are computed on pooled out-of-fold predictions, with 95% intervals from 500
patient-level bootstrap resamples.

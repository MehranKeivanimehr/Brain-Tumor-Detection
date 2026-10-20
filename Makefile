# One command per stage; `make all` reproduces every result and figure from scratch (CPU, ~4-5 h).
PY ?= python
export PYTHONPATH := src

.PHONY: all test legacy data slices features benchmark lopo conformal shift audit neighbours shortcut figures tables

all: test legacy data slices features benchmark lopo conformal shift audit neighbours shortcut figures tables

test:
	$(PY) -m pytest -q

legacy:            ## chapter zero: audit of the original MATLAB pipeline
	$(PY) scripts/00_legacy_audit.py

data:              ## UPENN-GBM FLAIR + AIMI segmentations from NCI Imaging Data Commons (~3.5 GB)
	$(PY) scripts/01_download_upenn_gbm.py

slices:
	$(PY) scripts/02_build_slices.py

features:
	$(PY) scripts/03_extract_features.py

benchmark:
	$(PY) scripts/04_benchmark.py --seeds 0

lopo:              ## closed-form leave-one-patient-out vs brute-force refits
	$(PY) scripts/12_lopo_validation.py

conformal:
	for f in classical resnet50; do for h in ridge elm; do \
	  $(PY) scripts/05_conformal.py --features $$f --head $$h --repeats 3; done; done

shift:
	$(PY) scripts/06_scanner_shift.py --features resnet50 --head elm --repeats 5

audit:
	$(PY) scripts/07_label_audit.py --config "resnet50|elm|patient|0"

neighbours:
	$(PY) scripts/08_neighbour_leakage.py --features resnet50

shortcut:
	$(PY) scripts/13_shortcut.py extract --features resnet50
	$(PY) scripts/13_shortcut.py run --features resnet50 --head elm

figures:
	$(PY) scripts/09_figures.py
	$(PY) scripts/10_animations.py

tables:
	$(PY) scripts/11_tables.py --inject README.md docs/results.md docs/legacy_audit.md

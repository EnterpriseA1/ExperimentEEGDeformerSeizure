# Validation — 2026-10-22

## Release checks

- Full pytest suite: **85 passed, 2 skipped**. The skipped checks need saved historical CBCE outputs or explicit opt-in CPU training with original checkpoint/protocol inputs.
- All 13 public command-line entry points returned successfully with `--help` after relocation.
- Analysis modules and historical L2-SP evaluator modules import successfully as packages.
- All five thesis architectures pass finite CPU forward and strict state reload checks for both 16 and 20 channels.
- A real synthetic EEGNet Lightning checkpoint passes the public prediction-to-cache-to-metric-report path, preserving the dev threshold, labels and recording seams.
- Existing tests cover loss/weight calibration, initialization, freezing, L2-SP synthetic CPU training/resume, samplers, metrics and LOSO protocols.
- All Python source parses; no machine-specific home/NAS paths or credentials were found in published source/docs.

## Fixes made during release preparation

- Relocated training code into `training/`, updated imports, subprocess commands, root resolution and code-hash references.
- Removed model-import side effects that overwrote `CUDA_VISIBLE_DEVICES`/`CUDA_DEVICE_ORDER`; the caller now controls device selection. Model forward computations were not changed.
- Prediction output accepts a filename in the current directory as well as a nested output path.
- TUSZ training and prediction use the same project-checkpoint loading compatibility setting as the existing LOSO runners for PyTorch 2.6+.
- Raw dataset locations are explicit CLI inputs rather than private machine defaults.
- Runtime dependency pins reflect the tested environment; the legacy environment listing is retained separately.

## Target-cohort equivalence

The earlier role-free preprocessing audit is recorded in [cohort_equivalence_audit.json](cohort_equivalence_audit.json). It compares fresh raw-record preprocessing with original saved LOSO pools: Siena PN00-4 has 525 windows/19 positives; CHB-MIT chb02_16 has 239 windows/21 positives. Common16 float16 signals, labels, start samples and recording seams match exactly.

All original case order and folds match the saved reference protocol (Siena14 cases/folds, CHB24 cases/23 patient folds). Synthetic tests compare old split-based inputs and the all-patient cohort through inventory/pool construction for every case, preserving byte-identical arrays and sample order for every fold role. See [preprocessing details](../prepare_dataset/README.md).

## Limits

Validation used Python 3.10 and the existing environment's runtime libraries; pytest was added in a separate temporary environment. The complete dependency set was not freshly installed in an empty environment. PyTorch was 2.7.1+cu128 / torchvision 0.22.1+cu128, with CPU execution for these tests.

No full EEG experiment or full raw-data preprocessing was rerun. These checks do not establish absence of every possible bug, clinical validity, or bit-identical retraining across hardware/software. Historical checkpoints, protocols, cohorts and result caches are external prerequisites. Renamed code changes provenance hashes and must not bypass existing frozen-run checks.

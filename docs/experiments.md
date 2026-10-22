# Reported experiment inputs and outputs

This repository covers the six result families confirmed for the thesis. It excludes old fixed-split patient reports, ensemble/polarity pilots, TUEV robustness and branch-influence diagnostics.

| Family | Code | Required inputs | Outputs |
|---|---|---|---|
| TUSZ ten-seed model comparison | `training/train_tusz.py`, `evaluation/predict.py`, `evaluation/compute_metrics.py` | Official processed train/dev/eval, model configuration and seed | Checkpoints, caches, window metrics per seed |
| Scratch vs transfer LOSO | `training/loso.py` | Complete CHB-MIT/Siena cohorts, TUSZ common16 source checkpoint | `save_loso/common16_legacy_f16_seed2024/` protocol, pools, fold results |
| Baseline transfer LOSO | `training/baseline_transfer.py` | TUSZ common16 data plus the reference LOSO folds/pools | Source checkpoints and target results under `save_loso/baselines_transfer/` |
| Target loss comparison | `training/cbce.py` | Frozen Focal sources, reference folds/pools; CBCE weights use train counts only | `save_loso/cbce_target_ratio2_seed2024/` |
| L2-SP lambda controls | `training/l2sp.py`, `evaluation/evaluate_l2sp_checkpoints.py`, `evaluation/evaluate_l2sp_lambda_sweep.py` | Reference source/protocol/pool; historical evaluators also require specific dev-result records | New unified-run namespace or historical evaluation report directories |
| Surgical early | `training/surgical.py --config early` | Reference Focal source/protocol/pool | `save_loso/surgical/early_<settings-hash>/` |

## Protocol details

- TUSZ: seeds 2023–2032; report window metrics per seed and summarize across seeds.
- Target LOSO: Siena 14 patients, CHB-MIT 23 patients from 24 cases; chb01/chb21 are one patient. One test patient and two cyclic inner-dev patients per fold.
- All target roles use four-second non-overlapping common16 windows at 200 Hz. Preserve `legacy_segment_f16` normalization for these historical comparisons.
- Target budgets in the reference runner: CHB-MIT 5 epochs, Siena 80 epochs, seed 2024. The shared runner freezes these in its protocol.
- Epoch selection uses dev AUPRC; threshold selection uses dev seizure-F1. Evaluation uses the stored threshold.

## Paths and provenance

The source checkpoint path in `training/loso.py` points to the historical TUSZ common16 pretraining run. It must exist before freezing a new protocol. Baseline and CBCE runners likewise reference their historical source layouts. These paths make the original result lineage explicit; they are not bundled weights.

The cohort inventory prefers `data_eeg_<dataset>_seizure_all/` and falls back to legacy train/dev/eval folders only when the all-patient cohort is absent. Switching layouts changes provenance. Do not overwrite an old protocol or bypass resume checks to reuse old results with a new layout. Training-code relocation also changes code hashes; use fresh run identities for new experiments.

The historical L2-SP evaluators and analysis scripts use specific saved result directories. Adapt input references only with verified fold/cohort/checkpoint provenance. Statistics are exploratory/post-hoc; multiple-comparison groups and pairing units are described in `analysis/README.md`.

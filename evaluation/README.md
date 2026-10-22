# Evaluation used by the paper/thesis

| File | Experiment | Input → output |
|---|---|---|
| `predict.py` | TUSZ benchmark, including Deformer/A4/baselines and reported seeds | Explicit checkpoint + processed eval split → probability cache with labels, IDs, recording seams and dev thresholds |
| `compute_metrics.py` | TUSZ window metrics and aggregation across seeds | Prediction caches → JSON/CSV/text summaries; ACC/AUROC/AP/seizure-F1 and related metrics |
| `evaluate_l2sp_checkpoints.py` | Historical L2-SP + CBCE, lambda=0.001, CHB-MIT/Siena LOSO | Frozen dev-result records + checkpoints + protocol → per-fold predictions/results under `reports/l2sp_test_20260929/` |
| `evaluate_l2sp_lambda_sweep.py` | CHB-MIT L2-SP lambda controls | Frozen control=0, weak=0.0001, strong=0.01 runs → `reports/l2sp_lambda_test_20261006/`; reads the previous evaluator's original=0.001 results |
| `export_recording_boundaries.py` | Support for recording-aware false-alarm metrics | Processed recording metadata → boundaries cache |
| `recover_tusz_boundaries.py` | Support for legacy TUSZ processed data | Original EDF headers + labels → boundaries cache; verify original recording order |

The last two files are support utilities, not additional thesis experiments. Boundaries are unnecessary for window ACC/AUROC/AP/F1 alone, but the retained scorer also computes false-alarm episodes. Its auxiliary patient-level outputs are not the thesis's TUSZ aggregation axis.

## LOSO evaluation inside runners

- Scratch/transfer Focal and surgical: shared worker in `scripts/run_loso_siena_chbmit.py`.
- Baseline transfer: target worker in `scripts/run_baseline_transfer_loso.py`.
- CBCE: target worker in `scripts/run_cbce_target_loso.py`.
- Unified L2-SP: `scripts/run_l2sp_loso.py` defaults to dev-only; `--evaluate-test` opts into shared-worker test evaluation.

LOSO workers write `test_predictions.npz` and `result.json` per fold. Their cache schema differs from the multi-model fixed-split cache accepted by `compute_metrics.py`; do not pass a raw LOSO cache to that CLI. The LOSO scorer imports the false-alarm helper from `evaluation.compute_metrics`.

The two L2-SP evaluators still encode historical run paths/protocols. They document those reported cohorts; they do not automatically discover new runs.

## TUSZ example

Replace placeholders with the dev-selected checkpoint and verified boundary file. These commands were not executed during packaging.

```bash
python evaluation/predict.py \
  --dataset TUSZ --split eval \
  --checkpoints 'Deformer_seed2024=/path/to/selected.ckpt' \
  --boundaries /path/to/boundaries_eval.npz \
  --out save/example/eval.npz

python evaluation/compute_metrics.py save/example/eval.npz \
  --out-prefix save/example/canonical \
  --report-title 'TUSZ window metrics'
```

Pass matching seed caches to summarize across seeds. Specify model/seed IDs. Historical defaults include an `A4-E20` title and aggregate filenames containing `3seed`; those names do not determine the actual input count. Statistical scripts expect the output layout documented in `analysis/README.md`.

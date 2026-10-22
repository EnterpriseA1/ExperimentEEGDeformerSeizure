# Statistical analyses supporting the reported experiments

These scripts read saved evaluation outputs; they do not train models or perform checkpoint inference. Run from the repository root, e.g. `python analysis/compare_tusz_models_across_seeds.py`.

| File | Comparisons | Historical inputs and outputs |
|---|---|---|
| `compare_experiment_auprc.py` | TUSZ architectures; CHB-MIT/Siena scratch vs transfer and target loss; CHB-MIT L2-SP controls and surgical early | TUSZ caches and LOSO predictions/protocols → `reports/statistical_tests_20261009/` |
| `adjust_pvalues_by_experiment.py` | Same AUPRC comparisons, with Holm families grouped by experiment | `reports/statistical_tests_20261009/results.json` → its `by_experiment/` subfolder |
| `compare_tusz_models_across_seeds.py` | Deformer vs A4/EEGNet/Conformer/TSception/EEGViT, seeds 2023–2032, ACC/AUROC/AP/seizure-F1 | `save_cbce/tusz_e10_seed2032_beta99999_20261007/{model}/ten_seed_summary.json` → `reports/tusz_seed_tests_20261017/` |

The seed2032 input directory also holds accumulated ten-seed summaries; this is not a single-seed analysis.

Preserve pairing IDs, cohorts, provenance and metrics. Analyses are exploratory/post-hoc. Grouping Holm by experiment was chosen after viewing original results; it controls multiplicity within each family, not across all thesis claims. Seed pairing and LOSO patient pairing describe different variation. Overlapping LOSO training sets limit independence assumptions.

Historical paths are retained for traceability. Adapt input references and verify the same pairing/audit checks when using another layout. Result files are not bundled.

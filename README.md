# Experiment EEG-Deformer Seizure

Training, evaluation and statistical analysis for six EEG seizure-detection experiment families reported in the paper/thesis. The models build on [EEG-Deformer](https://github.com/yi-ding-cs/EEG-Deformer) by Yi Ding and colleagues. The upstream license is preserved in [LICENSE](LICENSE).

## Repository structure

| Folder | Purpose |
|---|---|
| `prepare_dataset/` | TUSZ preprocessing and role-free CHB-MIT/Siena cohorts |
| `models/` | Deformer, EEGNet, TSception, Conformer, EEGViT, losses and L2-SP |
| `training/` | Fixed-split TUSZ and patient-disjoint LOSO experiment runners |
| `evaluation/` | Checkpoint predictions, window metrics and historical L2-SP evaluation |
| `analysis/` | Optional statistical comparisons supporting reported tables |
| `tests/` | Unit, synthetic integration and data-contract checks |
| `docs/` | Experiment map, environment and validation notes |
| `utils.py` | Shared data loading, normalization, sampling and provenance |

## Install and test

Use Python 3.10 on Linux. The runners use POSIX file locks. Create an isolated environment and choose a PyTorch CPU/CUDA build appropriate for your hardware.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`requirements.txt` pins direct runtime dependencies matching the validation environment. It is not a lockfile of every transitive package. The old environment list is retained only for reference in `docs/requirements-original.txt`. See [validation notes](docs/validation.md) for test coverage and limitations.

## Reported experiments

| Experiment | Main entry point |
|---|---|
| TUSZ architecture comparison, ten seeds | `training/train_tusz.py` |
| CHB-MIT/Siena scratch versus transfer LOSO | `training/loso.py` |
| Baseline transfer LOSO | `training/baseline_transfer.py` |
| Target Focal versus CBCE | `training/loso.py`, `training/cbce.py` |
| L2-SP and lambda controls | `training/l2sp.py` |
| Surgical early-layer fine-tuning | `training/surgical.py` |

TUSZ uses **window-level metrics per seed**, summarized across seeds on the same held-out eval cohort. CHB-MIT/Siena use **window metrics per held-out patient**, summarized across LOSO folds. A4 is a tuned Deformer configuration, not transfer fine-tuning.

Run commands from the repository root. Entry points support `--help`:

```bash
python training/train_tusz.py --help
python training/loso.py --help
python training/l2sp.py --help
python evaluation/predict.py --help
```

## Prepare target cohorts

```bash
python prepare_dataset/Siena.py --data-path /path/to/siena
python prepare_dataset/CHBMIT.py --data-path /path/to/chbmit
```

These commands create non-overlapping, all-patient cohorts; no target fixed-split JSON files are needed. LOSO assigns train/dev/test roles. The original case order is preserved to retain minibatch ordering. CHB-MIT still requires the existing verified-download marker. See [preprocessing documentation](prepare_dataset/README.md).

## Evaluate and analyze

```bash
python evaluation/predict.py --dataset TUSZ --split eval \
  --checkpoints 'Deformer_seed2024=/path/to/selected.ckpt' \
  --boundaries /path/to/boundaries_eval.npz --out save/example/eval.npz
python evaluation/compute_metrics.py save/example/eval.npz \
  --out-prefix save/example/metrics --report-title 'TUSZ window metrics'
```

Choose checkpoints by dev AUPRC and keep the checkpoint's dev-selected threshold for test. Recording boundaries support false-alarm episode metrics. Most LOSO runners evaluate inside their worker; historical L2-SP has separate evaluators. Statistical analysis is optional for model use. See [evaluation](evaluation/README.md), [analysis](analysis/README.md) and [experiment inputs](docs/experiments.md).

## Reproduction inputs

Raw EEG, processed arrays, source checkpoints, frozen protocols and historical predictions are not included. Some runners and statistical scripts intentionally retain paths to the original run layout; consult the experiment map and provide the matching inputs before execution. A source snapshot alone cannot regenerate the published tables without those inputs. Historical frozen-code replay also requires the corresponding code/environment, not just renamed files.

Only load project checkpoints from a trusted source: Lightning checkpoints include serialized configuration objects. No full EEG experiment was retrained during this repository preparation. Local tests and sampled data equivalence do not establish that retraining is bit-identical across GPU/software environments.

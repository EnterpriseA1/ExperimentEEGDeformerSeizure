# L2-SP LOSO runner

Run from the repository root after preparing the reference LOSO protocol, pool and source checkpoint described in [experiments.md](experiments.md).

```bash
python training/l2sp.py --dataset CHBMIT --all-folds --loss focal --l2sp-lambda 0.001 --run-id repeat01
python training/l2sp.py --dataset CHBMIT --all-folds --loss cbce --cbce-ratio 2 --l2sp-lambda 0.001 --run-id repeat01
```

These commands print a plan. Add `--execute` to train, and use a new run ID for a fresh repetition. The seed remains 2024; the run ID does not change it.

- Default is dev-only. `--evaluate-test` opts into test evaluation and changes experiment identity; it is not an evaluate-existing-checkpoint command.
- `--resume --execute` checks the frozen protocol/code hashes and skips completed folds. It does not silently reuse results from changed configurations.
- `--fold 0 --smoke --execute` runs the normal GPU worker with a reduced budget.
- Lambda zero provides the matched no-L2-SP control. Backbone weight decay defaults to zero and head decay to 1e-5.
- Outputs: `save_loso/l2sp_unified/<loss>/<settings-hash>/<run-id>/`.

Surgical fine-tuning has a separate entry point, `training/surgical.py`. Historical source checkpoints, reference protocols and data pools are external inputs, not included files. See [validation.md](validation.md) for which tests actually ran for this release.

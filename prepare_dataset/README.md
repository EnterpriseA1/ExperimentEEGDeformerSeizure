# Target preprocessing for thesis LOSO

CHB-MIT and Siena are prepared as one cohort, without preliminary train/dev/eval assignments:

```bash
python prepare_dataset/Siena.py --data-path /path/to/siena
python prepare_dataset/CHBMIT.py --data-path /path/to/chbmit
```

CHB-MIT retains the existing `.download_complete` integrity marker requirement. Outputs are `data_processed/data_eeg_Siena_seizure_all/` and `data_processed/data_eeg_CHBMIT_seizure_all/`. `--ROOT` selects another project/output root. Subject subsets are useful for preprocessing checks, but the LOSO runner requires the complete cohort.

All cases use 200 Hz, four-second non-overlapping windows by default. The signal filtering, resampling, channel construction and midpoint label methods are unchanged. The previous LOSO pool discarded alternate windows from the old half-overlap train sources; extracting a non-overlapping grid directly selects the same start samples within each recording.

`target_cohort.py` fixes case enumeration order, not fold membership. It deliberately preserves the old concatenation order because a different sample order can change training batches despite an identical random seed. CHB cases chb01/chb21 remain separate recording sources but are grouped into one patient by `make_folds()`. Sorted patient IDs determine held-out/test and inner-dev roles exactly as before.

The LOSO inventory prefers `_all` when present. It still reads the three old processed folders when `_all` is absent, allowing existing frozen runs to retain their old provenance. An incomplete `_all` cohort fails rather than silently falling back. Existing freeze/resume checks still reject changed source inventories: do not overwrite an old protocol or silently reuse old run results with new paths. Use a separate output checkout for fresh runs, or retain the legacy input layout when resuming the original run.

## Equivalence checks performed on 2026-10-22

- All 14 Siena and 24 CHB case identities and their order match the original saved protocol.
- All 14 Siena and 23 CHB patient-fold assignments match the original saved protocol.
- Synthetic integration tests compare old split inputs and new all-patient preprocessing through inventory and pool construction: `x`, `y`, recording seams, start samples and record IDs match byte for byte for every case; case concatenation order matches for every train/dev/test fold.
- Fresh preprocessing of one real seizure-containing recording per dataset matches the original saved LOSO pool exactly after the original common16/float16 conversion: Siena PN00-4 (525 windows, 19 positive), CHB-MIT chb02_16 (239 windows, 21 positive). Labels, start samples and seams also match.
- Signal/label extraction method ASTs match the prior exported implementation.

This establishes the tested data equivalence and preserves fold/sample ordering; it is not a claim that all raw recordings or full training were rerun. GPU nondeterminism, software versions and different checkpoints can still affect retrained scores. Existing reported results/checkpoints were not modified.

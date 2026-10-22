"""Fixed case order for reproducing thesis LOSO sample concatenation.

These are cohort inventories, not train/dev/test assignments. Keep this order:
changing it changes shuffled minibatches even when fold membership is unchanged.
CHB cases chb01/chb21 are grouped into one patient by the LOSO runner.
"""
from pathlib import Path

CASE_ORDER = {'CHBMIT': ['chb02', 'chb13', 'chb15', 'chb01', 'chb21', 'chb20', 'chb05', 'chb03', 'chb04', 'chb17', 'chb22', 'chb23', 'chb24', 'chb12', 'chb18', 'chb11', 'chb08', 'chb09', 'chb14', 'chb07', 'chb10', 'chb19', 'chb06', 'chb16'], 'Siena': ['PN00', 'PN01', 'PN03', 'PN06', 'PN07', 'PN10', 'PN11', 'PN12', 'PN14', 'PN16', 'PN05', 'PN13', 'PN09', 'PN17']}


def cohort_cases(dataset, raw_root):
    expected = CASE_ORDER[dataset]
    pattern = 'PN*' if dataset == 'Siena' else 'chb[0-9][0-9]'
    found = {p.name for p in Path(raw_root).glob(pattern) if p.is_dir()}
    if found != set(expected):
        raise ValueError(f'{dataset} cohort mismatch: missing={sorted(set(expected)-found)}, extra={sorted(found-set(expected))}')
    return list(expected)

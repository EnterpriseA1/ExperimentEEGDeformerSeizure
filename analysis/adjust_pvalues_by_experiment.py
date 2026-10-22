"""User-requested post-hoc Holm families; preserve original two-family results."""
import csv
import hashlib
import json
from pathlib import Path
try:
    from .compare_experiment_auprc import holm
except ImportError:  # Direct script execution.
    from compare_experiment_auprc import holm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/statistical_tests_20261009/results.json'
OUT = ROOT/'reports/statistical_tests_20261009/by_experiment'
GROUPS = {
 'TUSZ model comparison': ['TUSZ_EEGNet','TUSZ_Conformer','TUSZ_TSception','TUSZ_EEGViT'],
 'Scratch vs transfer': ['CHBMIT_transfer_Focal','CHBMIT_transfer_CBCE','Siena_transfer_Focal','Siena_transfer_CBCE'],
 'Target loss': ['CHBMIT_loss','Siena_loss'],
 'L2-SP': ['CHBMIT_lambda_weak','CHBMIT_lambda_original','CHBMIT_lambda_strong'],
 'Surgical Early': ['CHBMIT_surgical'],
}

def main():
    rows = json.loads(SOURCE.read_text())
    by_id = {r['id']: r for r in rows}
    ids = [i for group in GROUPS.values() for i in group]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError('Every original comparison must belong to exactly one group')
    OUT.mkdir(exist_ok=False)
    plan = dict(groups=GROUPS, correction='Holm within each experiment group',
                status='Post-hoc regrouping requested after viewing original results; not preregistered',
                limitation='Controls family-wise error within each group, not across all target claims together',
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                original_results='Parent directory retains original TUSZ4/TARGET10 analysis unchanged')
    (OUT/'regrouping_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=[]
    lines=['# User-requested grouping by experiment','',
           'Post-hoc regrouping. Raw p-values, deltas and pointwise95% CIs are unchanged. Original family-adjusted p-values are preserved in the CSV. This grouping does not control a single study-wide family-wise error rate.','']
    for group, keys in GROUPS.items():
        adjusted=holm([by_id[k]['p'] for k in keys])
        lines += ['## '+group,'','| Dataset | A -> B | Delta AUPRC | raw p | Holm within group |','|---|---|---:|---:|---:|']
        for key, p in zip(keys, adjusted):
            r = dict(by_id[key], correction_group=group,group_size=len(keys),p_holm_by_experiment=float(p),significant_by_experiment=bool(p<.05))
            result.append(r)
            lines.append(f"| {r['dataset']} | {r['a_name']} -> {r['b_name']} | {r['delta']:+.5f} | {r['p']:.5f} | {p:.5f} |")
        lines.append('')
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()

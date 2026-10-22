"""Audited AUPRC comparisons; reads saved predictions only, never trains a model.

Run: venvs/PL_5090/bin/python EEG_deformer_learn/analysis/compare_experiment_auprc.py
Positive delta always means B minus A. Tests are exploratory, not preregistered.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import stats
from sklearn.metrics import average_precision_score
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/statistical_tests_20261009'
BASE = ROOT / 'save_loso/common16_legacy_f16_seed2024'
CB = ROOT / 'save_loso/cbce_target_ratio2_seed2024'
SURG = ROOT / 'save_loso/surgical/early_579b438cdb6349db'
LAM = ROOT / 'reports/l2sp_lambda_test_20261006'
ORIG = ROOT / 'reports/l2sp_test_20260929'
TUSZ = ROOT / 'save_cbce/tusz_e10_seed2032_beta99999_20261007'
INPUTS = {}
AUDIT = []


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tracked(path):
    """เก็บ hash ไฟล์ต้นทาง เพื่อย้อนตรวจว่าผลมาจากไฟล์ใด"""
    path = Path(path)
    if str(path) not in INPUTS:
        INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return json.loads(tracked(path).read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def holm(pvalues):
    """Holm step-down: sort, multiply by remaining tests, cumulative maximum."""
    p = np.asarray(pvalues, dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid p values')
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def fingerprint(labels, seams):
    h = hashlib.sha256()
    h.update(np.asarray(labels, dtype=np.int8).tobytes())
    h.update(np.asarray(seams, dtype=np.bool_).tobytes())
    return h.hexdigest()


def checked_cache(path, reported, tusz=False):
    with np.load(tracked(path), allow_pickle=False) as z:
        y = z['labels']; p = z['probs'][0] if tusz else z['probs']; seam = z['seam']
        require(y.shape == p.shape == seam.shape and y.ndim == 1, f'Bad shapes: {path}')
        require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), f'Bad probabilities: {path}')
        require(set(np.unique(y)) == {0, 1}, f'Both classes required: {path}')
        ap = float(average_precision_score(y, p))
        require(abs(ap - reported) < 1e-9, f'AUPRC mismatch: {path}: {ap} vs {reported}')
        signature = fingerprint(y, seam)
        if tusz:
            signature += hashlib.sha256(z['subject_ids'].tobytes()).hexdigest()
        return signature, int(len(y)), int(y.sum())


def load_target(directory, pattern, dataset, folds, arm, gamma, lam=None, surgical=False):
    """อ่าน test รายผู้ป่วย; ปฏิเสธ smoke, missing/duplicate folds และ dev scores"""
    files = sorted(directory.glob(pattern))
    expected = {f['test'][0]: f['id'] for f in folds}
    require(len(files) == len(expected), f'Incomplete cohort: {directory}: {len(files)}')
    rows = {}
    for p in files:
        x = read(p); patient = x['patient']
        require(patient not in rows and patient in expected, f'Duplicate/unknown patient: {p}')
        require(x['fold'] == expected[patient] and x['dataset'] == dataset and x['seed'] == 2024, f'Identity mismatch: {p}')
        require(x.get('smoke', False) is False and x.get('test_evaluated', True) is True, f'Not held-out result: {p}')
        require(x['arm'] == arm, f'Wrong arm: {p}')
        value = float(x['auprc'])
        sig, n, pos = checked_cache(p.parent/'test_predictions.npz', value)
        require(n == x['n_windows'] and pos == x['positive_windows'], f'Test counts differ: {p}')
        config_path = p.parent/'config.json'
        if not config_path.exists():
            config_path = Path(x['checkpoint']).parent.parent/'config.json'
        config = read(config_path)
        for k, v in dict(random_seed=2024, max_epoch=5 if dataset=='CHBMIT' else 80,
                         lr=3e-4, batch_size=128, weight_decay=1e-5, num_chan=16,
                         num_time=800, norm_mode='legacy_segment_f16', focal_gamma=gamma,
                         use_weighted_sampler=False).items():
            require(config[k] == v, f'Config mismatch {k}: {config_path}')
        require(config['init_policy'] == ('network_strict' if arm=='transfer' else 'none'), f'Wrong initialization: {config_path}')
        if arm == 'transfer':
            require(Path(config['init_checkpoint']).name == Path(REFERENCE['source']).name and
                    Path(config['init_checkpoint']).resolve() == Path(REFERENCE['source']).resolve(), f'Source mismatch: {p}')
        if gamma == 0:
            neg, positive = config['cb_class_counts']; beta = config['cb_beta']
            ratio = (-np.expm1(neg*np.log(beta)))/(-np.expm1(positive*np.log(beta)))
            require(abs(ratio-2) < 1e-7, f'CBCE ratio differs: {p}')
        if lam is not None:
            require(x['l2sp_lambda'] == lam and config['l2sp_lambda'] == lam and
                    config['backbone_weight_decay'] == 0, f'L2SP mismatch: {p}')
        if surgical:
            require(config['surgical_train'] == 'cnn_encoder,pos_embedding,mlp_head', f'Wrong surgical variant: {p}')
        rows[patient] = dict(value=value, source=str(p), signature=sig, fold=x['fold'])
    require(set(rows) == set(expected), f'Patient set mismatch: {directory}')
    AUDIT.append(dict(cohort=str(directory), dataset=dataset, n=len(rows), ap_recomputed=True))
    return rows


def load_tusz(model):
    summary_path = TUSZ/model/'ten_seed_summary.json'
    x = read(summary_path); rows = {}
    for item in x['runs']:
        g = item['global']; match = re.fullmatch(re.escape(model) + r'_seed(\d+)/run_1', g['model_run'])
        require(match is not None, 'Unexpected model_run')
        seed = int(match[1]); require(seed not in rows, 'Duplicate seed')
        # interrupted runs are deliberately excluded; exactly one completed seed cache is required
        caches = [p for p in (ROOT/'save_cbce').glob(f'*/{model}/seed{seed}/eval.npz')
                  if 'interrupted' not in str(p) and 'beta9999_' not in str(p)]
        require(len(caches)==1, f'Ambiguous cache {model} {seed}: {caches}')
        cache=caches[0]; configs=list(cache.parent.glob('logs_TUSZ_*/run_1/hparams.yaml'))
        require(len(configs)==1,'Missing hparams')
        # BaseLoader parses the argparse tag as plain mappings; no Python object construction.
        cfg=yaml.load(tracked(configs[0]).read_text(), Loader=yaml.BaseLoader)['config']
        for k,v in dict(random_seed=seed,max_epoch=10,batch_size=128,num_chan=20,
                        focal_gamma=0,cb_beta=.99999,lr=.001,weight_decay=1e-5).items():
            require(float(cfg[k])==float(v),f'TUSZ config mismatch {model} {seed}: {k}')
        require(cfg['norm_mode']=='legacy_segment_f16','TUSZ normalization mismatch')
        sig,n,pos = checked_cache(cache,float(g['auprc']),tusz=True)
        require(n==g['n_windows'] and pos==g['tp']+g['fn'],'TUSZ population differs')
        rows[seed]=dict(value=float(g['auprc']),source=str(summary_path),cache=str(cache),signature=sig,fold='fixed_eval')
    require(set(rows)==set(range(2023,2033)),f'Incomplete seeds: {model}')
    AUDIT.append(dict(cohort=model,dataset='TUSZ',n=10,ap_recomputed=True))
    return rows


def align(a,b):
    """จับคู่ด้วย ID จริง ไม่ใช้ตำแหน่งแถวและไม่ตัดข้อมูลที่ขาดทิ้งเงียบ ๆ"""
    require(set(a)==set(b),'Pair IDs differ')
    keys=sorted(a)
    for key in keys:
        require(a[key]['signature']==b[key]['signature'] and a[key]['fold']==b[key]['fold'],f'Test population differs: {key}')
    av=np.array([a[k]['value'] for k in keys]); bv=np.array([b[k]['value'] for k in keys])
    require(np.isfinite(av).all() and np.isfinite(bv).all(),'Nonfinite pair')
    return keys,av,bv


def paired_summary(a,b):
    """t-test ของ mean(B-A); CI เป็น pointwise 95%, ไม่ใช่ simultaneous CI"""
    d=b-a; n=len(d)
    require(n>=3 and np.std(d,ddof=1)>0,'Degenerate paired sample')
    test=stats.ttest_rel(b,a,alternative='two-sided',nan_policy='raise')
    ci=test.confidence_interval(.95)
    sh=stats.shapiro(d)
    q1,q3=np.quantile(d,[.25,.75]); iqr=q3-q1
    # Fixed sensitivity analysis for ALL comparisons, never switch based on significance.
    w=stats.wilcoxon(np.round(d,12),alternative='two-sided',zero_method='wilcox',method='auto')
    return dict(n=n,mean_a=float(a.mean()),mean_b=float(b.mean()),delta=float(d.mean()),
                ci95_low=float(ci.low),ci95_high=float(ci.high),t=float(test.statistic),df=n-1,
                p=float(test.pvalue),paired_dz=float(d.mean()/d.std(ddof=1)),
                improved=int((d>0).sum()),worse=int((d<0).sum()),tied=int((d==0).sum()),
                shapiro_p=float(sh.pvalue),iqr_outliers=int(((d<q1-1.5*iqr)|(d>q3+1.5*iqr)).sum()),
                wilcoxon_p=float(w.pvalue))


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(out):
    global REFERENCE
    require(not out.exists(),f'Output already exists; use a new --out: {out}')
    out.mkdir(parents=True)
    # Freeze this analysis scope BEFORE computing p-values (still a post-hoc study).
    dump(out/'analysis_plan.json',dict(metric='AUPRC (average precision)',test='paired two-sided t',alpha=.05,
        families={'TUSZ':4,'TARGET':10},correction='Holm within each family',
        confidence_intervals='pointwise 95%, unadjusted',analysis_type='exploratory/post-hoc',
        sensitivity='Wilcoxon for all14, Welch for TUSZ4; corrected separately, not replacement tests',
        direction='B minus A; TUSZ B=Deformer, target B=transfer/CBCE/L2SP/surgical',
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    REFERENCE=read(BASE/'protocol.json'); cb=read(CB/'protocol.json'); surg=read(SURG/'protocol.json')
    require(cb['folds']==REFERENCE['folds'],'CBCE folds mismatch')
    require(cb['sources']['Deformer']['sha256']==REFERENCE['source_sha256'],'CBCE source mismatch')
    require(surg['folds']==REFERENCE['folds']['CHBMIT'] and surg['source_sha256']==REFERENCE['source_sha256'],'Surgical provenance mismatch')
    require(hashlib.sha256(Path(REFERENCE['source']).read_bytes()).hexdigest()==REFERENCE['source_sha256'],'Source checkpoint hash changed')
    datasets={m:load_tusz(m) for m in ['Deformer','EEGNet','Conformer','TSception','EEGViT']}
    comparisons=[]
    def add(cid,family,dataset,a_name,b_name,a,b,note=''):
        comparisons.append(dict(id=cid,family=family,dataset=dataset,a_name=a_name,b_name=b_name,a=a,b=b,note=note))
    for m in ['EEGNet','Conformer','TSception','EEGViT']:
        add('TUSZ_'+m,'TUSZ','TUSZ',m,'Deformer',datasets[m],datasets['Deformer'],
            'Seed-matched runs on fixed test set; same seed does not ensure common random numbers across architectures; Welch sensitivity provided.')
    for ds in ['CHBMIT','Siena']:
        folds=REFERENCE['folds'][ds]; arms={}
        for loss,gamma in [('Focal',2),('CBCE',0)]:
            for arm in ['scratch','transfer']:
                directory=BASE/ds if loss=='Focal' else CB/'Deformer'/ds/arm
                pattern=f'fold_*/{arm}/result.json' if loss=='Focal' else 'fold_*/run_1/result.json'
                arms[loss,arm]=load_target(directory,pattern,ds,folds,arm,gamma)
            add(f'{ds}_transfer_{loss}','TARGET',ds,f'{loss} scratch',f'{loss} full FT',arms[loss,'scratch'],arms[loss,'transfer'])
        add(f'{ds}_loss','TARGET',ds,'Focal full FT','CBCE full FT',arms['Focal','transfer'],arms['CBCE','transfer'])
        if ds=='CHBMIT': focal_full=arms['Focal','transfer']
    settings=[('control',0,LAM/'control/CHBMIT',ROOT/'save_loso/l2sp_unified/cbce/8d012f6a6d7f78e0/chbmit_lambda_20260930_r1/protocol.json'),
              ('weak',.0001,LAM/'weak/CHBMIT',ROOT/'save_loso/l2sp_unified/cbce/3d89ebedf9e99d9e/chbmit_lambda_20261003_r2/protocol.json'),
              ('original',.001,ORIG/'CHBMIT',ROOT/'save_loso/l2sp_cbce/7302283e50b40628/protocol.json'),
              ('strong',.01,LAM/'strong/CHBMIT',ROOT/'save_loso/l2sp_unified/cbce/3d75442cb681eb41/chbmit_lambda_20261003_r2/protocol.json')]
    lams={}; protocols={}
    for name,lam,directory,pp in settings:
        p=read(pp); protocols[name]=p
        for k,v in dict(source_sha256=REFERENCE['source_sha256'],folds=REFERENCE['folds']['CHBMIT'],
                        l2sp_lambda=lam,backbone_weight_decay=0,head_weight_decay=1e-5,
                        cbce_positive_negative_weight_ratio=2,seed=2024,epochs=5,lr=3e-4,batch_size=128).items():
            require(p[k]==v,f'L2SP protocol mismatch {name} {k}')
        lams[name]=load_target(directory,'fold_*/result.json','CHBMIT',REFERENCE['folds']['CHBMIT'],'transfer',0,lam)
    for name,lam,_,_ in settings[1:]:
        different=[k for k in protocols['control']['code_sha256'] if k in protocols[name]['code_sha256'] and
                   protocols['control']['code_sha256'][k]!=protocols[name]['code_sha256'][k]]
        note='Historical code hashes differ: '+', '.join(different)+'. Settings/folds/source matched; exact code equivalence not established.'
        add('CHBMIT_lambda_'+name,'TARGET','CHBMIT','CBCE lambda=0',f'CBCE lambda={lam:g}',lams['control'],lams[name],note)
    early=load_target(SURG/'CHBMIT','fold_*/transfer/result.json','CHBMIT',REFERENCE['folds']['CHBMIT'],'transfer',2,surgical=True)
    add('CHBMIT_surgical','TARGET','CHBMIT','Focal full FT','Focal surgical early',focal_full,early,
        'Surgical freezes parameters AND module training mode; effect is the whole surgical protocol.')
    require(len(comparisons)==14,'Unexpected comparison count')
    results=[]; pairs=[]
    for c in comparisons:
        keys,a,b=align(c['a'],c['b']); row={k:v for k,v in c.items() if k not in ['a','b']}
        row.update(paired_summary(a,b)); row['welch_p']=float(stats.ttest_ind(b,a,equal_var=False).pvalue) if c['family']=='TUSZ' else None
        results.append(row)
        for k,av,bv in zip(keys,a,b):
            pairs.append(dict(comparison=c['id'],unit='seed' if c['family']=='TUSZ' else 'patient',pair_id=k,
                a=float(av),b=float(bv),delta=float(bv-av),source_a=c['a'][k]['source'],source_b=c['b'][k]['source']))
    for fam in ['TUSZ','TARGET']:
        rows=[r for r in results if r['family']==fam]
        for field,new in [('p','p_holm'),('wilcoxon_p','wilcoxon_p_holm')]:
            for r,p in zip(rows,holm([r[field] for r in rows])):r[new]=float(p)
        for r in rows:r['significant_holm']=r['p_holm']<.05
        if fam=='TUSZ':
            for r,p in zip(rows,holm([r['welch_p'] for r in rows])):r['welch_p_holm']=float(p)
        else:
            for r in rows:r['welch_p_holm']=None
    csv_write(out/'results.csv',results);csv_write(out/'paired_values.csv',pairs)
    dump(out/'results.json',results)
    dump(out/'input_audit.json',dict(inputs_sha256=INPUTS,cohorts=AUDIT,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,source_checkpoint_sha256=REFERENCE['source_sha256']))
    make_report(out,results,comparisons)
    print((out/'results.md').read_text())


def make_report(out,results,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines=['# Paired t-tests: AUPRC — 2026-10-09','',
           'Exploratory analysis. Delta = B minus A; positive favors B. CI is pointwise 95% (not multiplicity-adjusted). Holm families: TUSZ4 and TARGET10.','',
           '| Dataset | A -> B | n | Delta | 95% CI | raw p | Holm p | Significant |',
           '|---|---|---:|---:|---|---:|---:|---|']
    for r in results:
        lines.append(f"| {r['dataset']} | {r['a_name']} -> {r['b_name']} | {r['n']} | {r['delta']:+.5f} | [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}] | {r['p']:.6g} | {r['p_holm']:.6g} | {'Yes' if r['significant_holm'] else 'No'} |")
    lines+=['','## Interpretation limits','',
        '- Failure to reject is not equivalence or evidence of no effect. No test-score-based method/lambda selection was performed here.',
        '- TUSZ tests describe training randomness on one fixed test cohort; seed matching is nominal across architectures, so Welch sensitivity is included.',
        '- Target observations are held-out patients but LOSO training sets overlap. Patient t-test p-values/intervals use an approximate independence assumption; this is not a corrected cross-validation significance guarantee.',
        '- Target uses one training seed. Patient SD is not training-seed SD.',
        '- Existing test results were seen before this analysis. It is post-hoc, not preregistered confirmation.',
        '- L2-SP protocols share source/folds/loss/WD but historical code hashes differ. No claim that all software was byte-identical or that algorithmic equivalence was proved.',
        '- Wilcoxon sensitivity tests symmetry/location, not exactly the same mean hypothesis. Shapiro diagnostics do not select which p-value to report.','',
        '## Diagnostics and sensitivity','',
        '| ID | Shapiro p | IQR outliers | Better/worse/tied | Wilcoxon Holm p | Welch Holm p (TUSZ only) |',
        '|---|---:|---:|---|---:|---:|']
    for r in results:
        welch='—' if r['welch_p_holm'] is None else f"{r['welch_p_holm']:.6g}"
        lines.append(f"| {r['id']} | {r['shapiro_p']:.5g} | {r['iqr_outliers']} | {r['improved']}/{r['worse']}/{r['tied']} | {r['wilcoxon_p_holm']:.6g} | {welch} |")
    lines+=['','## Provenance notes','']+[f"- {r['id']}: {r['note']}" for r in results if r['note']]
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(4,4,figsize=(15,12))
    for ax,c in zip(axes.flat,comparisons):
        _,a,b=align(c['a'],c['b']); stats.probplot(b-a,plot=ax);ax.set_title(c['id'],fontsize=9)
    for ax in list(axes.flat)[len(comparisons):]:ax.axis('off')
    fig.suptitle('Q-Q diagnostics of paired AUPRC differences (B - A)');fig.tight_layout()
    fig.savefig(out/'difference_qq.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,8));y=np.arange(len(results))
    d=np.array([r['delta'] for r in results]);lo=np.array([r['ci95_low'] for r in results]);hi=np.array([r['ci95_high'] for r in results])
    ax.errorbar(d*100,y,xerr=np.array([d-lo,hi-d])*100,fmt='o',capsize=3,color='#24638a')
    ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(y,[r['id'] for r in results]);ax.invert_yaxis()
    ax.set_xlabel('Mean AUPRC difference in percentage points (B - A), pointwise 95% CI')
    ax.set_title('Paired comparisons — exploratory; Holm decisions in results table')
    fig.tight_layout();fig.savefig(out/'effect_sizes.png',dpi=160);fig.savefig(out/'effect_sizes.pdf');plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    main(ap.parse_args().out)

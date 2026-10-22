"""Evaluate frozen completed L2SP+CBCE folds; never retrain or select on test."""
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
import json
import hashlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training import loso as common
OUT=ROOT/'reports/l2sp_lambda_test_20261006'

def main():
    import fcntl
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from models.model_handler import DLModel
    from utils import EEGDataset
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'eval.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ref=json.loads((common.OUT/'protocol.json').read_text())
    planfile=OUT/'plan.json'
    if planfile.exists():
        jobs=json.loads(planfile.read_text())
    else:
        jobs=[]
        for arm,key,run,lam in [
            ('control','8d012f6a6d7f78e0','chbmit_lambda_20260930_r1',0.0),
            ('weak','3d89ebedf9e99d9e','chbmit_lambda_20261003_r2',0.0001),
            ('strong','3d75442cb681eb41','chbmit_lambda_20261003_r2',0.01)]:
            ds='CHBMIT'
            base=ROOT/'save_loso/l2sp_unified/cbce'/key/run
            protocol=json.loads((base/'protocol.json').read_text())
            assert protocol['folds']==ref['folds'][ds]
            assert hashlib.sha256(common.SOURCE.read_bytes()).hexdigest()==protocol['source_sha256']
            files=sorted((base/ds).glob('fold_*/transfer/dev_result.json'))
            assert len(files)==23
            for f in files:
                d=json.loads(f.read_text())
                assert d['seed']==2024 and d['l2sp_lambda']==lam and d['target_cbce_ratio']==2
                assert json.loads((f.parent/'status.json').read_text())['status']=='completed_dev_only'
                ck=Path(d['checkpoint']);assert ck.is_file()
                jobs.append(dict(arm=arm,l2sp_lambda=lam,dataset=ds,fold=d['fold'],dev_result=str(f),checkpoint=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest()))
        common.write_json(planfile,jobs)
    for i,j in enumerate(jobs,1):
        ds=j['dataset']; dest=OUT/j['arm']/ds/j['fold'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():continue
        common.write_json(OUT/'status.json',dict(stage='evaluating',job=i,total=len(jobs),dataset=ds,arm=j['arm'],fold=j['fold']))
        ck=Path(j['checkpoint']);assert hashlib.sha256(ck.read_bytes()).hexdigest()==j['sha256']
        fold=next(f for f in ref['folds'][ds] if f['id']==j['fold'])
        assert not(set(fold['test']) & (set(fold['train'])|set(fold['dev'])))
        best=DLModel.load_from_checkpoint(str(ck),map_location='cpu',strict=True).to('cuda:0').eval()
        threshold=float(best.optimal_threshold.item());assert 0<=threshold<=1
        x,y,seams=common.arrays(ds,ref['datasets'][ds],fold['test'])
        loader=DataLoader(EEGDataset(x,y,idx_channels=None,norm_mode='legacy_segment_f16'),batch_size=32,shuffle=False,num_workers=2,pin_memory=True)
        chunks=[]
        with torch.inference_mode():
            for batch,_ in loader:
                chunks.append(torch.softmax(best(batch.to('cuda:0')),1)[:,1].cpu().numpy())
        probs=np.concatenate(chunks); labels=np.concatenate(y); seam=np.concatenate(seams)
        assert len(probs)==len(labels) and np.isfinite(probs).all()
        np.savez_compressed(dest/'test_predictions.npz',probs=probs,labels=labels,seam=seam,threshold=threshold)
        result=dict(dataset=ds,fold=fold['id'],patient=fold['test'][0],arm='transfer',model='Deformer',seed=2024,
                    checkpoint=str(ck),checkpoint_sha256=j['sha256'],threshold_from_dev=threshold,
                    l2sp_lambda=j['l2sp_lambda'],target_cbce_ratio=2,test_evaluated=True,**common.score(probs,labels,seam,threshold))
        common.write_json(dest/'result.json',result)
        print('DONE',i,len(jobs),ds,fold['id'],result['auprc'],flush=True)
        del best,loader,x,y,seams,chunks
        torch.cuda.empty_cache()
    report()
    common.write_json(OUT/'status.json',dict(stage='completed',jobs=len(jobs)))

def report():
    import statistics,csv
    rows=[]
    for arm,lam,base in [('control',0,OUT/'control/CHBMIT'),('weak',.0001,OUT/'weak/CHBMIT'),
                         ('original',.001,ROOT/'reports/l2sp_test_20260929/CHBMIT'),('strong',.01,OUT/'strong/CHBMIT')]:
        results=[json.loads(p.read_text()) for p in sorted(base.glob('fold_*/result.json'))]
        assert len(results)==23 and len({r['patient'] for r in results})==23
        summary=dict(arm=arm,l2sp_lambda=lam,n=23)
        for r in results:
            assert r['test_evaluated'] and r['l2sp_lambda']==lam and r['seed']==2024
            r['acc']=(r['tp']+r['tn'])/r['n_windows']
        for k in ['acc','auroc','auprc','f1sz']:
            v=[r[k] for r in results];assert all(x is not None for x in v)
            summary[k]=dict(mean=statistics.mean(v),sd=statistics.stdev(v))
        rows.append(summary)
    common.write_json(OUT/'summary.json',rows)
    lines=['| METHOD | ACC (%) | AUROC | AUPRC | F1 seizure |','|---|---:|---:|---:|---:|']
    for r in rows:
        name='TUSZ transfer + CB-CE (Control, λ=0)' if r['arm']=='control' else f"TUSZ transfer + L2-SP + CB-CE (λ={r['l2sp_lambda']:g})"
        cells=[]
        for k in ['acc','auroc','auprc','f1sz']:
            m,d=r[k]['mean'],r[k]['sd']
            cells.append(f'{m*100:.2f} ± {d*100:.2f}' if k=='acc' else f'{m:.4f} ± {d:.4f}')
        lines.append('| '+' | '.join([name]+cells)+' |')
    lines+=['','CHBMIT held-out TEST; mean ± sample SD across 23 patients; seed2024. Thresholds frozen from dev.']
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()

"""Evaluate frozen completed L2SP+CBCE folds; never retrain or select on test."""
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
import json
import hashlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_loso_siena_chbmit as common
OUT=ROOT/'reports/l2sp_test_20260929'

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
        for ds,key in [('Siena','93f462a84053a947'),('CHBMIT','7302283e50b40628')]:
            for f in sorted((ROOT/'save_loso/l2sp_cbce'/key/ds).glob('fold_*/transfer/dev_result.json')):
                d=json.loads(f.read_text())
                assert d['seed']==2024 and d['l2sp_lambda']==0.001 and d['target_cbce_ratio']==2
                assert json.loads((f.parent/'status.json').read_text())['status']=='completed_dev_only'
                ck=Path(d['checkpoint']);assert ck.is_file()
                jobs.append(dict(dataset=ds,fold=d['fold'],dev_result=str(f),checkpoint=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest()))
        common.write_json(planfile,jobs)
    for i,j in enumerate(jobs,1):
        ds=j['dataset']; dest=OUT/ds/j['fold'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():continue
        common.write_json(OUT/'status.json',dict(stage='evaluating',job=i,total=len(jobs),dataset=ds,fold=j['fold']))
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
                    l2sp_lambda=.001,target_cbce_ratio=2,test_evaluated=True,**common.score(probs,labels,seam,threshold))
        common.write_json(dest/'result.json',result)
        print('DONE',i,len(jobs),ds,fold['id'],result['auprc'],flush=True)
        del best,loader,x,y,seams,chunks
        torch.cuda.empty_cache()
    common.write_json(OUT/'status.json',dict(stage='completed',jobs=len(jobs)))

if __name__=='__main__':main()

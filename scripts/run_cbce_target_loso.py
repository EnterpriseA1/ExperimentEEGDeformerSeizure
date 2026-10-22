# หน้าที่: ใช้ source/folds เดิม แล้วเปลี่ยนเฉพาะ target loss เป็น CB-CE ratio2:1
# ใช้กับ: target-only CBCE LOSO

"""Target-only CBCE LOSO rerun. Frozen Focal sources and patient folds reused."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_loso_siena_chbmit as common
OUT = ROOT / 'save_loso/cbce_target_ratio2_seed2024'
BASE = ROOT / 'save_loso/baselines_transfer/common16_legacy_f16_seed2024'
MODELS = ('Deformer','EEGNet','Conformer','TSception','EEGViT')
TARGET_RATIO = 2.0

def dump(path, value):
    common.write_json(path, value)

def weight_ratio(negative, positive, beta):
    return -math.expm1(negative*math.log(beta)) / -math.expm1(positive*math.log(beta))

# หา beta จาก train counts ให้ weight ratio2:1; ไม่ optimize ด้วย dev/test
def beta_for_ratio(negative, positive, target=TARGET_RATIO):
    if not 0 < positive < negative or not 1 < target < negative/positive:
        raise ValueError('Target ratio must lie between1 and inverse prevalence ratio')
    lo, hi = 1e-12, 1-1e-12
    for _ in range(100):
        mid = (lo+hi)/2
        if weight_ratio(negative, positive, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2

def target_root(model, dataset, arm):
    assert model in MODELS and dataset in common.DATASETS
    assert arm == 'transfer' or (model == 'Deformer' and arm == 'scratch')
    return OUT / model / dataset / arm

def source_record(model):
    if model == 'Deformer':
        ck = common.SOURCE
    else:
        old = BASE/model/'source/TUSZ/pretrain_E20/selected_source.json'
        ck = Path(json.loads(old.read_text())['checkpoint'])
    manifest = json.loads((ck.parent.parent/'run_manifest.json').read_text())
    a = manifest['arguments']
    assert manifest['status'] == 'completed'
    assert a['model'] == model and a['num_chan'] == 16 and a['random_seed'] == 2024
    assert a['norm_mode'] == 'legacy_segment_f16' and a.get('focal_gamma',2) == 2 and a.get('cb_beta',0) == 0
    assert a['dataset'] == 'TUSZ' and a['max_epoch'] == 20
    return dict(model=model,checkpoint=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),selection='same frozen source as original Focal LOSO')

def select_source(model):
    frozen = json.loads((OUT/'protocol.json').read_text())['sources'][model]
    assert source_record(model) == frozen, 'Source changed'
    return frozen

# 222 target jobs: Deformer scratch/transfer และสี่ baseline transfer
def job_plan(folds):
    return [dict(model=m,dataset=d,fold_index=i,fold=folds[d][i]['id'],arm=a,epochs=common.EPOCHS[d])
            for i in range(23) for d in common.DATASETS if i<len(folds[d])
            for m in MODELS for a in (('scratch','transfer') if m=='Deformer' else ('transfer',))]

def freeze():
    reference_path=common.OUT/'protocol.json'
    ref=json.loads(reference_path.read_text())
    losses={}
    for d in common.DATASETS:
        assert ref['datasets'][d] == common.inventory(d)
        losses[d]={}
        for fold in ref['folds'][d]:
            n=p=0
            for e in ref['datasets'][d]:
                if e['patient'] not in fold['train']:
                    continue
                pool=common.OUT/d/'pool'/e['case']
                meta=json.loads((pool/'metadata.json').read_text())
                assert meta['source']==e and meta['label_audit']=='passed'
                y=np.load(pool/'y.npy',mmap_mode='r')
                assert np.all((y==0)|(y==1))
                p+=int(y.sum());n+=len(y)-int(y.sum())
            b=beta_for_ratio(n,p)
            losses[d][fold['id']]=dict(class_counts=[n,p],beta=b,positive_negative_weight_ratio=weight_ratio(n,p,b))
    code=['scripts/run_loso_siena_chbmit.py','models/model_handler.py','utils.py','Task.py']
    protocol=dict(seed=2024,scope='CBCE target-only; no TUSZ pretraining',models=list(MODELS),
                  sources={m:source_record(m) for m in MODELS},folds=ref['folds'],epochs=common.EPOCHS,
                  fold_losses=losses,loss='effective-number weighted softmax CE gamma0, train-only beta calibrated to ratio2',
                  beta_policy='same fold beta for all models and arms; predeclared ratio2; not validation-optimized',
                  normalization='legacy_segment_f16',channels='common16',lr=3e-4,batch_size=128,weight_decay=1e-5,
                  reference_protocol_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                  dependency_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in code},
                  checkpoint_selection='inner dev AUPRC',threshold='inner dev seizure-F1, frozen before test',
                  windows=ref['windows'],jobs=job_plan(ref['folds']),
                  caveats=['constant target weight ratio is equivalent to fixed relative weighted CE across folds',
                           'ratio2 is a moderate starting choice, not a proven optimum',
                           'same source checkpoints as Focal comparison; Deformer scratch remains random',
                           'source configuration dropout retained; Conformer/EEGViT use architecture defaults'])
    f=OUT/'protocol.json'
    if f.exists():
        assert json.loads(f.read_text())==protocol,'Frozen inputs/config changed'
    else:
        dump(f,protocol)
    dump(OUT/'queue_plan.json',protocol['jobs'])
    return protocol

# เปลี่ยน target เป็น CB-CE แต่ใช้ Focal source และ patient folds เดิม
def worker(model_name, dataset, fold_index, arm):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader
    from models.model_handler import DLModel
    from Task import initialize_model_from_checkpoint
    from utils import EEGDataset, seed_worker
    reference = json.loads((common.OUT / 'protocol.json').read_text())
    fold = reference['folds'][dataset][fold_index]
    dest = target_root(model_name, dataset, arm) / fold['id'] / 'run_1'
    if (dest / 'result.json').exists():
        return
    if (dest / 'status.json').exists():
        raise RuntimeError(f'Incomplete run: inspect {dest} before restarting')
    dest.mkdir(parents=True, exist_ok=True)
    dump(dest / 'status.json', {'stage': 'running', 'started': common.now(), 'model': model_name, 'fold': fold})
    try:
        source = select_source(model_name)
        pl.seed_everything(2024, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        checkpoint = torch.load(source['checkpoint'], map_location='cpu', weights_only=False)
        config = copy.deepcopy(checkpoint['hyper_parameters']['config'])
        del checkpoint
        assert config.model == model_name and config.num_chan == 16
        for key, value in dict(dataset=dataset, max_epoch=common.EPOCHS[dataset], lr=3e-4, batch_size=128,
                               weight_decay=1e-5, random_seed=2024, num_time=800, sampling_rate=200,
                               norm_mode='legacy_segment_f16', norm_clip=0.0, focal_alpha=0.0, focal_gamma=0.0,
                               cb_beta=0.0, head_warmup_epochs=0, backbone_lr_scale=1.0, mixed_precision=0,
                               train_patient_balanced=False, train_neg_pos_ratio=0.0, use_weighted_sampler=False,
                               init_policy='network_strict' if arm == 'transfer' else 'none', init_checkpoint=source['checkpoint'] if arm == 'transfer' else '', save_path=str(dest)).items():
            setattr(config, key, value)
        dump(dest / 'source_provenance.json', dict(source, weights_loaded=arm == 'transfer', purpose='network initialization' if arm == 'transfer' else 'configuration template only'))
        dump(dest / 'split.json', fold)
        loaders = {}
        counts = {}
        for role in ('train', 'dev'):
            x, y, _ = common.arrays(dataset, reference['datasets'][dataset], fold[role])
            counts[role] = {'patients': fold[role], 'windows': sum(map(len, y)), 'positive_windows': sum(int(v.sum()) for v in y)}
            assert 0 < counts[role]['positive_windows'] < counts[role]['windows']
            ds = EEGDataset(x, y, idx_channels=None, norm_mode='legacy_segment_f16')
            loaders[role] = DataLoader(ds, batch_size=128, shuffle=role == 'train', pin_memory=True,
                                       num_workers=4, persistent_workers=True, worker_init_fn=seed_worker)
        dump(dest / 'data_counts.json', counts)
        config.cb_class_counts = [counts['train']['windows'] - counts['train']['positive_windows'], counts['train']['positive_windows']]
        config.cb_beta = beta_for_ratio(*config.cb_class_counts)
        frozen_loss = json.loads((OUT / 'protocol.json').read_text())['fold_losses'][dataset][fold['id']]
        assert config.cb_class_counts == frozen_loss['class_counts']
        assert abs(config.cb_beta - frozen_loss['beta']) < 1e-12
        dump(dest / 'config.json', vars(config))
        from models.model_handler import effective_number_class_weights
        weights = effective_number_class_weights(config.cb_class_counts, config.cb_beta)
        assert abs(float(weights[1] / weights[0]) - TARGET_RATIO) < 1e-5
        dump(dest / 'loss_audit.json', dict(beta=config.cb_beta, class_counts=config.cb_class_counts, weights=weights.tolist(), ratio=float(weights[1]/weights[0]), count_source='this fold train only', gamma=0))
        net = DLModel(config)
        if arm == 'transfer':
            audit = initialize_model_from_checkpoint(net, source['checkpoint'], 'network_strict', str(dest / 'initialization_audit.json'))
            expected = sum(k.startswith('net.') for k in net.state_dict())
            assert audit['loaded_count'] == expected  # Not Deformer-specific76 tensors.
            assert not any(audit[k] for k in ('missing_network', 'skipped_shape', 'unexpected', 'intentionally_excluded_network', 'source_threshold_loaded', 'optimizer_state_loaded'))
        dump(dest / 'architecture.json', {'model': model_name, 'parameters': sum(p.numel() for p in net.parameters()),
             'dropout_modules': {name: mod.p for name, mod in net.named_modules() if isinstance(mod, torch.nn.modules.dropout._DropoutNd)}})
        callback = ModelCheckpoint(dirpath=str(dest / 'checkpoints'), monitor='val_auprc', mode='max', save_top_k=1)
        trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=common.EPOCHS[dataset], precision=32,
                             callbacks=[callback], logger=CSVLogger(str(dest), name='logs', version='run_1'),
                             enable_progress_bar=False, log_every_n_steps=50, num_sanity_val_steps=0)
        trainer.fit(net, train_dataloaders=loaders['train'], val_dataloaders=loaders['dev'])
        best = DLModel.load_from_checkpoint(callback.best_model_path, map_location='cpu').to('cuda:0').eval()
        threshold = float(best.optimal_threshold.item())
        # No test patient was loaded for checkpoint/threshold selection.
        x, y, seams = common.arrays(dataset, reference['datasets'][dataset], fold['test'])
        loader = DataLoader(EEGDataset(x, y, idx_channels=None, norm_mode='legacy_segment_f16'),
                            batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        chunks = []
        with torch.inference_mode():
            for batch, _ in loader:
                chunks.append(torch.softmax(best(batch.to('cuda:0')), 1)[:, 1].cpu().numpy())
        probs, labels, seam = np.concatenate(chunks), np.concatenate(y), np.concatenate(seams)
        assert len(probs) == len(labels) and np.isfinite(probs).all()
        np.savez_compressed(dest / 'test_predictions.npz', probs=probs, labels=labels, seam=seam, threshold=threshold)
        result = {'model': model_name, 'method': f'TUSZ_to_{dataset}' if arm == 'transfer' else 'scratch', 'dataset': dataset, 'fold': fold['id'],
                  'patient': fold['test'][0], 'arm': arm, 'seed': 2024,
                  'checkpoint': callback.best_model_path, 'source_checkpoint': source['checkpoint'] if arm == 'transfer' else None,
                  'best_dev_auprc': float(callback.best_model_score), 'threshold_from_dev': threshold,
                  **common.score(probs, labels, seam, threshold)}
        dump(dest / 'result.json', result)
        dump(dest / 'status.json', {'stage': 'completed', 'finished': common.now()})
        print('RESULT', json.dumps(result), flush=True)
    except BaseException as exc:
        dump(dest / 'status.json', {'stage': 'failed', 'finished': common.now(), 'error': repr(exc)})
        raise


def aggregate(model,dataset,arm):
    from sklearn.metrics import average_precision_score
    root=target_root(model,dataset,arm)
    files=sorted(root.glob('fold_*/run_1/result.json'))
    rows=[json.loads(f.read_text()) for f in files]
    if not rows:return
    c={k:sum(r[k] for r in rows) for k in ('tp','tn','fp','fn')}
    tp,tn,fp,fn=[c[k] for k in ('tp','tn','fp','fn')]
    ps=[];ys=[]
    for f in files:
        with np.load(f.parent/'test_predictions.npz') as z:
            ps.append(z['probs']);ys.append(z['labels'])
    summary=dict(model=model,dataset=dataset,arm=arm,seed=2024,completed_folds=len(rows),expected_folds=14 if dataset=='Siena' else 23,
                 ACC=(tp+tn)/sum(c.values()),AUPRC=float(average_precision_score(np.concatenate(ys),np.concatenate(ps))),
                 F1sz=2*tp/(2*tp+fp+fn),Precision=tp/(tp+fp) if tp+fp else None,Sensitivity=tp/(tp+fn),confusion_sum=c,
                 macro_patient={})
    for key in ('auprc','f1sz','precision','sensitivity'):
        v=[r[key] for r in rows if r[key] is not None]
        summary['macro_patient'][key]=dict(mean=float(np.mean(v)) if v else None,sample_sd=float(np.std(v,ddof=1)) if len(v)>1 else None,n=len(v))
    dump(root/'summary.json',summary)
    with (root/'per_patient_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries=[json.loads(p.read_text()) for p in OUT.glob('*/*/*/summary.json')]
    columns=['model','dataset','arm','seed','completed_folds','expected_folds','ACC','AUPRC','F1sz','Precision','Sensitivity']
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(summaries)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan-only',action='store_true');ap.add_argument('--worker',action='store_true')
    ap.add_argument('--model',choices=MODELS);ap.add_argument('--dataset',choices=common.DATASETS)
    ap.add_argument('--fold',type=int);ap.add_argument('--arm',choices=['scratch','transfer'])
    a=ap.parse_args();os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1';os.chdir(ROOT)
    if a.worker:
        worker(a.model,a.dataset,a.fold,a.arm);return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=freeze()
        if a.plan_only:
            print('Frozen',len(protocol['jobs']),'target jobs; no new source pretraining');return
        if (OUT/'ALL_COMPLETED').exists():return
        dump(OUT/'queue_status.json',dict(stage='waiting_original_loso',dependency=str(BASE/'ALL_COMPLETED'),updated=common.now()))
        while not (BASE/'ALL_COMPLETED').exists() or subprocess.run(['tmux','has-session','-t','baseline_transfer_loso'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            time.sleep(60)
        freeze()  # Reverify unchanged source files and code after the wait.
        for i,j in enumerate(protocol['jobs'],1):
            dest=target_root(j['model'],j['dataset'],j['arm'])/j['fold']/'run_1'
            if not (dest/'result.json').exists():
                dest.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,'-u',__file__,'--worker','--model',j['model'],'--dataset',j['dataset'],'--fold',str(j['fold_index']),'--arm',j['arm']]
                dump(dest/'launch_command.json',cmd)
                dump(OUT/'queue_status.json',dict(stage='training',job_number=i,total_jobs=len(protocol['jobs']),job=j,updated=common.now()))
                with (dest/'terminal.log').open('a') as f:
                    result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
                if result.returncode:
                    dump(OUT/'queue_status.json',dict(stage='failed',job=j,returncode=result.returncode,updated=common.now()))
                    raise RuntimeError(f'Failed worker: {dest}')
            aggregate(j['model'],j['dataset'],j['arm'])
        (OUT/'ALL_COMPLETED').touch()
        dump(OUT/'queue_status.json',dict(stage='completed',jobs=len(protocol['jobs']),updated=common.now()))

if __name__=='__main__':main()

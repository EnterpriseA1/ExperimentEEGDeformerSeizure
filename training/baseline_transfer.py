# หน้าที่: เทรน source ของสี่ baseline แล้ว transfer บน LOSO folds เดียวกับ Deformer
# ใช้กับ: EEGNet/Conformer/TSception/EEGViT transfer-only LOSO

"""Four common16 TUSZ sources, then patient-matched transfer-only LOSO.

Reuse frozen Deformer patient folds and canonical pools READ-ONLY. All new
outputs are namespaced by model, source/target stage, transfer method and fold.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import loso as common

OUT = ROOT / 'save_loso/baselines_transfer/common16_legacy_f16_seed2024'
MODELS = ('EEGNet', 'Conformer', 'TSception', 'EEGViT')
DATASETS = ('Siena', 'CHBMIT')


def source_dir(model):
    assert model in MODELS
    return OUT / model / 'source/TUSZ/pretrain_E20'


def target_root(model, dataset):
    assert model in MODELS and dataset in DATASETS
    return OUT / model / 'target' / dataset / f'TUSZ_to_{dataset}'


# สี่ source jobs + 148 target folds; ชุดนี้เป็น transfer-only ของ baseline
def job_plan(folds):
    jobs = [{'stage': 'source_pretrain', 'model': m, 'dataset': 'TUSZ', 'epochs': 20} for m in MODELS]
    for i in range(max(map(len, folds.values()))):
        for dataset in DATASETS:
            if i >= len(folds[dataset]):
                continue
            for model in MODELS:
                jobs.append({'stage': 'target_transfer', 'model': model, 'dataset': dataset,
                             'fold_index': i, 'fold': folds[dataset][i]['id'], 'epochs': common.EPOCHS[dataset]})
    return jobs


def dump(path, value):
    common.write_json(path, value)


def freeze():
    assert (common.OUT / 'ALL_COMPLETED').exists(), 'Reference LOSO must be complete'
    parent_path = common.OUT / 'protocol.json'
    parent = json.loads(parent_path.read_text())
    for d in DATASETS:
        assert parent['datasets'][d] == common.inventory(d), 'Reference input data changed'
        assert parent['folds'][d] == common.make_folds([e['patient'] for e in parent['datasets'][d]])
        for e in parent['datasets'][d]:
            pool = common.OUT / d / 'pool' / e['case']
            meta = json.loads((pool / 'metadata.json').read_text())
            assert meta['source'] == e and meta['step_samples'] == 800 and meta['label_audit'] == 'passed'
            x, y = np.load(pool / 'x.npy', mmap_mode='r'), np.load(pool / 'y.npy', mmap_mode='r')
            assert x.shape == (len(y), 16, 800) and x.dtype == np.float16
    tusz = []
    from utils import get_selected_channel_info
    for split in ('train', 'dev'):
        directory = ROOT / f'data_processed/data_eeg_TUSZ_seizure_{split}'
        indices, _ = get_selected_channel_info(str(directory), 'BL', 'common16')
        assert indices == list(range(8)) + list(range(12, 20))
        paths = sorted(directory.glob('sub*.pkl'))
        assert paths
        tusz.append({'split': split, 'channel_indices': indices,
                     'dataset_info_sha256': hashlib.sha256((directory / 'dataset_info.pkl').read_bytes()).hexdigest(),
                     'files': [{'path': str(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths]})
    protocol = {'schema_version': 1, 'models': list(MODELS), 'scope': 'transfer_only', 'seed': 2024,
                'reference_protocol': str(parent_path), 'reference_protocol_sha256': hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                'reference_pool_root': str(common.OUT), 'folds': parent['folds'],
                'tusz_source_inputs': tusz, 'source_epochs': 20, 'source_lr': 6e-4,
                'target_epochs': parent['epochs'], 'target_lr': 3e-4,
                'batch_size': 128, 'weight_decay': 1e-5, 'loss': 'Focal gamma2 alpha0 CB0',
                'normalization': 'legacy_segment_f16', 'training_precision': 'float32', 'channels': 'common16',
                'target_windows': parent['windows'], 'target_dev_selection': parent['dev_selection'],
                'checkpoint_selection': 'dev AUPRC, source on TUSZ dev and target on inner dev only',
                'threshold_selection': 'target inner-dev seizure F1, frozen before held-out patient inference',
                'dropout': {'EEGNet': '0.35 from config', 'TSception': '0.35 from config',
                            'Conformer': 'existing architecture defaults (includes0.5,0.3); ignores config dropout',
                            'EEGViT': 'existing architecture defaults0.1; ignores config dropout'},
                'initialization': 'fresh network_strict from SAME-MODEL TUSZ16 source every target fold',
                'execution': 'one GPU job at a time in tmux; no idle-GPU wait',
                'caveats': parent['caveats'] + ['fixed hyperparameters, not per-model LR optimization',
                                             'only transfer requested; no new scratch baseline LOSO'],
                'jobs': job_plan(parent['folds'])}
    p = OUT / 'protocol.json'
    if p.exists():
        assert json.loads(p.read_text()) == protocol, 'Frozen baseline protocol changed'
    else:
        dump(p, protocol)
    dump(OUT / 'queue_plan.json', protocol['jobs'])
    return protocol


def select_source(model):
    root = source_dir(model)
    log = root / f'logs_TUSZ_{model}/run_1'
    manifest = json.loads((log / 'run_manifest.json').read_text())
    assert manifest['status'] == 'completed'
    config = manifest['arguments']
    assert config['model'] == model and config['dataset'] == 'TUSZ'
    assert config['num_chan'] == 16 and config['channel_subset'] == 'common16'
    assert config['init_policy'] == 'none' and not config['init_checkpoint']
    assert config['norm_mode'] == 'legacy_segment_f16' and config['random_seed'] == 2024
    assert config['max_epoch'] == 20 and config['lr'] == 6e-4 and config['skip_final_eval']
    paths = list((log / 'checkpoints').glob('*.ckpt'))
    assert len(paths) == 1
    p = paths[0]
    with (log / 'result.csv').open() as f:
        result = {r['Metric']: float(r['Value']) for r in csv.DictReader(f)}
    record = {'model': model, 'checkpoint': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
              'best_tusz_dev_auprc': result['best_val'], 'selection': 'TUSZ dev only; no target data'}
    frozen = root / 'selected_source.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == record
    else:
        dump(frozen, record)
    return record


def pretrain(model):
    dest = source_dir(model)
    log = dest / f'logs_TUSZ_{model}/run_1'
    if log.exists():
        select_source(model)  # Reject incomplete pretraining instead of creating run_2.
        return
    command = [sys.executable, '-u', 'training/train_tusz.py', '--dataset', 'TUSZ', '--model', model,
               '--num-chan', '16', '--channel-subset', 'common16', '--num-time', '800', '--sampling-rate', '200',
               '--max-epoch', '20', '--batch-size', '128', '--lr', '6e-4', '--weight-decay', '1e-5',
               '--dropout', '0.35', '--random-seed', '2024', '--mixed-precision', '0', '--gpu', '0',
               '--norm-mode', 'legacy_segment_f16', '--focal-alpha', '0', '--focal-gamma', '2', '--cb-beta', '0',
               '--skip-final-eval', '--save-path', str(dest)]
    dest.mkdir(parents=True, exist_ok=True)
    dump(dest / 'launch_command.json', command)
    with (dest / 'terminal.log').open('a') as f:
        subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    select_source(model)


# เริ่ม target fold จาก TUSZ checkpoint ของโมเดลชนิดเดียวกัน
def worker(model_name, dataset, fold_index):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader
    from models.model_handler import DLModel
    from training.task import initialize_model_from_checkpoint
    from utils import EEGDataset, seed_worker
    reference = json.loads((common.OUT / 'protocol.json').read_text())
    fold = reference['folds'][dataset][fold_index]
    dest = target_root(model_name, dataset) / fold['id'] / 'run_1'
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
                               norm_mode='legacy_segment_f16', norm_clip=0.0, focal_alpha=0.0, focal_gamma=2.0,
                               cb_beta=0.0, head_warmup_epochs=0, backbone_lr_scale=1.0, mixed_precision=0,
                               train_patient_balanced=False, train_neg_pos_ratio=0.0, use_weighted_sampler=False,
                               init_policy='network_strict', init_checkpoint=source['checkpoint'], save_path=str(dest)).items():
            setattr(config, key, value)
        dump(dest / 'config.json', vars(config))
        dump(dest / 'source_provenance.json', source)
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
        net = DLModel(config)
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
        result = {'model': model_name, 'method': f'TUSZ_to_{dataset}', 'dataset': dataset, 'fold': fold['id'],
                  'patient': fold['test'][0], 'arm': 'transfer', 'seed': 2024,
                  'checkpoint': callback.best_model_path, 'source_checkpoint': source['checkpoint'],
                  'best_dev_auprc': float(callback.best_model_score), 'threshold_from_dev': threshold,
                  **common.score(probs, labels, seam, threshold)}
        dump(dest / 'result.json', result)
        dump(dest / 'status.json', {'stage': 'completed', 'finished': common.now()})
        print('RESULT', json.dumps(result), flush=True)
    except BaseException as exc:
        dump(dest / 'status.json', {'stage': 'failed', 'finished': common.now(), 'error': repr(exc)})
        raise


def aggregate(model, dataset):
    from sklearn.metrics import roc_auc_score, average_precision_score
    root = target_root(model, dataset)
    paths = sorted(root.glob('fold_*/run_1/result.json'))
    rows = [json.loads(p.read_text()) for p in paths]
    if not rows:
        return
    assert len({r['patient'] for r in rows}) == len(rows)
    assert all(r['model'] == model and r['dataset'] == dataset and r['arm'] == 'transfer' for r in rows)
    with (root / 'per_patient_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    counts = {key: sum(r[key] for r in rows) for key in ('tn', 'fp', 'fn', 'tp')}
    summary = {'model': model, 'method': f'TUSZ_to_{dataset}', 'seed': 2024,
               'completed_folds': len(rows), 'expected_folds': 14 if dataset == 'Siena' else 23,
               'confusion_sum': counts, 'macro_patient': {}}
    for metric in ('auroc', 'auprc', 'f1sz', 'sensitivity', 'specificity', 'precision', 'fa_per_24h'):
        values = [r[metric] for r in rows if r[metric] is not None]
        summary['macro_patient'][metric] = {'mean': float(np.mean(values)) if values else None,
                  'sample_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None, 'n': len(values)}
    probs, labels = [], []
    for path in paths:
        with np.load(path.parent / 'test_predictions.npz') as cache:
            probs.append(cache['probs']); labels.append(cache['labels'])
    p, y = np.concatenate(probs), np.concatenate(labels)
    tp, fp, fn, tn = (counts[k] for k in ('tp', 'fp', 'fn', 'tn'))
    summary['pooled'] = {'auroc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'auprc': float(average_precision_score(y, p)) if np.any(y == 1) else None,
        'f1sz': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        'sensitivity': tp/(tp+fn) if tp+fn else None, 'specificity': tn/(tn+fp) if tn+fp else None,
        'fa_per_24h': sum(r['false_alarm_episodes'] for r in rows)/(len(y)*4/3600)*24}
    summary['notes'] = ['macro SD describes patients, NOT seeds',
                        'pooled thresholded metrics use each fold own dev threshold',
                        'pooled AUROC/AP combine probabilities from different fold models; report macro alongside']
    dump(root / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    os.chdir(ROOT)
    if args.worker:
        worker(args.model, args.dataset, args.fold)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = freeze()
        print('BASELINE PLAN', len(protocol['jobs']), 'jobs:4 sources+148 transfer-only folds', flush=True)
        if args.plan_only:
            return
        for i, job in enumerate(protocol['jobs']):
            dump(OUT / 'queue_status.json', {'stage': job['stage'], 'job_number': i+1, 'total_jobs': len(protocol['jobs']), 'job': job, 'updated': common.now()})
            print(common.now(), 'JOB', i+1, job, flush=True)
            try:
                if job['stage'] == 'source_pretrain':
                    pretrain(job['model'])
                else:
                    dest = target_root(job['model'], job['dataset']) / job['fold'] / 'run_1'
                    if not (dest / 'result.json').exists():
                        dest.mkdir(parents=True, exist_ok=True)
                        cmd = [sys.executable, '-u', __file__, '--worker', '--model', job['model'],
                               '--dataset', job['dataset'], '--fold', str(job['fold_index'])]
                        dump(dest / 'launch_command.json', cmd)
                        with (dest / 'terminal.log').open('a') as log:
                            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
                    aggregate(job['model'], job['dataset'])
            except BaseException as exc:
                dump(OUT / 'queue_status.json', {'stage': 'failed', 'job': job, 'updated': common.now(), 'error': repr(exc)})
                raise
        dump(OUT / 'queue_status.json', {'stage': 'completed', 'jobs': len(protocol['jobs']), 'updated': common.now()})
        (OUT / 'ALL_COMPLETED').touch()


if __name__ == '__main__':
    main()

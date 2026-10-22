# หน้าที่: สร้าง patient folds และเทียบ Deformer scratch กับ TUSZ transfer
# ใช้กับ: Deformer scratch vs TUSZ transfer LOSO

"""Patient-disjoint LOSO, fixed target budget, scratch vs TUSZ initialization.

Independent workers release GPU/RAM after each run. No GPU-idle wait: explicitly
authorized shared-GPU execution. The all-patient cohort (or legacy folders for existing runs) supplies data;
all roles use a canonical non-overlapping 4-second grid within each recording.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_dataset.target_cohort import CASE_ORDER
OUT = ROOT / 'save_loso/common16_legacy_f16_seed2024'
SOURCE = ROOT / 'save_transfer/tusz16_to_chb16_e20_ft5_seed2024/S0_TUSZ_common16_scratch_E20/logs_TUSZ_Deformer/run_1/checkpoints/epoch=8-step=114777.ckpt'
DATASETS = ('Siena', 'CHBMIT')
EPOCHS = {'Siena': 80, 'CHBMIT': 5}


def effective_number_weight_ratio(negative, positive, beta):
    """Return positive/negative CB weight ratio without numerical cancellation."""
    return -math.expm1(negative * math.log(beta)) / -math.expm1(positive * math.log(beta))


def cbce_beta_for_ratio(negative, positive, target_ratio):
    """Solve beta from train-only counts for a predeclared CBCE weight ratio."""
    if not 0 < positive < negative or not 1 < target_ratio < negative / positive:
        raise ValueError('target ratio must lie between 1 and inverse prevalence ratio')
    lo, hi = 1e-12, 1 - 1e-12
    for _ in range(100):
        mid = (lo + hi) / 2
        if effective_number_weight_ratio(negative, positive, mid) < target_ratio:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


# CHB-MIT: chb01/chb21 เป็นคนเดียวกัน ต้องไม่แยกข้าม train/dev/test
def patient_key(name):
    return 'chb01_21' if name in ('chb01', 'chb21', 'chb01_21') else name


# LOSO: test1คน, dev2คนที่กำหนดแบบ cyclic, ที่เหลือเป็น train
def make_folds(patients):
    # Label-independent, frozen choice of two inner validation patients.
    patients = sorted(set(map(patient_key, patients)))
    assert len(patients) >= 4
    folds = []
    for i, test in enumerate(patients):
        dev = [patients[(i + 1) % len(patients)], patients[(i + 2) % len(patients)]]
        train = [p for p in patients if p not in [test] + dev]
        assert not set(train) & set(dev) and test not in train + dev
        folds.append({'id': f'fold_{i:02d}_{test}', 'test': [test], 'dev': dev, 'train': train})
    return folds


# เลือก grid4วินาทีไม่ overlap และเริ่มนับใหม่ทุก recording
def canonical_indices(record_index, window_index, step_samples):
    record_index = np.asarray(record_index)
    window_index = np.asarray(window_index)
    assert len(record_index) == len(window_index) and step_samples in (400, 800)
    assert len(record_index) > 0 and np.all(np.diff(record_index) >= 0)
    for r in np.unique(record_index):
        w = window_index[record_index == r]
        assert np.array_equal(w, np.arange(len(w))), 'Missing/reordered windows within recording'
    return np.flatnonzero((window_index * step_samples) % 800 == 0)


def inventory(dataset):
    from utils import get_selected_channel_info
    entries = []
    # Prefer the role-free cohort; retain old inventories for existing frozen runs.
    all_dir = ROOT / f'data_processed/data_eeg_{dataset}_seizure_all'
    splits = ('all',) if all_dir.exists() else ('train', 'dev', 'eval')
    for split in splits:
        directory = ROOT / f'data_processed/data_eeg_{dataset}_seizure_{split}'
        mp = directory / 'dataset_manifest.json'
        m = json.loads(mp.read_text())
        idx, names = get_selected_channel_info(str(directory), 'BL', 'common16')
        assert len(idx) == 16 and m['sampling_rate'] == 200 and m['segment_seconds'] == 4
        if split == 'all':
            assert m['split'] == 'all' and m['overlap'] == 0.0
        for s in m['subjects']:
            p = directory / s['subject_file']
            assert p.is_file()
            entries.append({'patient': patient_key(s['patient_id']), 'case': s.get('case_id', s['patient_id']),
                            'path': str(p), 'indices': idx, 'channels': names, 'original_split': split,
                            'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
                            'manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest()})
    assert len({e['case'] for e in entries}) == len(entries)
    if splits == ('all',):
        order = CASE_ORDER[dataset]
        assert {e['case'] for e in entries} == set(order), 'Incomplete LOSO cohort'
        entries.sort(key=lambda e: order.index(e['case']))
    assert len({e['patient'] for e in entries}) == (14 if dataset == 'Siena' else 23)
    return entries


# บันทึก folds/config/provenance และป้องกันใช้ผลเดิมกับ input ที่เปลี่ยน
def freeze_protocol():
    info = {d: inventory(d) for d in DATASETS}
    folds = {d: make_folds([e['patient'] for e in info[d]]) for d in DATASETS}
    protocol = {'version': 1, 'seed': 2024, 'source': str(SOURCE),
                'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'datasets': info, 'folds': folds, 'epochs': EPOCHS,
                'lr': {'scratch': 3e-4, 'transfer': 3e-4}, 'batch_size': 128,
                'normalization': 'legacy_segment_f16', 'channels': 'common16',
                'windows': '4s 200Hz, non-overlap for ALL train/dev/test roles',
                'loss': 'Focal gamma2 alpha0 CB0, natural sampling',
                'dev_selection': 'next two patients cyclically in sorted unique patient order; no test labels used',
                'model_selection': 'best inner dev AUPRC; dev seizure-F1 threshold; no LR search',
                'execution': 'serial workers, immediate GPU sharing without idle gate',
                'caveats': ['historical datasets previously explored; not a newly untouched cohort',
                            'CHB E5 is a matched short-budget comparison, not convergence claim',
                            'legacy float16 preprocessing retained; not unit-variance guarantee',
                            'non-overlapping training differs from historical fixed-split training']}
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Protocol/input changed; refusing silent reuse'
    else:
        write_json(path, protocol)
    return protocol


# แปลงข้อมูลเดิมเป็น pool แบบ non-overlap ที่ทุก target runner ใช้ร่วมกัน
def prepare_pool(dataset, entries):
    pool = OUT / dataset / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    for e in entries:
        dest = pool / e['case']
        done = dest / 'metadata.json'
        if done.exists():
            assert json.loads(done.read_text())['source'] == e
            continue
        dest.mkdir(exist_ok=True)
        print(now(), 'PREPARE', dataset, e['case'], flush=True)
        with open(e['path'], 'rb') as f:
            payload = pickle.load(f)
        data = payload['data']
        labels = np.asarray(payload['label']).reshape(-1)
        meta = payload['metadata']
        assert patient_key(meta['patient_id']) == e['patient']
        assert data.shape == (len(labels), len(meta.get('channels', range(20 if dataset == 'Siena' else 16))), 800)
        assert meta['sampling_rate'] == 200 and meta['segment_seconds'] == 4
        rec = np.asarray(meta['record_index'])
        win = np.asarray(meta['window_index'])
        assert len(rec) == len(labels)
        keep = canonical_indices(rec, win, int(meta['step_samples']))
        y = labels[keep].astype(np.int8)
        record_ids = rec[keep]
        start = win[keep].astype(np.int64) * int(meta['step_samples'])
        # Verify labels from stored event annotations, without changing their rule.
        expected = np.zeros(len(keep), dtype=np.int8)
        for r in np.unique(record_ids):
            mask = record_ids == r
            midpoint = (start[mask] + 400) / 200
            positive = np.zeros(mask.sum(), dtype=bool)
            for begin, end in meta['records'][int(r)]['seizure_intervals_seconds']:
                positive |= (midpoint >= begin) & (midpoint <= end)
            expected[mask] = positive
            assert np.all(np.diff(start[mask]) == 800)
        assert np.array_equal(y, expected), f'Midpoint label audit failed: {e["case"]}'
        mapped = np.lib.format.open_memmap(dest / 'x.npy', mode='w+', dtype=np.float16, shape=(len(keep), 16, 800))
        for offset in range(0, len(keep), 256):
            chunk = data[keep[offset:offset+256]][:, e['indices'], :].astype(np.float16)
            assert np.isfinite(chunk).all()
            mapped[offset:offset+len(chunk)] = chunk
        mapped.flush()
        del mapped, data, payload
        seam = np.r_[True, np.diff(record_ids) != 0]
        np.save(dest / 'y.npy', y)
        np.save(dest / 'seam.npy', seam)
        np.save(dest / 'start_sample.npy', start)
        np.save(dest / 'record_index.npy', record_ids)
        write_json(done, {'source': e, 'windows': len(y), 'positive_windows': int(y.sum()),
                          'seizure_events': sum(len(r['seizure_intervals_seconds']) for r in meta['records']),
                          'records': len(meta['records']), 'step_samples': 800, 'label_audit': 'passed'})


def arrays(dataset, entries, patients):
    selected = [e for e in entries if e['patient'] in patients]
    data, labels, seams = [], [], []
    for e in selected:
        p = OUT / dataset / 'pool' / e['case']
        data.append(np.load(p / 'x.npy', mmap_mode='r'))
        labels.append(np.load(p / 'y.npy', mmap_mode='r'))
        seams.append(np.load(p / 'seam.npy'))
    assert selected and {e['patient'] for e in selected} == set(patients)
    return data, labels, seams


def score(probs, labels, seam, threshold):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    from evaluation.compute_metrics import count_false_alarm_episodes
    preds = probs >= threshold
    pos = labels == 1
    tp, tn = int(np.sum(preds & pos)), int(np.sum(~preds & ~pos))
    fp, fn = int(np.sum(preds & ~pos)), int(np.sum(~preds & pos))
    fa, _ = count_false_alarm_episodes(preds, labels, seam)
    return {'auroc': float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else None,
            'auprc': float(average_precision_score(labels, probs)) if np.any(pos) else None,
            'f1sz': float(f1_score(labels, preds, zero_division=0)),
            'sensitivity': tp/(tp+fn) if tp+fn else None, 'specificity': tn/(tn+fp) if tn+fp else None,
            'precision': tp/(tp+fp) if tp+fp else None,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'false_alarm_episodes': fa, 'fa_per_24h': fa/(len(labels)*4/3600)*24,
            'n_windows': len(labels), 'positive_windows': int(pos.sum())}


# หนึ่ง fold/หนึ่ง arm: เลือกด้วย inner-dev ก่อนแตะ held-out test patient
def worker(dataset, fold_index, arm, smoke=False, run_root=None, l2sp_lambda=0.0,
           backbone_weight_decay=None, evaluate_test=True, target_cbce_ratio=None,
           surgical_train=None):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader
    from models.model_handler import DLModel
    from training.task import initialize_model_from_checkpoint
    from utils import EEGDataset, seed_worker
    protocol = json.loads((OUT / 'protocol.json').read_text())
    fold = protocol['folds'][dataset][fold_index]
    root = Path(run_root) if run_root is not None else OUT
    run_dir = root / dataset / fold['id'] / arm
    if smoke:
        run_dir = root / 'smoke' / dataset / fold['id'] / arm
    result_file = run_dir / ('result.json' if evaluate_test else 'dev_result.json')
    if result_file.exists():
        print('ALREADY COMPLETED', run_dir, flush=True)
        return
    if (run_dir / 'status.json').exists():
        raise RuntimeError(f'Incomplete existing trial, inspect before retry: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'status.json', {'status': 'running', 'started': now(), 'fold': fold, 'arm': arm, 'smoke': smoke})
    try:
        pl.seed_everything(2024, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        source = torch.load(SOURCE, map_location='cpu', weights_only=False)
        config = copy.deepcopy(source['hyper_parameters']['config'])
        del source
        settings = dict(dataset=dataset, model='Deformer', num_chan=16, num_time=800, num_class=2,
                        channel_subset='common16', sampling_rate=200, max_epoch=EPOCHS[dataset],
                        lr=3e-4, batch_size=128, weight_decay=1e-5, dropout=0.35, random_seed=2024,
                        norm_mode='legacy_segment_f16', norm_clip=0.0, focal_alpha=0.0, focal_gamma=2.0,
                        cb_beta=0.0, head_warmup_epochs=0, backbone_lr_scale=1.0,
                        mixed_precision=0, train_patient_balanced=False, train_neg_pos_ratio=0.0,
                        use_weighted_sampler=False, init_policy='network_strict' if arm == 'transfer' else 'none',
                        l2sp_lambda=l2sp_lambda, backbone_weight_decay=backbone_weight_decay,
                        init_checkpoint=str(SOURCE) if arm == 'transfer' else '', save_path=str(run_dir))
        for key, value in settings.items():
            setattr(config, key, value)
        if surgical_train is not None:
            # Set only when requested so every other arm keeps its config.json unchanged.
            config.surgical_train = surgical_train
        loaders = {}
        counts = {}
        # Never load test arrays for checkpoint selection.
        for role in ('train', 'dev'):
            x, y, _ = arrays(dataset, protocol['datasets'][dataset], fold[role])
            assert np.any(np.concatenate(y) == 1) and np.any(np.concatenate(y) == 0)
            counts[role] = {'patients': fold[role], 'windows': sum(map(len, y)), 'positive_windows': sum(int(v.sum()) for v in y)}
            ds = EEGDataset(x, y, idx_channels=None, norm_mode='legacy_segment_f16')
            loaders[role] = DataLoader(ds, batch_size=128, shuffle=role == 'train', pin_memory=True,
                                       num_workers=4, persistent_workers=True, worker_init_fn=seed_worker)
        write_json(run_dir / 'data_counts.json', counts)
        if target_cbce_ratio is not None:
            negative = counts['train']['windows'] - counts['train']['positive_windows']
            positive = counts['train']['positive_windows']
            config.focal_gamma = 0.0
            config.cb_class_counts = [negative, positive]
            config.cb_beta = cbce_beta_for_ratio(negative, positive, target_cbce_ratio)
            ratio = effective_number_weight_ratio(negative, positive, config.cb_beta)
            assert abs(ratio - target_cbce_ratio) < 1e-8
            write_json(run_dir / 'loss_audit.json', {
                'loss': 'CBCE', 'gamma': 0.0, 'beta': config.cb_beta,
                'class_counts': config.cb_class_counts,
                'positive_negative_weight_ratio': ratio,
                'count_source': 'this fold train only',
            })
        write_json(run_dir / 'config.json', vars(config))
        model = DLModel(config)
        if surgical_train is not None:
            # Fail loudly rather than silently training the full network as a baseline.
            requested = tuple(g.strip() for g in surgical_train.split(',') if g.strip())
            assert model.surgical_train == requested, (model.surgical_train, requested)
            assert any(not p.requires_grad for p in model.net.parameters()), 'no frozen parameters'
        if arm == 'transfer':
            audit = initialize_model_from_checkpoint(model, str(SOURCE), 'network_strict', str(run_dir / 'initialization_audit.json'))
            assert audit['loaded_count'] == 76 and not audit['missing_network'] and not audit['skipped_shape']
        callback = ModelCheckpoint(dirpath=str(run_dir / 'checkpoints'), monitor='val_auprc', mode='max', save_top_k=1)
        trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=1 if smoke else EPOCHS[dataset],
                             callbacks=[callback], logger=CSVLogger(str(run_dir), name='logs', version='run_1'),
                             precision=32, enable_progress_bar=False, log_every_n_steps=50,
                             num_sanity_val_steps=0, limit_train_batches=2 if smoke else 1.0,
                             limit_val_batches=2 if smoke else 1.0)
        trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['dev'])
        if not evaluate_test:
            write_json(result_file, {'dataset': dataset, 'fold': fold['id'], 'arm': arm,
                       'seed': 2024, 'checkpoint': callback.best_model_path,
                       'best_dev_auprc': float(callback.best_model_score),
                       'l2sp_lambda': l2sp_lambda, 'target_cbce_ratio': target_cbce_ratio,
                       'test_evaluated': False})
            write_json(run_dir / 'status.json', {'status': 'completed_dev_only', 'finished': now()})
            return
        best = DLModel.load_from_checkpoint(callback.best_model_path, map_location='cpu').to('cuda:0').eval()
        threshold = float(best.optimal_threshold.item())
        # Test is only touched after checkpoint/threshold selection has finished.
        x, y, seams = arrays(dataset, protocol['datasets'][dataset], fold['test'])
        test_ds = EEGDataset(x, y, idx_channels=None, norm_mode='legacy_segment_f16')
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        probabilities = []
        with torch.inference_mode():
            for i, (batch, _) in enumerate(test_loader):
                probabilities.append(torch.softmax(best(batch.to('cuda:0')), 1)[:, 1].cpu().numpy())
                if smoke and i == 1:
                    break
        probs = np.concatenate(probabilities)
        labels = np.concatenate(y)[:len(probs)]
        seam = np.concatenate(seams)[:len(probs)]
        assert np.isfinite(probs).all()
        if not smoke:
            assert len(probs) == sum(map(len, y))
        np.savez_compressed(run_dir / 'test_predictions.npz', probs=probs, labels=labels, seam=seam, threshold=threshold)
        result = {'dataset': dataset, 'fold': fold['id'], 'patient': fold['test'][0], 'arm': arm, 'seed': 2024,
                  'checkpoint': callback.best_model_path, 'best_dev_auprc': float(callback.best_model_score),
                  'threshold_from_dev': threshold, 'smoke': smoke, **score(probs, labels, seam, threshold)}
        write_json(result_file, result)
        write_json(run_dir / 'status.json', {'status': 'completed', 'finished': now()})
        print('RESULT', json.dumps(result), flush=True)
    except BaseException as exc:
        write_json(run_dir / 'status.json', {'status': 'failed', 'finished': now(), 'error': repr(exc)})
        raise


# แยก pooled metrics ออกจากค่าเฉลี่ยรายผู้ป่วย และรายงานจำนวน folds ที่จบ
def aggregate(dataset):
    rows = []
    for path in sorted((OUT / dataset).glob('fold_*/*/result.json')):
        row = json.loads(path.read_text())
        assert not row['smoke']
        rows.append(row)
    if not rows:
        return
    with (OUT / dataset / 'per_patient_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summaries = {}
    for arm in ('scratch', 'transfer'):
        group = [r for r in rows if r['arm'] == arm]
        assert len({r['patient'] for r in group}) == len(group)
        if not group:
            continue
        summary = {'completed_folds': len(group), 'expected_folds': 14 if dataset == 'Siena' else 23,
                   'macro_patient': {}, 'confusion_sum': {k: sum(r[k] for r in group) for k in ('tn', 'fp', 'fn', 'tp')}}
        for metric in ('auroc', 'auprc', 'f1sz', 'sensitivity', 'specificity', 'fa_per_24h'):
            v = [r[metric] for r in group if r[metric] is not None]
            summary['macro_patient'][metric] = {'mean': float(np.mean(v)) if v else None,
                'sample_sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'n': len(v)}
        c = summary['confusion_sum']
        denom = 2*c['tp'] + c['fp'] + c['fn']
        summary['pooled_f1sz_fold_specific_thresholds'] = 2*c['tp']/denom if denom else 0.0
        summary['pooled_fa_per_24h'] = sum(r['false_alarm_episodes'] for r in group)/(sum(r['n_windows'] for r in group)*4/3600)*24
        summaries[arm] = summary
    write_json(OUT / dataset / 'summary.json', summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--arm', choices=('scratch', 'transfer'), default='scratch')
    args = parser.parse_args()
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    os.chdir(ROOT)
    if args.worker:
        worker(args.dataset, args.fold, args.arm, args.smoke)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = freeze_protocol()
        jobs = [{'dataset': d, 'fold_index': i, 'fold': protocol['folds'][d][i]['id'], 'arm': arm,
                 'epochs': EPOCHS[d], 'lr': 3e-4}
                for i in range(23) for d in DATASETS if i < len(protocol['folds'][d]) for arm in ('scratch', 'transfer')]
        write_json(OUT / 'queue_plan.json', jobs)
        print('FROZEN PLAN', len(jobs), 'jobs; GPU sharing enabled; no idle wait', flush=True)
        if args.plan_only:
            return
        prepared = set()
        for index, job in enumerate(jobs):
            d = job['dataset']
            if d not in prepared:
                write_json(OUT / 'queue_status.json', {'stage': 'preparing_data', 'dataset': d, 'updated': now()})
                prepare_pool(d, protocol['datasets'][d])
                prepared.add(d)
            run_dir = OUT / d / job['fold'] / job['arm']
            if (run_dir / 'result.json').exists():
                aggregate(d)
                continue
            write_json(OUT / 'queue_status.json', {'stage': 'training', 'job_number': index+1, 'total_jobs': len(jobs), 'job': job, 'updated': now()})
            print(now(), 'START', index+1, '/', len(jobs), job, flush=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / 'terminal.log').open('a') as log:
                cmd = [sys.executable, '-u', __file__, '--worker', '--dataset', d, '--fold', str(job['fold_index']), '--arm', job['arm']]
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            if result.returncode:
                write_json(OUT / 'queue_status.json', {'stage': 'failed', 'job': job, 'returncode': result.returncode, 'log': str(run_dir/'terminal.log'), 'updated': now()})
                raise RuntimeError(f'Worker failed: {run_dir}; no automatic settings changes')
            aggregate(d)
        write_json(OUT / 'queue_status.json', {'stage': 'completed', 'total_jobs': len(jobs), 'updated': now()})
        (OUT / 'ALL_COMPLETED').touch()


if __name__ == '__main__':
    main()

# หน้าที่: กำหนดแผน surgical fine-tuning บน LOSO เดิม; เริ่มเทรนเมื่อระบุ --execute
# ใช้กับ: Surgical FT (Lee et al., ICLR 2023) เทียบกับ Focal full fine-tuning เดิม

"""Review a Deformer LOSO surgical fine-tuning plan; --execute starts training.

Reuses the frozen patient folds, source checkpoint, data pool, Focal loss,
normalization, seed and epochs of the Focal transfer arm in
save_loso/common16_legacy_f16_seed2024, so each surgical result can be compared
fold-by-fold against that existing full fine-tuning result.  Only the set of
trainable blocks differs; frozen blocks also keep BatchNorm in eval mode.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import loso as common

# Predeclared configurations. Everything not listed is frozen.
CONFIGS = {
    # Lee et al.: input-level shift -> adapt the earliest layers.
    'early': 'cnn_encoder,pos_embedding,mlp_head',
    # Contrast: adapt the transformer, keep the convolutional encoder fixed.
    'late': 'transformer,mlp_head',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=common.DATASETS)
    select = ap.add_mutually_exclusive_group(required=True)
    select.add_argument('--fold', type=int)
    select.add_argument('--all-folds', action='store_true')
    ap.add_argument('--config', required=True, choices=sorted(CONFIGS))
    ap.add_argument('--smoke', action='store_true', help='1 epoch, 2 batches; writes under smoke/')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()
    ref_path = common.OUT / 'protocol.json'
    ref = json.loads(ref_path.read_text())
    folds = ref['folds'][args.dataset]
    if args.fold is not None and not 0 <= args.fold < len(folds):
        ap.error('fold index outside frozen protocol')
    if args._worker and (not args.execute or args.fold is None):
        ap.error('internal worker requires --execute and one --fold')
    indices = list(range(len(folds))) if args.all_folds else [args.fold]
    groups = CONFIGS[args.config]
    settings = dict(dataset=args.dataset, surgical_config=args.config, surgical_train=groups,
                    source=str(common.SOURCE), source_sha256=ref['source_sha256'],
                    reference_protocol_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                    baseline='save_loso/common16_legacy_f16_seed2024 transfer arm (full fine-tuning)',
                    normalization='legacy_segment_f16', precision=32, loss='Focal gamma2 alpha0 CB0',
                    checkpoint_selection='inner dev AUPRC', threshold='inner dev seizure-F1',
                    seed=2024, epochs=common.EPOCHS[args.dataset], lr=3e-4, batch_size=128,
                    frozen_blocks='requires_grad False and module eval() (BatchNorm statistics fixed)')
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    run_root = ROOT / 'save_loso/surgical' / f'{args.config}_{key}'
    print(json.dumps(dict(settings, output=str(run_root),
                          selected_folds=[folds[i]['id'] for i in indices]), indent=2))
    if not args.execute:
        return
    # Same compatibility setting as the existing LOSO launchers, for trusted local checkpoints.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    if hashlib.sha256(common.SOURCE.read_bytes()).hexdigest() != ref['source_sha256']:
        raise RuntimeError('Source checkpoint changed')
    if args._worker:
        common.worker(args.dataset, args.fold, 'transfer', smoke=args.smoke,
                      run_root=run_root, surgical_train=groups)
        return
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = ['models/model_handler.py', 'training/task.py', 'utils.py',
                'training/loso.py', 'training/surgical.py']
        protocol = dict(settings, folds=folds,
                        code_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in code})
        path = run_root / 'protocol.json'
        if not args.smoke:
            if path.exists() and json.loads(path.read_text()) != protocol:
                raise RuntimeError('Existing experiment code/protocol changed; refusing reuse')
            common.write_json(path, protocol)
        for i in indices:
            cmd = [sys.executable, '-u', __file__, '--dataset', args.dataset, '--fold', str(i),
                   '--config', args.config, '--execute', '--_worker']
            if args.smoke:
                cmd.append('--smoke')
            with (run_root / f'fold_{i:02d}{"_smoke" if args.smoke else ""}.log').open('a') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()

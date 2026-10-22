"""Deformer L2-SP LOSO runner: Focal or CBCE; plan-only unless --execute.

Fresh runs use a new --run-id. --resume requires an identical frozen protocol.
Seed2024 and data/optimizer settings remain those of the shared LOSO worker.
Surgical fine-tuning is intentionally a separate runner. Use shell for queues.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_loso_siena_chbmit as common


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=common.DATASETS)
    selected = ap.add_mutually_exclusive_group(required=True)
    selected.add_argument('--fold', type=int)
    selected.add_argument('--all-folds', action='store_true')
    ap.add_argument('--loss', choices=('focal', 'cbce'), default='focal')
    ap.add_argument('--l2sp-lambda', type=float, required=True)
    ap.add_argument('--backbone-weight-decay', type=float, default=0.0)
    ap.add_argument('--cbce-ratio', type=float, default=None)
    ap.add_argument('--run-id', default='default', help='Use a new ID to train again without reusing old results')
    ap.add_argument('--resume', action='store_true', help='Continue this run only if its full protocol matches')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--evaluate-test', action='store_true', help='Opt in to held-out test evaluation; default dev-only')
    ap.add_argument('--smoke', action='store_true', help='One fold, one epoch, two batches; separate output identity')
    ap.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if any(not math.isfinite(v) or v < 0 for v in (args.l2sp_lambda, args.backbone_weight_decay)):
        ap.error('lambda and backbone weight decay must be finite and >=0')
    if args.loss == 'cbce':
        args.cbce_ratio = 2.0 if args.cbce_ratio is None else args.cbce_ratio
        if not math.isfinite(args.cbce_ratio) or args.cbce_ratio <= 1:
            ap.error('CBCE ratio must be finite and >1')
    elif args.cbce_ratio is not None:
        ap.error('--cbce-ratio requires --loss cbce')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.run_id):
        ap.error('--run-id must be a simple name, not a path')
    if args.smoke and args.all_folds:
        ap.error('--smoke requires one --fold')
    if args._worker and (not args.execute or args.fold is None):
        ap.error('internal worker requires --execute and one --fold')
    return args


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_plan(args):
    ref_path = common.OUT / 'protocol.json'
    ref = json.loads(ref_path.read_text())
    folds = ref['folds'][args.dataset]
    if args.fold is not None and not 0 <= args.fold < len(folds):
        raise ValueError('fold index outside frozen protocol')
    indices = list(range(len(folds))) if args.all_folds else [args.fold]
    settings = dict(
        runner_version=2, dataset=args.dataset, task_loss=args.loss,
        focal_gamma=2.0 if args.loss == 'focal' else 0.0,
        cbce_positive_negative_weight_ratio=args.cbce_ratio,
        beta_policy='train-fold effective-number calibration' if args.loss == 'cbce' else 'disabled',
        l2sp_lambda=args.l2sp_lambda, backbone_weight_decay=args.backbone_weight_decay,
        head_weight_decay=1e-5, evaluate_test=args.evaluate_test, smoke=args.smoke,
        source=str(common.SOURCE), source_sha256=ref['source_sha256'],
        reference_protocol_sha256=sha256(ref_path), normalization='legacy_segment_f16',
        precision=32, checkpoint_selection='inner dev AUPRC', seed=2024,
        epochs=1 if args.smoke else common.EPOCHS[args.dataset], lr=3e-4, batch_size=128,
        penalty='lambda/2 * SUM backbone squared distance',
        scope='backbone including biases and BN affine; excluding mlp_head and BN running statistics',
    )
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    output = ROOT / 'save_loso/l2sp_unified' / args.loss / key / args.run_id
    code = ['models/l2sp.py', 'models/model_handler.py', 'models/EEGDeformer.py',
            'Task.py', 'utils.py', 'scripts/run_loso_siena_chbmit.py', 'scripts/run_l2sp_loso.py']
    protocol = dict(settings, run_id=args.run_id, folds=folds,
                    code_sha256={f: sha256(ROOT / f) for f in code})
    return ref, indices, output, protocol


def validate_inputs(dataset, ref):
    if sha256(common.SOURCE) != ref['source_sha256']:
        raise RuntimeError('Source checkpoint changed')
    if common.inventory(dataset) != ref['datasets'][dataset]:
        raise RuntimeError('Reference dataset inventory changed')
    for entry in ref['datasets'][dataset]:
        pool = common.OUT / dataset / 'pool' / entry['case']
        metadata = json.loads((pool / 'metadata.json').read_text())
        if metadata['source'] != entry or metadata['label_audit'] != 'passed':
            raise RuntimeError('Reference pool provenance mismatch')
        for name in ('x.npy', 'y.npy', 'seam.npy'):
            if not (pool / name).is_file():
                raise FileNotFoundError(pool / name)


def verify_protocol(output, protocol):
    path = output / 'protocol.json'
    if not path.is_file() or json.loads(path.read_text()) != protocol:
        raise RuntimeError('Missing or changed protocol/code; use a NEW --run-id, never rewrite historical hashes')


def prepare_output(output, protocol, resume):
    if resume:
        verify_protocol(output, protocol)
    else:
        # mkdir(exist_ok=False) also prevents two fresh invocations sharing a run ID.
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RuntimeError('Run already exists; use a new --run-id for fresh training or --resume') from exc
        common.write_json(output / 'protocol.json', protocol)


def worker_command(args, index):
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--dataset', args.dataset,
           '--fold', str(index), '--loss', args.loss, '--l2sp-lambda', str(args.l2sp_lambda),
           '--backbone-weight-decay', str(args.backbone_weight_decay), '--run-id', args.run_id,
           '--execute', '--_worker']
    if args.loss == 'cbce':
        cmd += ['--cbce-ratio', str(args.cbce_ratio)]
    if args.evaluate_test:
        cmd.append('--evaluate-test')
    if args.smoke:
        cmd.append('--smoke')
    return cmd


def dispatch_worker(args, output):
    common.worker(args.dataset, args.fold, 'transfer', smoke=args.smoke, run_root=output,
                  l2sp_lambda=args.l2sp_lambda, backbone_weight_decay=args.backbone_weight_decay,
                  target_cbce_ratio=args.cbce_ratio, evaluate_test=args.evaluate_test)


def main(argv=None):
    args = parse_args(argv)
    ref, indices, output, protocol = build_plan(args)
    print(json.dumps(dict(protocol, output=str(output),
                         selected_folds=[ref['folds'][args.dataset][i] for i in indices]), indent=2))
    if not args.execute:
        return
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    validate_inputs(args.dataset, ref)
    if args._worker:
        verify_protocol(output, protocol)
        dispatch_worker(args, output)
        return
    prepare_output(output, protocol, args.resume)
    with (output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_protocol(output, protocol)
        for index in indices:
            with (output / f'fold_{index:02d}.log').open('a') as log:
                subprocess.run(worker_command(args, index), cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()

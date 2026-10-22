# หน้าที่: รับ config และเริ่มเทรนแบบ fixed split (TUSZ/Siena/CHB-MIT)
# ใช้กับ: Fixed split ทุก dataset: baseline, A/B/C, transfer, CBCE

# ==============================================================================
# ANTIGRAVITY NEW FILE: Main training script for TUSZ seizure detection
# ==============================================================================
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from argparse import ArgumentParser
import time
print(f"[{time.strftime('%H:%M:%S')}] เริ่มต้นสคริปต์: กำลังโหลดไลบรารีพื้นฐาน...")

import os
# Reload project-owned Lightning checkpoints on PyTorch 2.6+.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np
from utils import log2csv, ensure_path, log_command, log_run_manifest

print(f"[{time.strftime('%H:%M:%S')}] กำลังโหลด PyTorch และโมเดล (ขั้นตอนนี้อาจใช้นานถ้ารันบน NAS)...")
from Task import TUSZ_train

print(f"[{time.strftime('%H:%M:%S')}] โหลดไลบรารีเสร็จสิ้น! เข้าสู่การตั้งค่า Parameters...")
parser = ArgumentParser()

######## Data ########
parser.add_argument('--dataset', type=str, default='TUSZ')
parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
parser.add_argument('--label-type', type=str, default='seizure')
parser.add_argument('--num-chan', type=int, default=20)   # 20 channels = ACNS TCP-20 bipolar montage
parser.add_argument('--num-time', type=int, default=800)
parser.add_argument('--segment', type=int, default=4, help='segment length in seconds')
parser.add_argument('--sampling-rate', type=int, default=200)
parser.add_argument('--data-format', type=str, default='eeg')
parser.add_argument('--channel-subset', type=str, default='all', choices=['all', 'common16'],
                    help='all=ทุก channel ตาม dataset | common16=16 longitudinal bipolar channels ที่ TUSZ/CHB-MIT ใช้ร่วมกัน')
parser.add_argument('--norm-clip', type=float, default=0.0,
                    help='clip ค่าหลัง normalize เป็น +-N (0=ปิด). แนะนำ 10 เมื่อใช้ --norm-mode subject '
                         'เพื่อกัน artifact แอมพลิจูดสูงทำ gradient ระเบิด')
parser.add_argument('--norm-mode', type=str, default='legacy_segment_f16',
                    choices=['legacy_segment_f16', 'segment_f32', 'subject', 'train_channel_f32'],
                    help="legacy_segment_f16=พฤติกรรม float16 เดิม | segment_f32=per-window "
                         "float32 พร้อม legacy +1e-6 V | subject=สถิติทั้งคน | "
                         "train_channel_f32=fixed train-only per-channel z-score")
######## Training Process ########
parser.add_argument('--random-seed', type=int, default=2023)
parser.add_argument('--use-tuev-hnm', action='store_true', help='Use TUEV dataset for Hard Negative Mining')
parser.add_argument('--use-weighted-sampler', action='store_true', help='Use WeightedRandomSampler to balance seizure/non-seizure in each batch')
parser.add_argument(
    '--train-neg-pos-ratio', type=float, default=0.0,
    help='Dynamic train-only undersampling: keep every positive and rotate this many negatives per positive each epoch (0=off)',
)
parser.add_argument('--focal-alpha', type=float, default=0.0, help='positive-class weight ใน FocalLoss (0=ปิด ใช้ gamma อย่างเดียว | rebalance โดยไม่ซ้ำข้อมูล ปลอดภัยต่อ overfit กว่า sampler)')
parser.add_argument('--focal-gamma', type=float, default=2.0,
                    help='Focal focusing exponent (default 2 preserves existing runs)')
parser.add_argument('--cb-beta', type=float, default=0.0,
                    help='Cui et al. effective-number class balancing beta in (0,1); 0=off')
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--additional-epoch', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--head-warmup-epochs', type=int, default=0,
                    help='Freeze non-head network parameters for this many initial epochs')
parser.add_argument('--backbone-lr-scale', type=float, default=1.0,
                    help='Backbone LR multiplier after warm-up; head always uses --lr')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 regularization (เพิ่มเพื่อลด overfit)')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--val-rate', type=float, default=0.2)

parser.add_argument('--save-path', default='./save/')
parser.add_argument('--load-path', default='./data_processed/')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mixed-precision', type=int, default=0)
parser.add_argument('--skip-final-eval', action='store_true',
                    help='Finish after dev-selected training; do not load or score held-out eval')
parser.add_argument('--init-checkpoint', default='',
                    help='Optional Lightning checkpoint used only to initialize model weights')
parser.add_argument('--l2sp-lambda', type=float, default=0.0,
                    help='lambda/2 times SUM squared distance to source backbone; 0 disables')
parser.add_argument('--backbone-weight-decay', type=float, default=None,
                    help='Default: 0 with L2-SP; otherwise --weight-decay')
parser.add_argument('--init-policy', default='none',
                    choices=['none', 'network_strict', 'backbone_strict', 'compatible'],
                    help='network_strict=load every network tensor | '
                         'backbone_strict=load every network tensor except net.mlp_head.* | '
                         'compatible=load only name/shape-compatible tensors')
######## Model Parameters ########
parser.add_argument('--model', type=str, default='Deformer',
                    choices=['Deformer', 'EEGNet', 'TSception', 'Conformer', 'EEGViT'])
parser.add_argument('--graph-type', type=str, default='BL', choices=['TS', 'BL'])
parser.add_argument('--kernel-length', type=int, default=21)
parser.add_argument('--T', type=int, default=64)
parser.add_argument('--AT', type=int, default=32)
parser.add_argument('--num-layers', type=int, default=6)

import glob

args = parser.parse_args()

import math
if args.l2sp_lambda < 0 or not math.isfinite(args.l2sp_lambda):
    parser.error('--l2sp-lambda must be finite and >= 0')
if args.l2sp_lambda > 0 and args.init_policy not in ('network_strict', 'backbone_strict'):
    parser.error('L2-SP requires strict source initialization')
if args.backbone_weight_decay is not None and (args.backbone_weight_decay < 0 or not math.isfinite(args.backbone_weight_decay)):
    parser.error('--backbone-weight-decay must be finite and >= 0')

if bool(args.init_checkpoint) != (args.init_policy != 'none'):
    parser.error('--init-checkpoint and a non-none --init-policy must be provided together')
if args.channel_subset == 'common16' and args.num_chan != 16:
    parser.error('--channel-subset common16 requires --num-chan 16')

if args.train_neg_pos_ratio < 0:
    parser.error('--train-neg-pos-ratio must be >= 0')
enabled_samplers = sum((
    bool(args.use_weighted_sampler),
    args.train_neg_pos_ratio > 0,
))
if enabled_samplers > 1:
    parser.error('weighted and negative-ratio samplers are mutually exclusive')
if args.head_warmup_epochs < 0 or args.head_warmup_epochs >= args.max_epoch:
    parser.error('--head-warmup-epochs must be >= 0 and less than --max-epoch')
if not 0 < args.backbone_lr_scale <= 1:
    parser.error('--backbone-lr-scale must be in (0, 1]')
if args.head_warmup_epochs > 0 and args.init_policy != 'backbone_strict':
    parser.error('--head-warmup-epochs requires --init-policy backbone_strict')
if args.focal_gamma < 0:
    parser.error('--focal-gamma must be >= 0')
if args.cb_beta < 0 or args.cb_beta >= 1:
    parser.error('--cb-beta must be 0 (off) or in (0, 1)')
if args.cb_beta > 0 and args.focal_alpha > 0:
    parser.error('--cb-beta and --focal-alpha are mutually exclusive')
if args.cb_beta > 0 and enabled_samplers:
    parser.error('--cb-beta cannot be combined with a train sampler in this controlled experiment')

logs_name = 'logs_{}_{}'.format(args.dataset, args.model)
base_log_path = os.path.join(args.save_path, logs_name)

# Find existing runs to determine the next run index
if not os.path.exists(base_log_path):
    run_idx = 1
else:
    existing_runs = glob.glob(os.path.join(base_log_path, 'run_*'))
    run_numbers = []
    for run_dir in existing_runs:
        try:
            num = int(os.path.basename(run_dir).split('_')[1])
            run_numbers.append(num)
        except (IndexError, ValueError):
            pass
    run_idx = max(run_numbers) + 1 if run_numbers else 1

experiment_ID = f'run_{run_idx}'
log_path = os.path.join(base_log_path, experiment_ID)
args.norm_stats_path = (
    os.path.abspath(os.path.join(log_path, 'train_channel_f32_stats.npz'))
    if args.norm_mode == 'train_channel_f32' else ''
)
ensure_path(log_path)
log_command(log_path, args)
log_run_manifest(log_path, args, experiment_ID, status='started')

# Run split-based TUSZ training
try:
    results = TUSZ_train(
        args=args,
        experiment_ID=experiment_ID,
        logs_name=logs_name
    )
except KeyboardInterrupt as exc:
    log_run_manifest(log_path, args, experiment_ID, status='interrupted', error=repr(exc))
    raise
except BaseException as exc:
    log_run_manifest(log_path, args, experiment_ID, status='failed', error=repr(exc))
    raise
else:
    log2csv(os.path.join(log_path, 'result.csv'), results[0])
    log_run_manifest(log_path, args, experiment_ID, status='completed')
print(f"TUSZ model training completed and results logged successfully in {log_path}.")

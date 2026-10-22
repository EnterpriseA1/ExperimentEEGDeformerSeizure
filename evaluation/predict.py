# หน้าที่: infer checkpoint แล้วเก็บ probabilities/labels/threshold/seams เพื่อใช้ประเมินซ้ำ
# ใช้กับ: fixed split ทุก dataset

"""เก็บ per-window probability ของทุกโมเดลลงไฟล์เดียว (รันครั้งเดียว ใช้ได้ตลอด)

ทำไมต้องมี:
  ทุกสคริปต์ประเมินผลตอนนี้คำนวณ probability แล้วทิ้ง ต้องรัน inference ใหม่ทุกครั้ง
  งานที่จะทำต่อ (temporal smoothing, event-level scoring, per-subject calibration,
  การเลือก threshold ที่ FA เป้าหมาย) ล้วนต้องใช้ probability ชุดเดียวกันซ้ำๆ
  → เก็บครั้งเดียวแล้วทำงานที่เหลือบน CPU ได้หมด ไม่ต้องแตะ GPU อีก

สิ่งที่เก็บ:
  probs        (n_model, n_window)  ความน่าจะเป็นคลาส seizure
  labels       (n_window,)          label จริง
  subject_ids  (n_window,)          window นี้มาจาก subject ไหน
  seam         (n_window,) bool     True = window นี้เริ่ม recording ใหม่ (ห้าม smooth ข้าม)
  thresholds   (n_model,)           optimal_threshold ที่ติดมากับ checkpoint
  norm_modes   (n_model,)           preprocessing ที่ใช้ตอน eval (ต้องตรงกับตอนเทรน)

Usage:
    python evaluation/predict.py --run 3
    python evaluation/predict.py --run 3 --models "Deformer,Conformer"
    python evaluation/predict.py --runs "Deformer=4" --norm-mode segment_f32
    python evaluation/predict.py \
        --checkpoints "A4=save_hparam/A4_dropout035/logs_TUSZ_Deformer/run_1/checkpoints/model.ckpt"
"""
import os
import os.path as osp
import sys
import glob
import re
import argparse
import warnings

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils import load_data, get_channel_info, prepare_data_for_training, canonical_norm_mode
from models.model_handler import DLModel

warnings.filterwarnings('ignore')

ALL_MODELS = [
    'Deformer', 'EEGNet', 'TSception', 'Conformer', 'EEGViT',
]


def get_checkpoint(log_dir, run_id):
    if run_id is None:
        return None, None
    if not osp.exists(log_dir):
        return None, None
    runs = sorted([d for d in os.listdir(log_dir) if d.startswith('run_')],
                  key=lambda x: int(x.split('_')[1]))
    if not runs:
        return None, None
    target = f'run_{run_id}'
    if target not in runs:
        return None, None
    ck = glob.glob(osp.join(log_dir, target, 'checkpoints', '*.ckpt'))
    return (ck[0], target) if ck else (None, None)


def main():
    # Lightning 2.0 checkpoints contain the project configuration Namespace.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='TUSZ')
    ap.add_argument('--label-type', default='seizure')
    ap.add_argument('--load-path', default='./data_processed/')
    ap.add_argument('--data-format', default='eeg')
    ap.add_argument('--graph-type', default='BL')
    ap.add_argument('--split', default='eval')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--run', type=int, default=None)
    ap.add_argument('--runs', default=None, help='ระบุราย model เช่น "Deformer=4,Conformer=3"')
    ap.add_argument(
        '--checkpoints', default=None,
        help='explicit checkpoint เช่น "A4=/path/model.ckpt"; คั่นหลายตัวด้วย comma',
    )
    ap.add_argument('--models', default=None)
    ap.add_argument('--norm-mode', default=None,
                    choices=['legacy_segment_f16', 'segment', 'segment_f32', 'subject'],
                    help='บังคับ norm_mode (default: อ่านจาก checkpoint)')
    ap.add_argument('--boundaries', default=None, help='ไฟล์จาก recover_tusz_boundaries.py')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.run is None and not args.runs and not args.checkpoints:
        ap.error('ต้องระบุ --run, --runs หรือ --checkpoints')

    if osp.basename(os.getcwd()) == 'evaluation':
        os.chdir('..')

    per_model_run = {}
    if args.runs:
        for item in args.runs.split(','):
            k, v = item.split('=')
            per_model_run[k.strip()] = int(v)

    models = [m.strip() for m in args.models.split(',')] if args.models else ALL_MODELS
    eval_path = osp.join(args.load_path,
                         f'data_{args.data_format}_{args.dataset}_{args.label_type}_{args.split}')
    subj_ids = sorted(int(re.search(r'sub(\d+)\.pkl', f).group(1))
                      for f in glob.glob(osp.join(eval_path, 'sub*.pkl')))
    idx_chan, _ = get_channel_info(load_path=eval_path, graph_type=args.graph_type)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Resolve checkpoints before the expensive dataset load so configuration or
    # checkpoint errors fail fast.  map_location below keeps CUDA checkpoints
    # evaluable in CPU-only environments.
    selected = []
    if args.checkpoints:
        for item in args.checkpoints.split(','):
            model_name, ckpt = item.split('=', 1)
            model_name = model_name.strip()
            ckpt = osp.abspath(ckpt.strip())
            if not osp.isfile(ckpt):
                raise FileNotFoundError(f'{model_name}: ไม่พบ checkpoint {ckpt}')
            run_name = osp.basename(osp.dirname(osp.dirname(ckpt)))
            selected.append((model_name, ckpt, run_name))

    if args.run is not None or args.runs:
        for model_name in models:
            run_id = per_model_run.get(model_name, args.run)
            ckpt, run_name = get_checkpoint(
                f'save/logs_TUSZ_{model_name}', run_id
            )
            if not ckpt:
                print(f'[!] ข้าม {model_name}: ไม่พบ checkpoint (run={run_id})')
                continue
            selected.append((model_name, ckpt, run_name))
    if not selected:
        raise ValueError('ไม่พบ checkpoint ที่ระบุ')

    print('=' * 74)
    print(f'  Cache per-window probabilities — split = {args.split}')
    print('=' * 74)
    print(f'[*] {len(subj_ids)} subjects, {len(selected)} models, device={device}')

    # ---- โหลด data ครั้งเดียว ----
    print('[*] Loading data...')
    subj_raw = [load_data(load_path=eval_path, load_idx=[s], concat=False)
                for s in tqdm(subj_ids, desc='    subjects')]

    labels = np.concatenate([l[0] for _, l in subj_raw])
    sub_of_win = np.concatenate([np.full(len(l[0]), s, dtype=np.int32)
                                 for s, (_, l) in zip(subj_ids, subj_raw)])

    # ---- seam: window แรกของแต่ละ recording ----
    seam = np.zeros(len(labels), dtype=bool)
    bpath = args.boundaries or f'save/boundaries_{args.split}.npz'
    if osp.exists(bpath):
        b = np.load(bpath, allow_pickle=True)
        if not bool(b['verified']):
            print(f'[!] {bpath} ยังไม่ผ่านการตรวจสอบ — seam อาจผิด')
        off = 0
        bmap = dict(zip(b['subject_ids'].tolist(), b['boundaries']))
        for s, (_, l) in zip(subj_ids, subj_raw):
            n = len(l[0])
            seam[off] = True                              # ต้นของ subject ก็คือรอยต่อ
            for cut in bmap.get(s, []):
                if 0 < cut < n:
                    seam[off + int(cut)] = True
            off += n
        print(f'[*] seam {int(seam.sum()):,} จุด จาก {len(labels):,} window')
    else:
        print(f'[!] ไม่พบ {bpath} — seam จะเป็นค่าว่าง '
              f'(รัน recover_tusz_boundaries.py ก่อนถ้าจะทำ temporal processing)')

    loader_cache = {}

    def get_loader(mode):
        if mode not in loader_cache:
            data_list = [d[0] for d, _ in subj_raw]
            label_list = [l[0] for _, l in subj_raw]
            loader_cache[mode] = prepare_data_for_training(
                data_list=data_list,
                label_list=label_list,
                idx=idx_chan,
                batch_size=args.batch_size,
                shuffle=False,
                norm_mode=mode,
            )
        return loader_cache[mode]

    # ---- inference ----
    names, probs, thrs, modes = [], [], [], []
    for m, ckpt, run_name in selected:
        model = DLModel.load_from_checkpoint(
            ckpt, map_location=device
        ).to(device).eval()
        thr = model.optimal_threshold.item() if hasattr(model, 'optimal_threshold') else 0.5
        mode = canonical_norm_mode(
            args.norm_mode or getattr(model.config, 'norm_mode', 'segment')
        )
        print(f'\n[*] {m} ({run_name})  norm={mode}  threshold={thr:.4f}')

        p = []
        with torch.no_grad():
            for x, _ in tqdm(
                get_loader(mode), desc='    windows', leave=False
            ):
                p.append(
                    torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy()
                )
        p = np.concatenate(p)
        assert len(p) == len(labels), f'{m}: ได้ {len(p)} ไม่ตรงกับ label {len(labels)}'

        names.append(f'{m}/{run_name}')
        probs.append(p.astype(np.float32))
        thrs.append(thr)
        modes.append(mode)
        del model
        torch.cuda.empty_cache()

    if not probs:
        print('[!] ไม่มีโมเดลไหนถูกประเมินเลย')
        return

    out = args.out or f'save/probs_{args.split}.npz'
    os.makedirs(osp.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        model_names=np.array(names),
        probs=np.stack(probs),
        labels=labels.astype(np.int8),
        subject_ids=sub_of_win,
        seam=seam,
        thresholds=np.array(thrs, dtype=np.float32),
        norm_modes=np.array(modes),
    )
    print(f'\nบันทึก {len(names)} โมเดล × {len(labels):,} window ที่ {out}')
    print('→ งานที่เหลือ (smoothing / event-level / calibration) ทำบน CPU ได้เลย')


if __name__ == '__main__':
    main()

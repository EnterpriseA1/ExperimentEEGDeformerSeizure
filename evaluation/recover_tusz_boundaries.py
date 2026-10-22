# หน้าที่: กู้รอยต่อ recording ของ TUSZ รุ่นเก่าจาก EDF headers
# ใช้กับ: legacy TUSZ metadata recovery

"""กู้ขอบเขต recording ของแต่ละ subject กลับมาจาก EDF header

ทำไมต้องมี:
  `prepare_dataset/TUSZ.py` ต่อทุก EDF session ของผู้ป่วยคนเดียวเป็น array เดียว
  โดย **ไม่เก็บตัวคั่น** ว่า recording ไหนจบตรงไหน (เก็บแค่ 'data' กับ 'label')
  และ `glob.glob` ที่ใช้จัดลำดับไฟล์ **ไม่ได้ sort** → ลำดับไฟล์ในผู้ป่วยคนหนึ่ง
  ไม่ใช่ลำดับเวลาจริง

  ผลคือ window ที่อยู่ติดกันใน array อาจมาจากคนละ session ที่ห่างกันเป็นวัน
  → ถ้าทำ temporal smoothing หรือรวม event ข้ามรอยต่อนี้ = ผิด

วิธี:
  replicate glob + grouping ของ create_dataset() ทุกขั้น แล้วอ่าน **เฉพาะ header**
  ของ EDF (preload=False จึงเร็ว) เพื่อคำนวณจำนวน window ต่อ record
    n_window = int(round(n_times * target_fs / sfreq)) // win_len      (overlap = 0)

  glob บน filesystem เดิมที่ไฟล์ไม่ถูกแตะ = deterministic → ลำดับที่ได้ตรงกับตอน preprocess

การตรวจสอบ:
  สคริปต์เทียบ "ผลรวมจำนวน window" กับจำนวน label จริงใน .pkl ทุก subject
  ถ้าไม่ตรงจะรายงานออกมา ไม่เงียบ

Usage:
    python evaluation/recover_tusz_boundaries.py --split eval
    python evaluation/recover_tusz_boundaries.py --split eval --out save/boundaries_eval.npz
"""
import os
import os.path as osp
import glob
import pickle
import argparse
import warnings

import numpy as np

warnings.filterwarnings('ignore')


def group_by_patient(data_path, split):
    """คืน dict {patient_id: [edf paths]} ตามลำดับเดียวกับ create_dataset()"""
    split_path = osp.join(data_path, split)
    all_edf = glob.glob(osp.join(split_path, '**', '*.edf'), recursive=True)
    all_edf = [f for f in all_edf if '_bipolar' not in f]      # เหมือน TUSZ.py:192

    records = {}
    for fpath in all_edf:
        pid = None
        for p in fpath.split(os.sep):
            if p.startswith('aaaa') and len(p) == 8:            # เหมือน TUSZ.py:202
                pid = p
                break
        if pid:
            records.setdefault(pid, []).append(fpath)
    return records


def windows_per_record(edf_path, target_fs, win_len):
    """จำนวน window ที่ record นี้จะให้ — อ่านแค่ header ไม่โหลดสัญญาณ"""
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    n_resampled = int(round(raw.n_times * target_fs / raw.info['sfreq']))
    return n_resampled // win_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', required=True)
    ap.add_argument('--split', default='eval', choices=['train', 'dev', 'eval'])
    ap.add_argument('--load-path', default='./data_processed/')
    ap.add_argument('--dataset', default='TUSZ')
    ap.add_argument('--label-type', default='seizure')
    ap.add_argument('--data-format', default='eeg')
    ap.add_argument('--segment', type=int, default=4)
    ap.add_argument('--sampling-rate', type=int, default=200)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if osp.basename(os.getcwd()) == 'evaluation':
        os.chdir('..')

    win_len = args.segment * args.sampling_rate
    pkl_dir = osp.join(args.load_path,
                       f'data_{args.data_format}_{args.dataset}_{args.label_type}_{args.split}')

    records = group_by_patient(args.data_path, args.split)
    patient_list = sorted(records.keys())                        # เหมือน TUSZ.py:211

    print('=' * 74)
    print(f'  Recover recording boundaries — split = {args.split}')
    print('=' * 74)
    print(f'[*] ผู้ป่วยที่เจอจาก EDF : {len(patient_list)}')
    print(f'[*] window length        : {win_len} samples ({args.segment}s @ {args.sampling_rate}Hz)')

    subj_ids, all_bounds, all_nwin = [], [], []
    mismatched = []

    for idx, pid in enumerate(patient_list):
        pkl_path = osp.join(pkl_dir, f'sub{idx}.pkl')
        if not osp.exists(pkl_path):
            continue

        n_label = len(pickle.load(open(pkl_path, 'rb'))['label'])

        counts = [windows_per_record(f, args.sampling_rate, win_len) for f in records[pid]]
        bounds = np.cumsum(counts)[:-1]        # ตำแหน่งรอยต่อ (ไม่รวมปลายสุด)
        total = int(np.sum(counts))

        if total != n_label:
            mismatched.append((idx, pid, total, n_label))

        subj_ids.append(idx)
        all_bounds.append(np.asarray(bounds, dtype=np.int64))
        all_nwin.append(n_label)
        print(f'  sub{idx:<4d} {pid}  records={len(counts):3d}  '
              f'window: header={total:6d} pkl={n_label:6d}  '
              f'{"OK" if total == n_label else "MISMATCH"}')

    print('\n' + '-' * 74)
    n_ok = len(subj_ids) - len(mismatched)
    print(f'ตรงกัน {n_ok}/{len(subj_ids)} subject')
    if mismatched:
        print('[!] ไม่ตรง — อย่าเพิ่งใช้ผลนี้ทำ temporal processing:')
        for idx, pid, t, n in mismatched:
            print(f'    sub{idx} ({pid}): header={t} pkl={n} ต่าง {t - n:+d}')
    else:
        total_seam = sum(len(b) for b in all_bounds)
        total_win = sum(all_nwin)
        print(f'รอยต่อทั้งหมด {total_seam:,} จุด จาก {total_win:,} window '
              f'({total_seam / total_win * 100:.2f}%)')
        print('→ smoothing/นับ event ต้องไม่ข้ามรอยต่อเหล่านี้')

    out = args.out or f'save/boundaries_{args.split}.npz'
    os.makedirs(osp.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        subject_ids=np.asarray(subj_ids),
        n_windows=np.asarray(all_nwin),
        boundaries=np.array(all_bounds, dtype=object),
        verified=len(mismatched) == 0,
    )
    print(f'\nบันทึกที่ {out}')


if __name__ == '__main__':
    main()

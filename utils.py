# หน้าที่: อ่านข้อมูล, normalize, sampling และบันทึก provenance ของการทดลอง
# ใช้กับ: ทุกชุดผ่าน EEGDataset; sampler เฉพาะชุดที่เปิดใช้

from torch.utils.data import Dataset, Sampler
import torch
import numpy as np
import os.path as osp
import pickle
import random
from torch.utils.data import DataLoader
import csv
import os
import re
import sys


LEGACY_NORM_MODE = 'legacy_segment_f16'
TRAIN_CHANNEL_F32_NORM_MODE = 'train_channel_f32'
TRAIN_CHANNEL_F32_EPS = np.float32(1e-8)


# รองรับชื่อ normalization ใน checkpoint เก่า โดยคงพฤติกรรมเดิม
def canonical_norm_mode(norm_mode):
    """Return the explicit normalization name used by new runs.

    ``segment`` remains a read-only compatibility alias for old checkpoints.
    """
    if norm_mode in (None, 'segment', LEGACY_NORM_MODE):
        return LEGACY_NORM_MODE
    return norm_mode


# Float32 controls: fit mean/std จาก train เท่านั้น แล้วใช้ชุดเดิมกับ dev/eval
def compute_train_channel_stats(data_list, idx_channels=None, chunk_size=256):
    """Compute leakage-safe channel statistics from the train split only.

    Inputs are cast from their storage dtype to float32 before squaring or any
    reduction. Float64 accumulators avoid precision loss across millions of
    samples; persisted statistics and runtime normalization are float32.
    """
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')

    total_count = 0
    channel_sum = None
    channel_square_sum = None
    for each_subject in data_list:
        for start in range(0, len(each_subject), chunk_size):
            chunk = each_subject[start:start + chunk_size]
            chunk = chunk[:, idx_channels, :] if idx_channels is not None else chunk
            chunk = chunk.astype(np.float32, copy=False)
            if channel_sum is None:
                channel_sum = np.zeros(chunk.shape[1], dtype=np.float64)
                channel_square_sum = np.zeros(chunk.shape[1], dtype=np.float64)
            elif chunk.shape[1] != len(channel_sum):
                raise ValueError('All train arrays must have the same selected channel count')

            channel_sum += chunk.sum(axis=(0, 2), dtype=np.float64)
            channel_square_sum += np.square(chunk, dtype=np.float32).sum(
                axis=(0, 2), dtype=np.float64
            )
            total_count += int(chunk.shape[0] * chunk.shape[2])

    if channel_sum is None or total_count == 0:
        raise ValueError('Cannot compute normalization statistics from an empty train split')

    mean64 = channel_sum / total_count
    variance64 = np.maximum(channel_square_sum / total_count - np.square(mean64), 0.0)
    mean = mean64.astype(np.float32)[:, None]
    std = np.sqrt(variance64).astype(np.float32)[:, None]
    std = np.maximum(std, TRAIN_CHANNEL_F32_EPS)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('Non-finite train-channel normalization statistics')
    return mean, std, total_count


def seed_worker(worker_id):
    """Seed each DataLoader worker so that data loading is reproducible.
    Must be used with pl.seed_everything() (or torch.manual_seed) in the main process.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ทุกชุดทดลอง: ดึง window, เลือก channel และ normalize ให้ตรง config ตอนเทรน
class EEGDataset(Dataset):
    """norm_mode:
      'legacy_segment_f16' — พฤติกรรมเดิม: คำนวณ mean/std บน float16 โวลต์
                            ชื่อนี้ระบุชัดว่าไม่ใช่ textbook z-score
      'segment' — compatibility alias สำหรับ checkpoint เก่าเท่านั้น
      'subject' — z-score ด้วยสถิติของทั้งคน → ยังจัดการความต่างระหว่างคน
                  แต่เก็บความต่างของแอมพลิจูดภายในคน (seizure สูงกว่า background) ไว้
      'train_channel_f32' — fixed per-channel z-score จาก train split เท่านั้น;
                            runtime normalization เป็น float32
    """
    def __init__(self, data_list, label_list, idx_channels=None, norm_mode=LEGACY_NORM_MODE,
                 norm_clip=0.0, norm_stats=None):
        self.data_list = data_list
        self.label_list = label_list
        self.idx_channels = idx_channels
        self.norm_mode = canonical_norm_mode(norm_mode)
        # clip หลัง normalize — จำเป็นสำหรับ subject mode เท่านั้น
        # segment mode บังคับ std=1 ทุกหน้าต่างอยู่แล้ว ค่าสูงสุดจึงอยู่ที่ ~4-6 เสมอ
        # แต่ subject mode ปล่อย artifact (ขยับตัว/electrode pop) เข้ามาเต็มๆ:
        #   วัดจริง p50=3.9  p99=26.7  p99.9=58.7  max=128
        #   2.2% ของ segment เกิน ±20 → ที่ batch=128 เกือบทุก batch มี artifact
        #   → gradient ระเบิด, โมเดลไม่เรียนรู้ (run_5: val_auprc ดีสุดที่ ep0 แล้วเหวี่ยงลง)
        self.norm_clip = float(norm_clip)

        self.norm_stats = None
        if self.norm_mode == TRAIN_CHANNEL_F32_NORM_MODE:
            if norm_stats is None or len(norm_stats) != 2:
                raise ValueError('train_channel_f32 requires fixed (mean, std) train statistics')
            mean, std = (np.asarray(value, dtype=np.float32) for value in norm_stats)
            expected_channels = len(idx_channels) if idx_channels is not None else data_list[0].shape[1]
            expected_shape = (expected_channels, 1)
            if mean.shape != expected_shape or std.shape != expected_shape:
                raise ValueError(
                    f'train_channel_f32 stats must have shape {expected_shape}; '
                    f'got mean={mean.shape}, std={std.shape}'
                )
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError('train_channel_f32 statistics must be finite with std > 0')
            self.norm_stats = (mean, np.maximum(std, TRAIN_CHANNEL_F32_EPS))
        elif self.norm_mode == 'subject':
            # สถิติต่อ channel ของทั้งคน — คิดครั้งเดียวตอนสร้าง dataset
            #
            # ⚠️ ต้อง subsample + cast float32 ก่อน: load_data() เก็บเป็น float16
            #    ซึ่ง numpy ไม่มี SIMD path ให้ → reduction ช้ากว่า float32 ~8 เท่า
            #    (วัดจริง: 4.86 vs 0.63 s/subject → 579 คน = 47 นาที vs 6 นาที)
            #    การประมาณ mean/std ใช้ ~256 segment ก็แม่นเกินพอ (>200,000 sample/channel)
            MAX_SEG = 256
            self.subj_stats = []
            for d in self.data_list:
                stride = max(1, len(d) // MAX_SEG)
                arr = d[::stride]
                arr = arr[:, idx_channels, :] if idx_channels is not None else arr
                arr = arr.astype(np.float32)
                self.subj_stats.append((arr.mean(axis=(0, 2))[:, None],
                                        arr.std(axis=(0, 2))[:, None]))
        elif self.norm_mode not in (LEGACY_NORM_MODE, 'segment_f32'):
            raise ValueError(
                "norm_mode ต้องเป็น 'legacy_segment_f16', 'segment_f32' หรือ "
                f"'subject' (ได้ {norm_mode!r})"
            )

        self.index_map = []
        for list_idx, (d, l) in enumerate(zip(self.data_list, self.label_list)):
            for local_idx in range(l.shape[0]):
                self.index_map.append((list_idx, local_idx))

        self.total_length = len(self.index_map)

    def __getitem__(self, index):
        list_idx, local_idx = self.index_map[index]

        data = self.data_list[list_idx][local_idx] # shape: (chan, time)
        label = self.label_list[list_idx][local_idx]

        # Select specific channels
        if self.idx_channels is not None:
            data = data[self.idx_channels, :]

        # ══════════════════════════════════════════════════════════════════════
        # ⛔ ห้ามเติม data = data.astype(np.float32) ตรงนี้ ถ้าไม่พร้อมเทรนใหม่ทั้งหมด
        #
        # data เก็บเป็น float16 หน่วยโวลต์ (median |ค่า| ~6e-06) ซึ่งอยู่ในโซน subnormal
        # ของ float16 (ค่าปกติต่ำสุด 6.1e-05) → mean/std ที่คำนวณจึงคลาดเคลื่อน
        # ผลคือ output ของ normalization มี std ~14 ไม่ใช่ 1.0 ตามที่ z-score ควรได้
        #
        # เคยลอง cast เป็น float32 ก่อนคำนวณ (ซึ่ง "ถูกต้อง" กว่า) แล้วเกิด 2 ปัญหา:
        #   1. output std เปลี่ยนจาก ~14.4 -> ~0.88 (ต่างกัน 16 เท่า)
        #      → checkpoint เก่าทุกตัวที่เทรนบนสเกลเดิมใช้ไม่ได้ทันที
        #        (วัดจริง: ทุกโมเดลทาย non-seizure หมด TP=0 FP=0, AUPRC 61.7 -> 14.0)
        #   2. ผลแย่ลงด้วย: best val_auprc 0.4974 -> 0.4431 (Deformer seed 2024)
        #
        # ตรวจแล้วว่าเสถียรในทางปฏิบัติ: ไม่มี inf/NaN เลย (0/2000 segment)
        # ⚠️ ตอนเขียนวิทยานิพนธ์ต้องอธิบายตามที่โค้ดทำจริง ไม่ใช่เขียนแค่ "z-score"
        # ══════════════════════════════════════════════════════════════════════

        if self.norm_mode == TRAIN_CHANNEL_F32_NORM_MODE:
            data = data.astype(np.float32)
            mean, std = self.norm_stats
        elif self.norm_mode == 'segment_f32':
            # per-window float32 reduction แต่ยังคง legacy denominator แบบ std + 1e-6 V
            # ใช้เฉพาะกับ checkpoint ที่เทรนด้วย preprocessing แบบนี้ (Deformer run_4)
            # อย่าใช้กับ checkpoint อื่น — สเกล input ต่างกัน 16 เท่า
            data = data.astype(np.float32)
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
        elif self.norm_mode == 'subject':
            # subject mode ต้อง cast float32 (สถิติเป็น float32 อยู่แล้ว)
            # โหมดนี้ยังไม่ถูกใช้ในผลการทดลองใดๆ — ดู log 2026-07-21 (run_5 เรียนไม่ได้)
            data = data.astype(np.float32)
            mean, std = self.subj_stats[list_idx]
        else:
            # Instance Normalization (per channel) — vectorized (เร็วกว่า ~3x)
            # [OLD] for-loop version (ผลเหมือนกันเป๊ะ แต่ช้ากว่า):
            # for c in range(data.shape[0]):
            #     mean = np.mean(data[c, :])
            #     std = np.std(data[c, :])
            #     data[c, :] = (data[c, :] - mean) / (std + 1e-6)
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
        if self.norm_mode == TRAIN_CHANNEL_F32_NORM_MODE:
            data = (data - mean) / std
        else:
            data = (data - mean) / (std + 1e-6)

        if self.norm_clip > 0:
            np.clip(data, -self.norm_clip, self.norm_clip, out=data)

        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.tensor(label).long()

        return data_tensor, label_tensor

    def __len__(self):
        return self.total_length


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def load_data_per_subject(load_path, sub):
    sub_code = 'sub' + str(sub) + '.pkl'
    path = osp.join(load_path, sub_code)
    with open(path, 'rb') as file:
        dataset = pickle.load(file)
    data = dataset['data']
    label = dataset['label']
    return data, label


def get_channel_info(load_path, graph_type):
    path_info = osp.join(load_path, 'dataset_info.pkl')
    with open(path_info, 'rb') as file:
        dataset_info = pickle.load(file)
    graph_idx = dataset_info[graph_type]
    original_order = dataset_info['BL']
    input_subgraph = False
    for item in graph_idx:
        if isinstance(item, list):
            input_subgraph = True
    idx_new = []
    num_chan_local_graph = []
    if not input_subgraph:
        for chan in graph_idx:
            idx_new.append(original_order.index(chan))
    else:
        for i in range(len(graph_idx)):
            num_chan_local_graph.append(len(graph_idx[i]))
            for chan in graph_idx[i]:
                idx_new.append(original_order.index(chan))
    return idx_new, num_chan_local_graph


COMMON16_LONGITUDINAL_ALIASES = [
    ('FP1-F7',), ('F7-T3', 'F7-T7'), ('T3-T5', 'T7-P7'), ('T5-O1', 'P7-O1'),
    ('FP2-F8',), ('F8-T4', 'F8-T8'), ('T4-T6', 'T8-P8'), ('T6-O2', 'P8-O2'),
    ('FP1-F3',), ('F3-C3',), ('C3-P3',), ('P3-O1',),
    ('FP2-F4',), ('F4-C4',), ('C4-P4',), ('P4-O2',),
]


# Transfer common16: บังคับชื่อและลำดับ bipolar channels ให้ตรงข้าม dataset
def get_selected_channel_info(load_path, graph_type, channel_subset='all'):
    """Resolve model input indices and enforce the cross-dataset common16 order."""
    if channel_subset == 'all':
        return get_channel_info(load_path, graph_type)
    if channel_subset != 'common16':
        raise ValueError(f'Unknown channel subset: {channel_subset}')
    if graph_type != 'BL':
        raise ValueError('common16 currently requires --graph-type BL')

    path_info = osp.join(load_path, 'dataset_info.pkl')
    with open(path_info, 'rb') as file:
        dataset_info = pickle.load(file)
    original_order = dataset_info['BL']
    indices = []
    resolved_names = []
    for aliases in COMMON16_LONGITUDINAL_ALIASES:
        matches = [name for name in aliases if name in original_order]
        if len(matches) != 1:
            raise ValueError(
                f'Expected exactly one common16 alias from {aliases}, found {matches} '
                f'in {path_info}'
            )
        resolved_names.append(matches[0])
        indices.append(original_order.index(matches[0]))
    if len(set(indices)) != 16:
        raise ValueError(f'common16 channel indices are not unique: {indices}')
    return indices, resolved_names


# โหลดข้อมูลรายคน; การ cast float16 ตรงนี้เป็นส่วนหนึ่งของ legacy data contract
def load_data(load_path, load_idx, keep_subject=False, concat=False):
    data, label = [], []
    total_files = len(load_idx)
    print(f"--> Start loading {total_files} files from: {load_path}")

    for i, idx in enumerate(load_idx):
        # Console debug (Updates on the same line)
        if (i + 1) % 10 == 0 or (i + 1) == total_files:
            sys.stdout.write(f"\r    Progress: [{i+1}/{total_files}] files loaded ({(i+1)/total_files*100:.1f}%)")
            sys.stdout.flush()

        data_per_sub, label_per_sub = load_data_per_subject(load_path=load_path, sub=idx)

        # Downcast to float16 to save RAM — เก็บเป็น "โวลต์" ตามที่ MNE คืนมา (ไม่คูณสเกล)
        #
        # ⚠️ อย่าคูณ 1e6 ให้เป็นไมโครโวลต์ — เคยลองแล้วผลแย่ลงชัดเจน
        #    เหตุผล: ค่าระดับโวลต์ (median |ค่า| ~6e-06) ตกอยู่ในโซน subnormal ของ float16
        #    (ค่าปกติต่ำสุด 6.1e-05) ทำให้เกิด quantization ที่ขั้นคงที่
        #    → หลัง z-score ต่อหน้าต่าง ค่า std ที่ออกมาผันตามแอมพลิจริงของหน้าต่างนั้น
        #      กล่าวคือ "แอมพลิจูด" ยังรอดผ่าน per-window normalization มาได้
        #
        #    วัดเทียบบน eval (AUROC ของ output std ต่อ seizure):
        #      โวลต์  + cast f32 ตอนคำนวณ : std 0.88±0.10  r=+0.981  AUROC 0.862
        #      โวลต์  ไม่ cast (legacy)   : std 14.4±9.2   r=+0.635  AUROC 0.801  <- baseline เดิม
        #      ไมโครโวลต์ + cast f32      : std 1.00±0.00  r=+0.789  AUROC 0.769  (แอมพลิจูดหาย)
        #      ไมโครโวลต์ ไม่ cast        : std 0.82±2.16  r=-0.688  AUROC 0.121  (ล้น พังกลับทาง)
        data_per_sub = data_per_sub.astype(np.float16)

        if keep_subject:
            data.append(data_per_sub)
            label.append(label_per_sub)
            assert concat is not True, "Please set concat False is keep_subject is True"
        else:
            if isinstance(data_per_sub, np.ndarray):
                data.append(data_per_sub)
            else:
                data.extend(data_per_sub)

            if isinstance(label_per_sub, np.ndarray):
                label.append(label_per_sub)
            else:
                label.extend(label_per_sub)

    if concat:
        data = np.concatenate(data)  # --> seg, chan, data
        label = np.concatenate(label)

    print("\n    [Done] Loaded data successfully.")
    return data, label










# Weighted-sampler baseline: เพิ่มโอกาสสุ่มคลาสที่มีตัวอย่างน้อย เฉพาะ train
def make_weighted_sampler(label_list):
    """คำนวณ per-sample weight จาก class frequency แล้ว return WeightedRandomSampler
    ใช้สำหรับ train loader เท่านั้น เพื่อแก้ class imbalance

    label_list: list of np.ndarray (เหมือนที่ load_data return)
    """
    from torch.utils.data import WeightedRandomSampler

    # รวม label ทั้งหมดเป็น 1D array
    all_labels = np.concatenate(label_list)          # shape: (N,)
    classes, counts = np.unique(all_labels, return_counts=True)

    print(f"    [Sampler] Class distribution: { {int(c): int(n) for c, n in zip(classes, counts)} }")
    print(f"    [Sampler] Imbalance ratio (majority/minority): {counts.max()/counts.min():.1f}x")

    # weight ของแต่ละ class = 1 / count → ทำให้ทุก class ถูก sample เท่ากัน
    class_weight = 1.0 / counts.astype(np.float64)
    class_weight_map = {int(c): w for c, w in zip(classes, class_weight)}

    # assign weight ให้ทุก sample
    sample_weights = np.array([class_weight_map[int(l)] for l in all_labels])
    sample_weights_tensor = torch.from_numpy(sample_weights).float()

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True    # ต้อง True เพราะ minority class มีน้อยกว่า epoch size
    )
    return sampler


# D3/D3.1: ปรับสัดส่วนผู้ป่วยด้วย power โดยไม่กำหนด seizure ratio โดยตรง
class PatientBalancedSampler(Sampler):
    """Temper patient contribution while preserving within-patient sampling.

    The epoch length stays equal to the full training set. Samples are uniform
    within each patient, so class prevalence is not explicitly altered. Multiple
    case files with the same patient ID are pooled before quotas are assigned.
    Per-window weight is ``n_patient ** (-power)``: power=0 reproduces the
    natural patient distribution and power=1 gives equal patient contribution.
    """
    def __init__(self, label_list, patient_ids, seed, power=1.0):
        if len(label_list) != len(patient_ids):
            raise ValueError(
                "patient_ids must align one-to-one with train subject files "
                f"(labels={len(label_list)}, patient_ids={len(patient_ids)})"
            )
        if not label_list:
            raise ValueError("PatientBalancedSampler requires non-empty data")
        if not 0 <= float(power) <= 1:
            raise ValueError("PatientBalancedSampler power must be in [0, 1]")

        grouped = {}
        offset = 0
        for labels, patient_id in zip(label_list, patient_ids):
            count = int(np.asarray(labels).reshape(-1).shape[0])
            if count < 1:
                raise ValueError(f"Patient {patient_id!r} has no samples")
            grouped.setdefault(str(patient_id), []).append(
                np.arange(offset, offset + count, dtype=np.int64)
            )
            offset += count

        self.patient_indices = {
            patient_id: np.concatenate(chunks)
            for patient_id, chunks in grouped.items()
        }
        self.patient_ids = list(self.patient_indices)
        self.total_samples = offset
        self.power = float(power)
        self._rng = np.random.default_rng(int(seed))
        self.epoch = 0
        base, remainder = divmod(self.total_samples, len(self.patient_ids))
        self.base_quota = base
        self.remainder = remainder
        counts = np.asarray(
            [len(self.patient_indices[patient_id]) for patient_id in self.patient_ids],
            dtype=np.int64,
        )
        if self.power == 0:
            quotas = counts.copy()
        elif self.power == 1:
            quotas = np.full(len(self.patient_ids), base, dtype=np.int64)
            quotas[:remainder] += 1
        else:
            patient_mass = counts.astype(np.float64) ** (1.0 - self.power)
            exact_quotas = self.total_samples * patient_mass / patient_mass.sum()
            quotas = np.floor(exact_quotas).astype(np.int64)
            quota_remainder = self.total_samples - int(quotas.sum())
            if quota_remainder:
                fractional = exact_quotas - quotas
                priority = np.argsort(-fractional, kind="stable")
                quotas[priority[:quota_remainder]] += 1
        self.patient_quotas = quotas
        print(
            "    [PatientSampler] "
            f"patients={len(self.patient_ids)}, samples/epoch={self.total_samples}, "
            f"power={self.power:g}, quota/patient={int(quotas.min())}..{int(quotas.max())}, "
            f"source-count-range={int(counts.min())}..{int(counts.max())}"
        )

    def __iter__(self):
        # Preserve D3's rotating remainder assignments at full balancing strength.
        remainder_start = self.epoch % len(self.patient_ids)
        selected = []
        for position, patient_id in enumerate(self.patient_ids):
            if self.power == 1:
                relative = (position - remainder_start) % len(self.patient_ids)
                quota = self.base_quota + int(relative < self.remainder)
            else:
                quota = int(self.patient_quotas[position])
            indices = self.patient_indices[patient_id]
            selected.append(
                self._rng.choice(indices, size=quota, replace=quota > len(indices))
            )
        epoch_indices = self._rng.permutation(np.concatenate(selected))
        self.epoch += 1
        return iter(epoch_indices.tolist())

    def __len__(self):
        return self.total_samples


# C1/C2/T3/T4: เก็บ positive ครบ แล้วหมุน negative ตาม ratio ในแต่ละ epoch
class RotatingBinaryUnderSampler(Sampler):
    """Keep every positive and rotate a fixed-ratio subset of negatives each epoch.

    Sampling is global across the training split. Negatives are sampled without
    replacement within an epoch. A seeded shuffled pool is consumed circularly,
    so successive epochs rotate coverage deterministically.
    """
    def __init__(self, label_list, neg_pos_ratio, seed):
        labels = np.concatenate(label_list).reshape(-1)
        unique = set(np.unique(labels).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(
                "RotatingBinaryUnderSampler requires binary labels encoded as 0/1 "
                f"(found {sorted(unique)})"
            )
        if neg_pos_ratio <= 0:
            raise ValueError("neg_pos_ratio must be > 0")

        self.positive_indices = np.flatnonzero(labels == 1).astype(np.int64)
        negative_indices = np.flatnonzero(labels == 0).astype(np.int64)
        if len(self.positive_indices) == 0 or len(negative_indices) == 0:
            raise ValueError("Undersampling requires at least one sample from each class")

        requested_negatives = int(np.floor(float(neg_pos_ratio) * len(self.positive_indices)))
        if requested_negatives < 1:
            raise ValueError("neg_pos_ratio selects zero negatives; increase the ratio")
        self.negatives_per_epoch = min(requested_negatives, len(negative_indices))
        self.neg_pos_ratio = float(neg_pos_ratio)
        self._rng = np.random.default_rng(int(seed))
        self._negative_pool = self._rng.permutation(negative_indices)
        self._negative_cursor = 0
        self.epoch = 0

        effective_ratio = self.negatives_per_epoch / len(self.positive_indices)
        print(
            "    [UnderSampler] "
            f"positive/all={len(self.positive_indices)}, "
            f"negative/epoch={self.negatives_per_epoch}, "
            f"effective negative:positive={effective_ratio:.3f}:1, "
            f"samples/epoch={len(self)}"
        )

    def _take_rotating_negatives(self):
        chunks = []
        remaining = self.negatives_per_epoch
        while remaining:
            available = len(self._negative_pool) - self._negative_cursor
            take = min(remaining, available)
            chunks.append(
                self._negative_pool[self._negative_cursor:self._negative_cursor + take]
            )
            self._negative_cursor += take
            remaining -= take
            if self._negative_cursor == len(self._negative_pool):
                self._negative_cursor = 0
        return np.concatenate(chunks)

    def __iter__(self):
        negative_indices = self._take_rotating_negatives()
        epoch_indices = np.concatenate((self.positive_indices, negative_indices))
        epoch_indices = self._rng.permutation(epoch_indices)
        self.epoch += 1
        return iter(epoch_indices.tolist())

    def __len__(self):
        return len(self.positive_indices) + self.negatives_per_epoch


# เลือก sampler ได้ครั้งละหนึ่งแบบ; validation/evaluation ใช้ distribution เดิม
def prepare_data_for_training(data_list, label_list, idx, batch_size, shuffle,
                              use_weighted_sampler=False, norm_mode=LEGACY_NORM_MODE,
                              norm_clip=0.0, train_neg_pos_ratio=0.0, train_patient_balanced=False,
                              patient_sampling_power=1.0, patient_ids=None, sampler_seed=0,
                              norm_stats=None):
    dataset = EEGDataset(data_list, label_list, idx_channels=idx, norm_mode=norm_mode,
                         norm_clip=norm_clip, norm_stats=norm_stats)
    sampler_count = sum((
        bool(use_weighted_sampler),
        train_neg_pos_ratio > 0,
        bool(train_patient_balanced),
    ))
    if sampler_count > 1:
        raise ValueError("Use only one train sampling method")
    if use_weighted_sampler:
        sampler = make_weighted_sampler(label_list)
        # WeightedRandomSampler และ shuffle=True ใช้พร้อมกันไม่ได้ใน DataLoader
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,
                            pin_memory=True, num_workers=8, persistent_workers=True,
                            worker_init_fn=seed_worker)
    elif train_neg_pos_ratio > 0:
        sampler = RotatingBinaryUnderSampler(
            label_list=label_list,
            neg_pos_ratio=train_neg_pos_ratio,
            seed=sampler_seed,
        )
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,
                            pin_memory=True, num_workers=8, persistent_workers=True,
                            worker_init_fn=seed_worker)
    elif train_patient_balanced:
        sampler = PatientBalancedSampler(
            label_list=label_list,
            patient_ids=patient_ids,
            seed=sampler_seed,
            power=patient_sampling_power,
        )
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,
                            pin_memory=True, num_workers=8, persistent_workers=True,
                            worker_init_fn=seed_worker)
    else:
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                            pin_memory=True, num_workers=8, persistent_workers=True,
                            worker_init_fn=seed_worker)
    return loader





def log_command(log_path, args):
    import json
    cmd_file = os.path.join(log_path, 'command.txt')
    with open(cmd_file, 'w') as f:
        f.write("=== Command Line ===\n")
        f.write("python " + " ".join(sys.argv) + "\n\n")
        f.write("=== Parsed Arguments ===\n")
        f.write(json.dumps(vars(args), indent=4))


def collect_dataset_manifests(args):
    """Collect immutable preprocessing manifests when the processed splits have them."""
    manifests = {}
    for split in ('train', 'dev', 'eval'):
        split_dir = osp.join(
            args.load_path,
            f'data_{args.data_format}_{args.dataset}_{args.label_type}_{split}',
        )
        manifest_path = osp.join(split_dir, 'dataset_manifest.json')
        if osp.exists(manifest_path):
            with open(manifest_path, 'r') as file:
                manifests[split] = __import__('json').load(file)
        else:
            manifests[split] = {
                'status': 'legacy_dataset_without_manifest',
                'path': osp.abspath(split_dir),
            }
    return manifests


# เก็บ config/แหล่งข้อมูล/สถานะ เพื่อย้อนจากผลลัพธ์กลับไปยังการทดลอง
def log_run_manifest(log_path, args, experiment_id, status, error=None):
    """Create/update a machine-readable lifecycle and provenance record."""
    import datetime
    import json

    ensure_path(log_path)
    path = osp.join(log_path, 'run_manifest.json')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if osp.exists(path):
        with open(path, 'r') as file:
            manifest = json.load(file)
    else:
        manifest = {
            'experiment_id': experiment_id,
            'started_at_utc': now,
            'command': [sys.executable, *sys.argv],
            'arguments': vars(args),
            'datasets': collect_dataset_manifests(args),
        }
    manifest['status'] = status
    manifest['updated_at_utc'] = now
    if status == 'completed':
        manifest['completed_at_utc'] = now
    if error is not None:
        manifest['error'] = str(error)
    with open(path, 'w') as file:
        json.dump(manifest, file, indent=2, sort_keys=True)



def log2csv(csv_file, content):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row (column names)
        writer.writerow(["Metric", "Value"])

        # Write the data from the dictionary
        for key, value in content.items():
            writer.writerow([key, value])

    print(f"Data has been logged to {csv_file}")

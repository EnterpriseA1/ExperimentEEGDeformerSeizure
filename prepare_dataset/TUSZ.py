# หน้าที่: แปลง EDF/annotation เป็น TCP20, 200Hz, windows4วินาทีและ midpoint labels
# ใช้กับ: เตรียม TUSZ ทุก fixed split และ source pretraining

# ==============================================================================
# ANTIGRAVITY NEW FILE: TUSZ Preprocessing Script for seizure detection
# ==============================================================================
import os
import os.path as osp
import argparse
import pickle
import numpy as np
import mne
import glob
import json
import datetime
try:
    from .prepare_data import PrepareData
except ImportError:  # Support ``python prepare_dataset/TUSZ.py``.
    from prepare_data import PrepareData

class TUSZ(PrepareData):
    def __init__(self, args):
        super(TUSZ, self).__init__(args)

        # ACNS TCP-20 bipolar montage (มาตรฐาน TUH ใช้อ่าน seizure)
        # = TCP 22 คู่ตาม AAREADME.txt แต่ตัดคู่ A1-T3 (#8) และ T4-A2 (#13) ออก
        # เพราะไฟล์ montage 03_tcp_ar_a ไม่มี A1/A2 → TCP-20 ใช้ได้ 100% ของไฟล์
        # (ต่างจากเดิม 19-ch monopolar ที่ทิ้งไฟล์ 3.4% เพราะขาด FZ/PZ ซึ่งไม่อยู่ใน TCP อยู่แล้ว)
        self.bipolar_pairs = [
            ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
            ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
            ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'), ('C4', 'T4'),
            ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
            ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
        ]

        # electrode 17 ตัวที่ TCP-20 ต้องใช้ (ทุกไฟล์ใน TUSZ มีครบ 100%)
        self.required_electrodes = sorted({e for pair in self.bipolar_pairs for e in pair})

        # ชื่อ channel หลังหักลบ bipolar (นี่คือ channel จริงที่โมเดลเห็น = 20 ช่อง)
        self.original_order = ['{}-{}'.format(a, b) for a, b in self.bipolar_pairs]
        self.BL = self.original_order

        # Channel layouts (TSception and legacy metadata; defined using bipolar channel names)
        self.graph_fro = [['FP1-F7', 'FP1-F3'], ['FP2-F8', 'FP2-F4'], ['F7-T3', 'F8-T4'], ['F3-C3', 'F4-C4']]
        self.graph_gen = [['FP1-F7', 'FP2-F8'], ['C3-CZ', 'CZ-C4'], ['T3-C3', 'C4-T4']]
        self.graph_hem = [['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'T3-C3', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'C3-CZ'],
                          ['FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'C4-T4', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'CZ-C4']]
        self.TS = self.original_order  # TSception: ใช้ทั้ง 20 คู่

        # --- ANTIGRAVITY EDIT: START (Append split to label_type to separate data folders by split) ---
        self.args.label_type = f"{args.label_type}_{args.split}"
        # --- ANTIGRAVITY EDIT: END ---

    def clean_channel_name(self, name):
        """Standardize raw channel names from EDF files to standard 10-20 system"""
        name = name.upper()
        name = name.replace("EEG ", "")
        name = name.replace("-REF", "")
        name = name.replace("-LE", "")
        name = name.strip()
        # Handle spelling variations
        if name == "FP1": return "FP1"
        if name == "FP2": return "FP2"
        if name == "FZ": return "FZ"
        if name == "CZ": return "CZ"
        if name == "PZ": return "PZ"
        return name

    # อ่าน EDF/annotation แล้วสร้าง bipolar windows และ label จากกึ่งกลาง window
    def load_subject_data(self, edf_paths):
        """Loads EDF files and matches them with their corresponding csv_bi labels"""
        subject_data = []
        subject_labels = []
        record_metadata = []
        window_record_index = []
        window_index = []

        standard_channels = set(self.required_electrodes)

        for edf_path in sorted(edf_paths):
            # The label file is typically in the same folder with the same name but .csv_bi extension
            csv_bi_path = edf_path.replace(".edf", ".csv_bi")
            if not osp.exists(csv_bi_path):
                # Try finding in the directory if file naming differs slightly
                base_name = osp.splitext(osp.basename(edf_path))[0]
                dir_path = osp.dirname(edf_path)
                matching = glob.glob(osp.join(dir_path, f"{base_name}*.csv_bi"))
                if len(matching) > 0:
                    csv_bi_path = matching[0]
                else:
                    print(f"Warning: No label file found for {edf_path}. Skipping.")
                    continue

            try:
                # Read EDF using MNE
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                source_sfreq = float(raw.info['sfreq'])
                source_n_samples = int(raw.n_times)
                duration_seconds = source_n_samples / source_sfreq

                # Standardize channel names
                rename_dict = {}
                for ch in raw.ch_names:
                    cleaned = self.clean_channel_name(ch)
                    if cleaned in standard_channels:
                        rename_dict[ch] = cleaned

                raw.rename_channels(rename_dict)

                # Verify that all 17 electrodes required by TCP-20 exist in the file
                available = set(raw.ch_names)
                missing = standard_channels - available
                if len(missing) > 0:
                    print(f"Skipping {edf_path} because electrodes {missing} are missing.")
                    continue

                # Select only the required electrodes (monopolar, ยังไม่หักลบ)
                raw.pick_channels(self.required_electrodes)
                raw.reorder_channels(self.required_electrodes)

                # --- Clinical Filtering ---
                # 1. Notch filter to remove 60 Hz powerline noise (US standard for TUH dataset)
                raw.notch_filter(freqs=60.0, verbose=False)
                # 2. Band-pass filter (0.5 Hz to 99.0 Hz to avoid Nyquist at 200Hz)
                raw.filter(l_freq=0.5, h_freq=99.0, method='iir', verbose=False)

                # Resample to target sampling rate (e.g. 200 Hz)
                raw.resample(self.args.sampling_rate, verbose=False)

                # --- ANTIGRAVITY EDIT: START (Changed to float32 to reduce storage size by 50%) ---
                mono_data = raw.get_data().astype(np.float32)   # (17, time) monopolar
                # --- ANTIGRAVITY EDIT: END ---

                # --- TCP-20 bipolar derivation ---
                # หักลบคู่ bipolar ตรงนี้ (บนสัญญาณดิบที่ filter แล้ว) — ต้องทำ *ก่อน* instance norm
                # ที่อยู่ใน EEGDataset เสมอ ไม่งั้นค่าจะผิดความหมายทางสรีรวิทยา
                ch_idx = {ch: i for i, ch in enumerate(self.required_electrodes)}
                eeg_data = np.stack(
                    [mono_data[ch_idx[a]] - mono_data[ch_idx[b]] for a, b in self.bipolar_pairs]
                )   # (20, time)

                # Parse seizure onset-offset times from the .csv_bi file
                seizure_intervals = []
                with open(csv_bi_path, 'r') as f:
                    for line in f:
                        if line.startswith("#") or line.strip() == "":
                            continue
                        parts = line.strip().split(",")
                        if len(parts) < 4:
                            continue
                        channel, start, stop, label = parts[0], parts[1], parts[2], parts[3]
                        if channel == "TERM" and label == "seiz":
                            seizure_intervals.append((float(start), float(stop)))

                # Slice signal into non-overlapping windows (or overlapping based on overlap rate)
                win_len = int(self.args.segment * self.args.sampling_rate)
                step = int(win_len * (1 - self.args.overlap))

                total_samples = eeg_data.shape[1]
                idx_start = 0
                idx_end = win_len
                record_idx = len(record_metadata)
                local_window_idx = 0

                while idx_end <= total_samples:
                    # Slice signal segment
                    segment_data = eeg_data[:, idx_start:idx_end]

                    # Determine label based on window midpoint
                    t_start = idx_start / self.args.sampling_rate
                    t_end = idx_end / self.args.sampling_rate
                    midpoint = (t_start + t_end) / 2.0

                    label_val = 0
                    for s_start, s_end in seizure_intervals:
                        if s_start <= midpoint <= s_end:
                            label_val = 1
                            break

                    subject_data.append(segment_data)
                    subject_labels.append(label_val)
                    window_record_index.append(record_idx)
                    window_index.append(local_window_idx)

                    idx_start += step
                    idx_end = idx_start + win_len
                    local_window_idx += 1

                record_metadata.append({
                    'record_id': osp.splitext(osp.basename(edf_path))[0],
                    'source_path': osp.relpath(edf_path, self.data_path),
                    'duration_seconds': duration_seconds,
                    'source_sampling_rate': source_sfreq,
                    'source_num_samples': source_n_samples,
                    'processed_num_samples': total_samples,
                    'num_windows': local_window_idx,
                })

            except Exception as e:
                print(f"Error reading {edf_path}: {e}")
                continue

        if len(subject_data) > 0:
            metadata = {
                'records': record_metadata,
                # Resolve record_id as records[record_index[i]]['record_id'].
                # This avoids repeating a long string for every EEG window.
                'record_index': np.asarray(window_record_index, dtype=np.int32),
                'window_index': np.asarray(window_index, dtype=np.int32),
                'segment_seconds': float(self.args.segment),
                'sampling_rate': int(self.args.sampling_rate),
                'overlap': float(self.args.overlap),
                'step_samples': int(self.args.segment * self.args.sampling_rate
                                    * (1 - self.args.overlap)),
            }
            assert len(metadata['record_index']) == len(subject_data)
            return np.stack(subject_data), np.asarray(subject_labels), metadata
        return None, None, None

    def create_dataset(self):
        """Discovers all EDF files grouped by patient, processes them and saves subject-level files"""
        print("Discovering TUSZ EDF files...")

        # --- ANTIGRAVITY EDIT: START (Search only inside the specified split directory) ---
        split_path = osp.join(self.data_path, self.args.split)
        print(f"Searching inside split folder: {split_path}")
        edf_pattern = osp.join(split_path, "**", "*.edf")
        # --- ANTIGRAVITY EDIT: END ---
        all_edf_files = sorted(glob.glob(edf_pattern, recursive=True))

        # Filter out bipolar reference files if they exist (TUSZ sometimes has duplicate _bipolar.edf)
        all_edf_files = sorted(f for f in all_edf_files if "_bipolar" not in f)

        print(f"Found {len(all_edf_files)} EDF records in total.")

        # Group EDF files by patient ID (e.g. 'aaaaatvr')
        # Path structure: .../edf/train/aaaaatvr/s005_2015/...
        patient_records = {}
        for fpath in all_edf_files:
            parts = fpath.split(os.sep)
            # Find the patient directory name (which is always 8 chars starting with 'aaaaa')
            patient_id = None
            for p in parts:
                if p.startswith("aaaa") and len(p) == 8:
                    patient_id = p
                    break

            if patient_id:
                if patient_id not in patient_records:
                    patient_records[patient_id] = []
                patient_records[patient_id].append(fpath)

        patient_list = sorted(list(patient_records.keys()))
        print(f"Total unique patients discovered: {len(patient_list)}")

        # In quick-run or sub-sampling mode, limit subjects to speed up preprocessing
        if self.args.max_subjects > 0:
            patient_list = patient_list[:self.args.max_subjects]
            print(f"Sub-sampling enabled: Processing only first {self.args.max_subjects} patients.")

        manifest_subjects = []
        total_records = 0
        total_windows = 0
        total_seizure_windows = 0

        for idx, patient_id in enumerate(patient_list):
            edf_paths = sorted(patient_records[patient_id])
            print(f"[{idx+1}/{len(patient_list)}] Processing patient {patient_id} ({len(edf_paths)} files)...")

            data, labels, metadata = self.load_subject_data(edf_paths)
            if data is not None:
                metadata['patient_id'] = patient_id
                # Save using the base class save utility
                self.save(data, labels, idx, metadata=metadata)
                n_records = len(metadata['records'])
                n_windows = len(labels)
                n_seizure = int(np.sum(labels == 1))
                total_records += n_records
                total_windows += n_windows
                total_seizure_windows += n_seizure
                manifest_subjects.append({
                    'subject_file': f'sub{idx}.pkl',
                    'patient_id': patient_id,
                    'num_records': n_records,
                    'num_windows': n_windows,
                    'num_seizure_windows': n_seizure,
                })
                print(f"--> Saved sub{idx}.pkl: Data shape {data.shape}, Labels shape {labels.shape} (Seizures: {np.sum(labels == 1)})")
            else:
                print(f"--> No valid segments extracted for patient {patient_id}.")

        output_dir = osp.join(
            self.save_path,
            'data_{}_{}_{}'.format(
                self.args.data_format, self.args.dataset, self.args.label_type
            ),
        )
        os.makedirs(output_dir, exist_ok=True)
        manifest = {
            'schema_version': 1,
            'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'dataset': self.args.dataset,
            'split': self.args.split,
            'data_format': self.args.data_format,
            'label_type': self.args.label_type,
            'montage': 'ACNS_TCP20_bipolar',
            'channels': self.original_order,
            'sampling_rate': self.args.sampling_rate,
            'segment_seconds': self.args.segment,
            'overlap': self.args.overlap,
            'edf_ordering': 'lexicographic_path',
            'total_subjects': len(manifest_subjects),
            'total_records': total_records,
            'total_windows': total_windows,
            'total_seizure_windows': total_seizure_windows,
            'subjects': manifest_subjects,
        }
        with open(osp.join(output_dir, 'dataset_manifest.json'), 'w') as file:
            json.dump(manifest, file, indent=2, sort_keys=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path configuration
    parser.add_argument('--ROOT', type=str, default=osp.join(os.getcwd()))
    parser.add_argument('--dataset', type=str, default='TUSZ')
    parser.add_argument('--data-path', type=str, required=True)

    # Preprocessing hyperparameters
    # --- ANTIGRAVITY EDIT: START (Added split choice to separate train, dev, and eval sets) ---
    parser.add_argument('--split', type=str, default='train', choices=['train', 'dev', 'eval'],
                        help='Which dataset split to process')
    # --- ANTIGRAVITY EDIT: END ---
    parser.add_argument('--max-subjects', type=int, default=10,
                        help='Limit the number of subjects to process (0 for all, default 10 for quick start)')
    parser.add_argument('--label-type', type=str, default='seizure')
    parser.add_argument('--segment', type=int, default=4, help='Segment window length in seconds')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap rate (0.0 for non-overlapping)')
    parser.add_argument('--sampling-rate', type=int, default=200, help='Target resampling rate')
    parser.add_argument('--data-format', type=str, default='eeg')
    parser.add_argument('--graph-type', type=str, default='BL')
    parser.add_argument('--model', type=str, default='Deformer')

    args = parser.parse_args()
    if not 0.0 <= args.overlap < 1.0:
        parser.error('--overlap ต้องอยู่ในช่วง [0, 1)')

    # Initialize and run TUSZ preprocessing
    tusz_prep = TUSZ(args)
    tusz_prep.create_dataset()
    print("TUSZ dataset preparation completed successfully.")

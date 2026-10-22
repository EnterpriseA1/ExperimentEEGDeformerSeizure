# หน้าที่: อ่าน Siena annotation และสร้าง windows/metadata สำหรับผู้ป่วยทั้งหมดโดยไม่แบ่ง roles
# ใช้กับ: Siena all-patient cohort สำหรับ target LOSO

"""Preprocess the Siena Scalp EEG Database for seizure detection.

The output contract intentionally matches ``prepare_dataset/TUSZ.py``:
20 TCP bipolar channels, 200 Hz, 4-second windows, midpoint labels, and
subject-level pickle files.  All patients use non-overlapping windows; the LOSO runner assigns roles.
"""

import argparse
import datetime
import json
import os
import os.path as osp
import pickle
import re
import tempfile
from pathlib import Path

import mne
import numpy as np

try:
    from .prepare_data import PrepareData
    from .target_cohort import cohort_cases
except ImportError:  # Support ``python prepare_dataset/Siena.py``.
    from prepare_data import PrepareData
    from target_cohort import cohort_cases


TCP20_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"), ("C4", "T4"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]

# Typos in the official Seizures-list files, expressed annotation -> EDF.
ANNOTATION_FILENAME_ALIASES = {
    "PN01.edf": "PN01-1.edf",
    "PNO6-1.edf": "PN06-1.edf",
    "PNO6-2.edf": "PN06-2.edf",
    "PNO6-4.edf": "PN06-4.edf",
    "PN11-.edf": "PN11-1.edf",
}

# Confirmed by epilepsy2bids' public Siena annotation loader.  Keeping this
# explicit prevents a silent heuristic from changing clinical labels.
TIME_OVERRIDES = {
    ("PN00-3.edf", 0, "end"): "18.29.29",
}

EXPECTED_SUBJECTS = {
    "PN00", "PN01", "PN03", "PN05", "PN06", "PN07", "PN09",
    "PN10", "PN11", "PN12", "PN13", "PN14", "PN16", "PN17",
}
EXPECTED_SEIZURES = 47


def _extract_clock(line):
    """Return the last HH.MM.SS/HH:MM:SS clock in a line.

    The last clock deliberately selects PN10's explicitly stated electric
    onset and the corrected alternative end time when both appear.
    """
    line = line.replace("1 6.49.25", "16.49.25")
    matches = re.findall(r"(?<!\d)(\d{1,2})[.:](\d{2})[.:](\d{2})(?!\d)", line)
    if not matches:
        raise ValueError("No clock time found in line: {!r}".format(line.rstrip()))
    hour, minute, second = map(int, matches[-1])
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError("Invalid clock time in line: {!r}".format(line.rstrip()))
    return hour * 3600 + minute * 60 + second


def _relative_seconds(clock_seconds, registration_start):
    return float((clock_seconds - registration_start) % (24 * 3600))


# แปลงเวลา seizure เป็นวินาทีอ้างอิง EDF รวมกรณีข้ามเที่ยงคืน
def parse_subject_annotations(subject_dir):
    """Parse one official Siena seizure list into EDF-relative intervals."""
    subject_dir = Path(subject_dir)
    subject = subject_dir.name
    summary_path = subject_dir / "Seizures-list-{}.txt".format(subject)
    lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()

    intervals = {p.name: [] for p in sorted(subject_dir.glob("*.edf"))}
    registration_starts = {}
    current_file = None
    pending_start = None

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("file name:"):
            annotated_name = stripped.split(":", 1)[1].strip()
            current_file = ANNOTATION_FILENAME_ALIASES.get(annotated_name, annotated_name)
            if current_file not in intervals:
                raise ValueError(
                    "{} references missing EDF {!r}".format(summary_path, current_file)
                )
            pending_start = None
        elif stripped.lower().startswith("registration start time:"):
            if current_file is None:
                raise ValueError("Registration time appears before a file name")
            registration_starts[current_file] = _extract_clock(stripped)
        elif re.match(r"^(seizure )?start time:", stripped, flags=re.IGNORECASE):
            if current_file is None:
                raise ValueError("Seizure start appears before a file name")
            if current_file not in registration_starts:
                raise ValueError("No registration start known for {}".format(current_file))
            pending_start = _extract_clock(stripped)
        elif re.match(r"^(seizure )?end time:", stripped, flags=re.IGNORECASE):
            if current_file is None or pending_start is None:
                raise ValueError("Seizure end appears without a matching start")
            event_index = len(intervals[current_file])
            override = TIME_OVERRIDES.get((current_file, event_index, "end"))
            end_clock = _extract_clock(override if override is not None else stripped)
            registration_start = registration_starts[current_file]
            start = _relative_seconds(pending_start, registration_start)
            end = _relative_seconds(end_clock, registration_start)
            if end <= start:
                end += 24 * 3600
            intervals[current_file].append((start, end))
            pending_start = None

    if pending_start is not None:
        raise ValueError("Unmatched seizure start in {}".format(summary_path))
    return intervals


def audit_annotations(raw_root):
    """Validate all annotations and return intervals plus audit statistics."""
    all_intervals = {}
    total_events = 0
    for subject in sorted(EXPECTED_SUBJECTS):
        subject_dir = Path(raw_root) / subject
        subject_intervals = parse_subject_annotations(subject_dir)
        for edf_name, events in subject_intervals.items():
            edf_path = subject_dir / edf_name
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
            duration = raw.n_times / float(raw.info["sfreq"])
            for start, end in events:
                if not (0 <= start < end <= duration + 1.0):
                    raise ValueError(
                        "Out-of-range event {} {}: ({}, {}) duration={}".format(
                            subject, edf_name, start, end, duration
                        )
                    )
            total_events += len(events)
        all_intervals[subject] = subject_intervals
    if total_events != EXPECTED_SEIZURES:
        raise ValueError(
            "Expected {} Siena seizures, parsed {}".format(EXPECTED_SEIZURES, total_events)
        )
    return all_intervals, {"subjects": len(all_intervals), "seizures": total_events}


class Siena(PrepareData):
    def __init__(self, args):
        super().__init__(args)
        self.bipolar_pairs = TCP20_PAIRS
        self.required_electrodes = sorted({x for pair in TCP20_PAIRS for x in pair})
        self.original_order = ["{}-{}".format(a, b) for a, b in TCP20_PAIRS]
        self.BL = self.original_order
        self.TS = self.original_order
        self.graph_fro = [["FP1-F7", "FP1-F3"], ["FP2-F8", "FP2-F4"],
                          ["F7-T3", "F8-T4"], ["F3-C3", "F4-C4"]]
        self.graph_gen = [["FP1-F7", "FP2-F8"], ["C3-CZ", "CZ-C4"],
                          ["T3-C3", "C4-T4"]]
        self.graph_hem = [
            ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "T3-C3", "FP1-F3",
             "F3-C3", "C3-P3", "P3-O1", "C3-CZ"],
            ["FP2-F8", "F8-T4", "T4-T6", "T6-O2", "C4-T4", "FP2-F4",
             "F4-C4", "C4-P4", "P4-O2", "CZ-C4"],
        ]
        if args.overlap != 0.0:
            raise ValueError("LOSO cohort preprocessing requires --overlap 0")
        self.args.label_type = "{}_all".format(args.label_type)

    @staticmethod
    def clean_channel_name(name):
        name = name.upper().replace("EEG ", "").replace("-REF", "").replace("-LE", "")
        return name.strip()

    def _load_record(self, edf_path, seizure_intervals):
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        source_sfreq = float(raw.info["sfreq"])
        source_n_samples = int(raw.n_times)

        channel_map = {}
        for original in raw.ch_names:
            cleaned = self.clean_channel_name(original)
            if cleaned in self.required_electrodes:
                if cleaned in channel_map:
                    raise ValueError("Duplicate electrode {} in {}".format(cleaned, edf_path))
                channel_map[cleaned] = original
        missing = set(self.required_electrodes) - set(channel_map)
        if missing:
            raise ValueError("Missing electrodes {} in {}".format(sorted(missing), edf_path))

        picks = [channel_map[ch] for ch in self.required_electrodes]
        raw.pick(picks)
        raw.rename_channels({channel_map[ch]: ch for ch in self.required_electrodes})
        raw.reorder_channels(self.required_electrodes)
        raw.load_data()
        raw.notch_filter(freqs=50.0, method="iir", verbose=False)
        raw.filter(l_freq=0.5, h_freq=99.0, method="iir", verbose=False)
        raw.resample(self.args.sampling_rate, verbose=False)

        mono = raw.get_data().astype(np.float32)
        index = {ch: i for i, ch in enumerate(self.required_electrodes)}
        eeg = np.stack([mono[index[a]] - mono[index[b]] for a, b in self.bipolar_pairs])
        del mono, raw

        win_len = int(self.args.segment * self.args.sampling_rate)
        step = int(win_len * (1.0 - self.args.overlap))
        starts = np.arange(0, max(eeg.shape[1] - win_len + 1, 0), step, dtype=np.int64)
        windows = np.empty((len(starts), len(self.bipolar_pairs), win_len), dtype=np.float32)
        labels = np.zeros(len(starts), dtype=np.int64)
        for i, start_sample in enumerate(starts):
            windows[i] = eeg[:, start_sample:start_sample + win_len]
            midpoint = (start_sample + win_len / 2.0) / self.args.sampling_rate
            labels[i] = any(start <= midpoint <= end for start, end in seizure_intervals)

        metadata = {
            "record_id": edf_path.stem,
            "source_path": osp.relpath(str(edf_path), self.data_path),
            "duration_seconds": source_n_samples / source_sfreq,
            "source_sampling_rate": source_sfreq,
            "source_num_samples": source_n_samples,
            "processed_num_samples": int(eeg.shape[1]),
            "num_windows": int(len(labels)),
            "seizure_intervals_seconds": [list(x) for x in seizure_intervals],
        }
        return windows, labels, metadata

    def load_subject_data(self, subject, subject_intervals):
        data_parts, label_parts, records = [], [], []
        record_indices, window_indices = [], []
        subject_dir = Path(self.data_path) / subject
        for record_idx, edf_path in enumerate(sorted(subject_dir.glob("*.edf"))):
            data, labels, record = self._load_record(
                edf_path, subject_intervals[edf_path.name]
            )
            data_parts.append(data)
            label_parts.append(labels)
            records.append(record)
            record_indices.append(np.full(len(labels), record_idx, dtype=np.int32))
            window_indices.append(np.arange(len(labels), dtype=np.int32))
            print("  {}: windows={}, seizure_windows={}".format(
                edf_path.name, len(labels), int(labels.sum())
            ), flush=True)
        metadata = {
            "patient_id": subject,
            "records": records,
            "record_index": np.concatenate(record_indices),
            "window_index": np.concatenate(window_indices),
            "segment_seconds": float(self.args.segment),
            "sampling_rate": int(self.args.sampling_rate),
            "overlap": float(self.args.overlap),
            "step_samples": int(self.args.segment * self.args.sampling_rate * (1 - self.args.overlap)),
            "line_frequency_hz": 50.0,
            "bandpass_hz": [0.5, 99.0],
            "montage": "ACNS_TCP20_bipolar",
        }
        return np.concatenate(data_parts), np.concatenate(label_parts), metadata

    def _atomic_save(self, data, labels, subject_index, metadata):
        output_dir = Path(self.save_path) / "data_{}_{}_{}".format(
            self.args.data_format, self.args.dataset, self.args.label_type
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "sub{}.pkl".format(subject_index)
        with tempfile.NamedTemporaryFile(dir=str(output_dir), suffix=".tmp", delete=False) as tmp:
            pickle.dump({"data": data, "label": labels, "metadata": metadata}, tmp)
            tmp_path = tmp.name
        os.replace(tmp_path, target)
        return target

    def create_dataset(self):
        ordered_cases = cohort_cases(self.args.dataset, self.data_path)
        annotations, audit = audit_annotations(self.data_path)
        subjects = ordered_cases
        if self.args.subjects:
            requested = set(self.args.subjects)
            unknown = requested - set(subjects)
            if unknown:
                raise ValueError("Subjects are not in {} split: {}".format(
                    "all", sorted(unknown)
                ))
            subjects = [s for s in subjects if s in requested]
        if self.args.max_subjects > 0:
            subjects = subjects[:self.args.max_subjects]

        output_dir = Path(self.save_path) / "data_{}_{}_{}".format(
            self.args.data_format, self.args.dataset, self.args.label_type
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for index, subject in enumerate(subjects):
            target = output_dir / "sub{}.pkl".format(index)
            if target.exists() and not self.args.overwrite:
                raise FileExistsError("{} exists; pass --overwrite to replace it".format(target))
            print("[{}/{}] Processing {}...".format(index + 1, len(subjects), subject), flush=True)
            data, labels, metadata = self.load_subject_data(subject, annotations[subject])
            target = self._atomic_save(data, labels, index, metadata)
            entries.append({
                "subject_file": target.name,
                "patient_id": subject,
                "num_records": len(metadata["records"]),
                "num_windows": int(len(labels)),
                "num_seizure_windows": int(labels.sum()),
            })
            print("  saved {} shape={} seizure_windows={}".format(
                target, data.shape, int(labels.sum())
            ), flush=True)
            del data, labels

        dataset_info = {
            "original channel": self.original_order,
            "BL": self.BL, "TS": self.TS,
            "LGG-F": self.graph_fro, "LGG-H": self.graph_hem, "LGG-G": self.graph_gen,
        }
        with open(output_dir / "dataset_info.pkl", "wb") as handle:
            pickle.dump(dataset_info, handle)
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset": self.args.dataset,
            "split": "all",
            "cohort_order": ordered_cases,
            "role_assignment": "LOSO runner only",
            "annotation_audit": audit,
            "annotation_filename_aliases": ANNOTATION_FILENAME_ALIASES,
            "time_overrides": {"PN00-3.edf seizure_0 end": "18.29.29"},
            "montage": "ACNS_TCP20_bipolar",
            "channels": self.original_order,
            "sampling_rate": self.args.sampling_rate,
            "segment_seconds": self.args.segment,
            "overlap": self.args.overlap,
            "total_subjects": len(entries),
            "total_records": sum(x["num_records"] for x in entries),
            "total_windows": sum(x["num_windows"] for x in entries),
            "total_seizure_windows": sum(x["num_seizure_windows"] for x in entries),
            "subjects": entries,
        }
        with open(output_dir / "dataset_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)


def build_parser():
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--ROOT", default=str(project_root))
    parser.add_argument("--dataset", default="Siena")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--label-type", default="seizure")
    parser.add_argument("--segment", type=int, default=4)
    parser.add_argument("--sampling-rate", type=int, default=200)
    parser.add_argument("--data-format", default="eeg")
    parser.add_argument("--graph-type", default="BL")
    parser.add_argument("--model", default="Deformer")
    parser.add_argument("--overlap", type=float, default=0.0, choices=[0.0],
                        help="LOSO uses non-overlapping windows for every patient")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")
    if args.segment <= 0 or args.sampling_rate <= 0:
        parser.error("--segment and --sampling-rate must be positive")
    Siena(args).create_dataset()
    print("Siena {} preparation completed successfully.".format("all"))

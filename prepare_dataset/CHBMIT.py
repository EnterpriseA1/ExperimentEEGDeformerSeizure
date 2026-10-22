# หน้าที่: อ่าน CHB-MIT summary และสร้าง common16 windows โดยแยกผู้ป่วย
# ใช้กับ: CHBMIT all-patient cohort สำหรับ target LOSO

"""Preprocess CHB-MIT for TUSZ-to-CHB Deformer transfer.

Output contract: 16 common longitudinal bipolar channels, 200 Hz, four-second
windows, midpoint seizure labels, and one pickle per case. All cases use
non-overlapping windows. The LOSO runner assigns roles and groups chb01/chb21
as one patient.
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
except ImportError:
    from prepare_data import PrepareData
    from target_cohort import cohort_cases


COMMON16_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

EXPECTED_CASES = {"chb{:02d}".format(i) for i in range(1, 25)}
EXPECTED_SEIZURES = 198
SAME_PATIENT_CASES = {"chb01", "chb21"}

# The official RECORDS-WITH-SEIZURES file names chb07_18, while the official
# summary and the accompanying WFDB annotation both name chb07_19. Normalize
# this known upstream list typo before cross-validating annotation sources.
SEIZURE_RECORD_ALIASES = {
    "chb07/chb07_18.edf": "chb07/chb07_19.edf",
}


def canonical_channel_name(name):
    """Normalize an EDF bipolar channel name without changing its montage."""
    value = name.upper().strip().replace("EEG ", "")
    value = value.replace("-REF", "").replace("-LE", "").replace(" ", "")
    # MNE appends -0/-1 when the EDF contains duplicate labels.
    duplicate_suffix = re.match(r"^(.*)-\d+$", value)
    if duplicate_suffix and duplicate_suffix.group(1) in COMMON16_CHANNELS:
        value = duplicate_suffix.group(1)
    legacy_to_modern = {"T3": "T7", "T5": "P7", "T4": "T8", "T6": "P8"}
    parts = value.split("-")
    if len(parts) == 2:
        value = "-".join(legacy_to_modern.get(part, part) for part in parts)
    return value


def canonical_electrode_name(name):
    """Return an electrode name for CS2-referenced or bare referential labels."""
    value = name.upper().strip().replace("EEG ", "")
    value = value.replace("-REF", "").replace("-LE", "").replace(" ", "")
    # chb12_28 onward use bare electrode labels and spell O1 with a zero.
    value = {"01": "O1"}.get(value, value)
    valid_electrodes = {
        electrode for channel in COMMON16_CHANNELS for electrode in channel.split("-")
    }
    if value in valid_electrodes:
        return value
    parts = value.split("-")
    if len(parts) != 2 or parts[1] not in {"CS2"}:
        return None
    legacy_to_modern = {"T3": "T7", "T5": "P7", "T4": "T8", "T6": "P8"}
    return legacy_to_modern.get(parts[0], parts[0])


def parse_summary(summary_path):
    """Return ``{edf_name: [(start_seconds, end_seconds), ...]}``."""
    summary_path = Path(summary_path)
    intervals = {}
    current_file = None
    pending_start = None
    for raw_line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        file_match = re.match(r"^File Name:\s*(\S+\.edf)\s*$", line, re.IGNORECASE)
        if file_match:
            current_file = file_match.group(1)
            intervals.setdefault(current_file, [])
            pending_start = None
            continue
        start_match = re.match(
            r"^Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+(?:\.\d+)?)\s+seconds?",
            line, re.IGNORECASE,
        )
        if start_match:
            if current_file is None:
                raise ValueError("Seizure start before File Name in {}".format(summary_path))
            pending_start = float(start_match.group(1))
            continue
        end_match = re.match(
            r"^Seizure(?:\s+\d+)?\s+End Time:\s*(\d+(?:\.\d+)?)\s+seconds?",
            line, re.IGNORECASE,
        )
        if end_match:
            if current_file is None or pending_start is None:
                raise ValueError("Seizure end without start in {}".format(summary_path))
            end = float(end_match.group(1))
            if end <= pending_start:
                raise ValueError("Invalid seizure interval in {}: {}-{}".format(
                    summary_path, pending_start, end
                ))
            intervals[current_file].append((pending_start, end))
            pending_start = None
    if pending_start is not None:
        raise ValueError("Unmatched seizure start in {}".format(summary_path))
    return intervals


def audit_annotations(raw_root):
    all_intervals = {}
    total_events = 0
    total_records = 0
    seizure_record_path = Path(raw_root) / "RECORDS-WITH-SEIZURES"
    seizure_records_raw = {
        line.strip()
        for line in seizure_record_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    seizure_records = {
        SEIZURE_RECORD_ALIASES.get(path, path) for path in seizure_records_raw
    }
    for case in sorted(EXPECTED_CASES):
        case_dir = Path(raw_root) / case
        summary = case_dir / "{}-summary.txt".format(case)
        intervals = parse_summary(summary)
        edf_names = {p.name for p in case_dir.glob("*.edf")}
        summary_only = set(intervals) - edf_names
        edf_only = edf_names - set(intervals)
        omitted_seizure_records = {
            name for name in edf_only
            if "{}/{}".format(case, name) in seizure_records
        }
        if summary_only or omitted_seizure_records:
            raise ValueError(
                "Summary/EDF mismatch for {}: summary_only={}, omitted_seizure_records={}".format(
                    case, sorted(summary_only), sorted(omitted_seizure_records)
                )
            )
        # chb24's official summary lists only seizure-containing EDFs. Files
        # omitted from the summary are valid negatives only after confirming
        # that RECORDS-WITH-SEIZURES does not list them.
        for name in edf_only:
            intervals[name] = []
        all_intervals[case] = intervals
        total_events += sum(len(events) for events in intervals.values())
        total_records += len(edf_names)
    if total_events != EXPECTED_SEIZURES:
        raise ValueError("Expected {} seizures, parsed {}".format(EXPECTED_SEIZURES, total_events))
    parsed_seizure_records = {
        "{}/{}".format(case, edf_name)
        for case, case_intervals in all_intervals.items()
        for edf_name, events in case_intervals.items()
        if events
    }
    if parsed_seizure_records != seizure_records:
        raise ValueError(
            "Parsed seizure-record set differs from RECORDS-WITH-SEIZURES: "
            "parsed_only={}, official_only={}".format(
                sorted(parsed_seizure_records - seizure_records),
                sorted(seizure_records - parsed_seizure_records),
            )
        )
    wfdb_annotation_records = {
        str(path.relative_to(raw_root)).removesuffix(".seizures")
        for path in Path(raw_root).glob("chb[0-9][0-9]/*.edf.seizures")
    }
    if parsed_seizure_records != wfdb_annotation_records:
        raise ValueError(
            "Parsed seizure-record set differs from .edf.seizures files: "
            "parsed_only={}, wfdb_only={}".format(
                sorted(parsed_seizure_records - wfdb_annotation_records),
                sorted(wfdb_annotation_records - parsed_seizure_records),
            )
        )
    return all_intervals, {
        "cases": len(all_intervals),
        "unique_patients": 23,
        "records": total_records,
        "records_with_seizures": len(seizure_records),
        "seizures": total_events,
        "seizure_record_aliases": SEIZURE_RECORD_ALIASES,
    }


class CHBMIT(PrepareData):
    def __init__(self, args):
        super().__init__(args)
        self.original_order = list(COMMON16_CHANNELS)
        self.BL = list(COMMON16_CHANNELS)
        self.TS = list(COMMON16_CHANNELS)
        self.graph_fro = [["FP1-F7", "FP1-F3"], ["FP2-F8", "FP2-F4"]]
        self.graph_gen = [["FP1-F7", "FP2-F8"], ["P7-O1", "P8-O2"]]
        self.graph_hem = [COMMON16_CHANNELS[0:4] + COMMON16_CHANNELS[8:12],
                          COMMON16_CHANNELS[4:8] + COMMON16_CHANNELS[12:16]]
        if args.overlap != 0.0:
            raise ValueError("LOSO cohort preprocessing requires --overlap 0")
        self.args.label_type = "{}_all".format(args.label_type)

    # ใช้ bipolar ที่มีโดยตรง หรือสร้างจาก electrode คู่เดียวกันใน common reference
    @staticmethod
    def _channel_plan(raw, edf_path):
        """Resolve direct bipolar channels or derive them from a common reference."""
        available = {}
        for index, name in enumerate(raw.ch_names):
            canonical = canonical_channel_name(name)
            if canonical in COMMON16_CHANNELS and canonical not in available:
                available[canonical] = (index, name)
        missing = set(COMMON16_CHANNELS) - set(available)
        if not missing:
            indices = [available[name][0] for name in COMMON16_CHANNELS]
            return indices, [(i, None) for i in range(len(indices))], [
                available[name][1] for name in COMMON16_CHANNELS
            ], "direct_bipolar"

        referential = {}
        for index, name in enumerate(raw.ch_names):
            electrode = canonical_electrode_name(name)
            if electrode and electrode not in referential:
                referential[electrode] = (index, name)
        required_electrodes = {
            electrode for channel in COMMON16_CHANNELS for electrode in channel.split("-")
        }
        missing_electrodes = required_electrodes - set(referential)
        if missing_electrodes:
            raise ValueError(
                "Missing common16 channels {} and referential electrodes {} in {}".format(
                    sorted(missing), sorted(missing_electrodes), edf_path
                )
            )

        selected_indices = []
        selected_positions = {}
        for electrode in sorted(required_electrodes):
            selected_positions[electrode] = len(selected_indices)
            selected_indices.append(referential[electrode][0])
        derivations = [
            (selected_positions[left], selected_positions[right])
            for left, right in (channel.split("-") for channel in COMMON16_CHANNELS)
        ]
        source_channels = [
            "{} minus {}".format(referential[left][1], referential[right][1])
            for left, right in (channel.split("-") for channel in COMMON16_CHANNELS)
        ]
        return selected_indices, derivations, source_channels, "derived_common_reference"

    def _load_record(self, edf_path, seizure_intervals):
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        source_sfreq = float(raw.info["sfreq"])
        source_n_samples = int(raw.n_times)
        indices, derivations, source_channels, channel_mode = self._channel_plan(raw, edf_path)
        raw.pick(indices)
        raw.load_data()
        raw.notch_filter(freqs=60.0, method="iir", verbose=False)
        raw.filter(l_freq=0.5, h_freq=99.0, method="iir", verbose=False)
        raw.resample(self.args.sampling_rate, verbose=False)
        selected = raw.get_data().astype(np.float32)
        del raw
        eeg = np.stack([
            selected[left] if right is None else selected[left] - selected[right]
            for left, right in derivations
        ]).astype(np.float32, copy=False)
        del selected

        win_len = int(self.args.segment * self.args.sampling_rate)
        step = int(win_len * (1.0 - self.args.overlap))
        starts = np.arange(0, max(eeg.shape[1] - win_len + 1, 0), step, dtype=np.int64)
        windows = np.empty((len(starts), len(COMMON16_CHANNELS), win_len), dtype=np.float32)
        labels = np.zeros(len(starts), dtype=np.int64)
        for i, start_sample in enumerate(starts):
            windows[i] = eeg[:, start_sample:start_sample + win_len]
            midpoint = (start_sample + win_len / 2.0) / self.args.sampling_rate
            labels[i] = any(start <= midpoint <= end for start, end in seizure_intervals)
        del eeg

        return windows, labels, {
            "record_id": edf_path.stem,
            "source_path": osp.relpath(str(edf_path), self.data_path),
            "source_channel_names": source_channels,
            "channel_construction": channel_mode,
            "source_sampling_rate": source_sfreq,
            "source_num_samples": source_n_samples,
            "duration_seconds": source_n_samples / source_sfreq,
            "num_windows": int(len(labels)),
            "seizure_intervals_seconds": [list(x) for x in seizure_intervals],
        }

    def load_case_data(self, case, case_intervals):
        data_parts, label_parts, records = [], [], []
        record_indices, window_indices = [], []
        case_dir = Path(self.data_path) / case
        for record_index, edf_path in enumerate(sorted(case_dir.glob("*.edf"))):
            data, labels, record = self._load_record(edf_path, case_intervals[edf_path.name])
            data_parts.append(data)
            label_parts.append(labels)
            records.append(record)
            record_indices.append(np.full(len(labels), record_index, dtype=np.int32))
            window_indices.append(np.arange(len(labels), dtype=np.int32))
            print("  {}: windows={}, seizure_windows={}".format(
                edf_path.name, len(labels), int(labels.sum())
            ), flush=True)
        metadata = {
            "patient_id": "chb01_21" if case in SAME_PATIENT_CASES else case,
            "case_id": case,
            "records": records,
            "record_index": np.concatenate(record_indices),
            "window_index": np.concatenate(window_indices),
            "segment_seconds": float(self.args.segment),
            "sampling_rate": int(self.args.sampling_rate),
            "overlap": float(self.args.overlap),
            "step_samples": int(self.args.segment * self.args.sampling_rate * (1 - self.args.overlap)),
            "line_frequency_hz": 60.0,
            "bandpass_hz": [0.5, 99.0],
            "montage": "CHBMIT_common16_longitudinal_bipolar",
            "channels": list(COMMON16_CHANNELS),
        }
        return np.concatenate(data_parts), np.concatenate(label_parts), metadata

    def _atomic_save(self, data, labels, case_index, metadata):
        output_dir = Path(self.save_path) / "data_{}_{}_{}".format(
            self.args.data_format, self.args.dataset, self.args.label_type
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "sub{}.pkl".format(case_index)
        with tempfile.NamedTemporaryFile(dir=str(output_dir), suffix=".tmp", delete=False) as tmp:
            pickle.dump({"data": data, "label": labels, "metadata": metadata}, tmp)
            tmp_path = tmp.name
        os.replace(tmp_path, target)
        return target

    def create_dataset(self):
        if not (Path(self.data_path) / ".download_complete").exists():
            raise RuntimeError("CHB-MIT download/checksum verification is not complete")
        ordered_cases = cohort_cases(self.args.dataset, self.data_path)
        annotations, audit = audit_annotations(self.data_path)
        cases = ordered_cases
        if self.args.subjects:
            requested = set(self.args.subjects)
            unknown = requested - set(cases)
            if unknown:
                raise ValueError("Cases are not in {} split: {}".format("all", sorted(unknown)))
            cases = [case for case in cases if case in requested]
        if self.args.max_subjects > 0:
            cases = cases[:self.args.max_subjects]

        output_dir = Path(self.save_path) / "data_{}_{}_{}".format(
            self.args.data_format, self.args.dataset, self.args.label_type
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for index, case in enumerate(cases):
            target = output_dir / "sub{}.pkl".format(index)
            if target.exists() and self.args.resume and not self.args.overwrite:
                with open(target, "rb") as handle:
                    cached = pickle.load(handle)
                labels = np.asarray(cached["label"])
                metadata = cached.get("metadata", {})
                if metadata.get("case_id") != case:
                    raise ValueError(
                        "Resume mismatch for {}: expected {}, found {}".format(
                            target, case, metadata.get("case_id")
                        )
                    )
                expected_step = int(self.args.segment * self.args.sampling_rate)
                if (metadata.get("step_samples") != expected_step
                        or metadata.get("sampling_rate") != self.args.sampling_rate
                        or metadata.get("segment_seconds") != self.args.segment
                        or metadata.get("overlap") != 0.0):
                    raise ValueError("Resume preprocessing mismatch in {}".format(target))
                if metadata.get("channels") != list(COMMON16_CHANNELS):
                    raise ValueError("Resume channel mismatch in {}".format(target))
                entries.append({
                    "subject_file": target.name,
                    "patient_id": metadata["patient_id"],
                    "case_id": case,
                    "num_records": len(metadata["records"]),
                    "num_windows": int(len(labels)),
                    "num_seizure_windows": int(labels.sum()),
                })
                print("[{}/{}] Reusing {} for {}".format(
                    index + 1, len(cases), target, case
                ), flush=True)
                del cached, labels, metadata
                continue
            if target.exists() and not self.args.overwrite:
                raise FileExistsError("{} exists; pass --overwrite to replace it".format(target))
            print("[{}/{}] Processing {}...".format(index + 1, len(cases), case), flush=True)
            data, labels, metadata = self.load_case_data(case, annotations[case])
            target = self._atomic_save(data, labels, index, metadata)
            entries.append({
                "subject_file": target.name,
                "patient_id": metadata["patient_id"],
                "case_id": case,
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
            "montage": "CHBMIT_common16_longitudinal_bipolar",
            "channels": self.original_order,
            "sampling_rate": self.args.sampling_rate,
            "segment_seconds": self.args.segment,
            "overlap": self.args.overlap,
            "line_frequency_hz": 60.0,
            "bandpass_hz": [0.5, 99.0],
            "label_rule": "window_midpoint_inside_seizure_interval",
            "total_cases": len(entries),
            "total_unique_patients": len({x["patient_id"] for x in entries}),
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
    parser.add_argument("--dataset", default="CHBMIT")
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
    parser.add_argument("--resume", action="store_true",
                        help="Reuse validated subject files and continue an interrupted cohort")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")
    if args.segment <= 0 or args.sampling_rate <= 0:
        parser.error("--segment and --sampling-rate must be positive")
    CHBMIT(args).create_dataset()
    print("CHB-MIT {} preparation completed successfully.".format("all"))

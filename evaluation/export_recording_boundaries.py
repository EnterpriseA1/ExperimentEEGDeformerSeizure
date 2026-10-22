# หน้าที่: ดึงรอยต่อ recording จาก metadata ที่บันทึกตอน preprocessing
# ใช้กับ: Siena/CHB และ processed data ที่มี metadata

"""Export verified recording boundaries from processed subject metadata."""

import argparse
import glob
import os
import pickle
import re

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.processed_dir, "sub*.pkl")),
        key=lambda path: int(re.search(r"sub(\d+)\.pkl$", path).group(1)),
    )
    if not paths:
        raise FileNotFoundError("No sub*.pkl files in {}".format(args.processed_dir))

    subject_ids = []
    boundaries = []
    n_windows = []
    for path in paths:
        subject_id = int(re.search(r"sub(\d+)\.pkl$", path).group(1))
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        metadata = payload.get("metadata", {})
        record_index = np.asarray(metadata.get("record_index", []), dtype=np.int64)
        labels = np.asarray(payload["label"])
        if len(record_index) != len(labels):
            raise ValueError(
                "{} metadata record_index length {} != labels {}".format(
                    path, len(record_index), len(labels)
                )
            )
        cuts = np.flatnonzero(np.diff(record_index) != 0) + 1
        expected_records = len(metadata.get("records", []))
        if len(cuts) + 1 != expected_records:
            raise ValueError(
                "{} has {} boundaries but {} metadata records".format(
                    path, len(cuts), expected_records
                )
            )
        subject_ids.append(subject_id)
        boundaries.append(cuts.astype(np.int64))
        n_windows.append(len(labels))
        print(
            "sub{}: windows={} records={} boundaries={}".format(
                subject_id, len(labels), expected_records, cuts.tolist()
            )
        )

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        args.out,
        subject_ids=np.asarray(subject_ids, dtype=np.int32),
        n_windows=np.asarray(n_windows, dtype=np.int64),
        boundaries=np.asarray(boundaries, dtype=object),
        verified=np.asarray(True),
    )
    print("Saved verified boundaries: {}".format(args.out))


if __name__ == "__main__":
    main()

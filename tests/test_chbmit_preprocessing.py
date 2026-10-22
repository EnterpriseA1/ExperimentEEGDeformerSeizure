# หน้าที่: ทดสอบความถูกต้องของ CHB preprocessing
# ใช้กับ: CHB preprocessing

import tempfile
import unittest
from pathlib import Path

from prepare_dataset.CHBMIT import (
    COMMON16_CHANNELS,
    canonical_channel_name,
    parse_summary,
)


def test_parse_summary_handles_numbered_and_unnumbered_events(tmp_path):
    path = tmp_path / "chb99-summary.txt"
    path.write_text(
        "File Name: chb99_01.edf\n"
        "Number of Seizures in File: 1\n"
        "Seizure Start Time: 10 seconds\n"
        "Seizure End Time: 20 seconds\n"
        "File Name: chb99_02.edf\n"
        "Number of Seizures in File: 2\n"
        "Seizure 1 Start Time: 30 seconds\n"
        "Seizure 1 End Time: 40 seconds\n"
        "Seizure 2 Start Time: 50 seconds\n"
        "Seizure 2 End Time: 60 seconds\n"
    )
    assert parse_summary(path) == {
        "chb99_01.edf": [(10.0, 20.0)],
        "chb99_02.edf": [(30.0, 40.0), (50.0, 60.0)],
    }


def test_channel_normalization_handles_mne_duplicates_and_legacy_names():
    assert canonical_channel_name("EEG FP1-F7-REF") == "FP1-F7"
    assert canonical_channel_name("T7-P7-0") == "T7-P7"
    assert canonical_channel_name("F7-T3") == "F7-T7"
    assert len(COMMON16_CHANNELS) == 16


def test_cohort_covers_cases_without_split_manifest(tmp_path):
    from prepare_dataset.target_cohort import cohort_cases
    raw = tmp_path / "raw"
    for i in range(1, 25):
        (raw / "chb{:02d}".format(i)).mkdir(parents=True)
    cases = cohort_cases("CHBMIT", raw)
    assert set(cases) == {"chb{:02d}".format(i) for i in range(1, 25)}
    assert len(cases) == 24
    (raw / "chb24").rmdir()
    with unittest.TestCase().assertRaisesRegex(ValueError, "cohort mismatch"):
        cohort_cases("CHBMIT", raw)


class CHBPreprocessingTests(unittest.TestCase):
    def test_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_parse_summary_handles_numbered_and_unnumbered_events(Path(tmp))

    def test_channels(self):
        test_channel_normalization_handles_mne_duplicates_and_legacy_names()

    def test_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_cohort_covers_cases_without_split_manifest(Path(tmp))

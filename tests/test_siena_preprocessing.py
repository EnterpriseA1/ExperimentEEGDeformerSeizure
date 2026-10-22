# หน้าที่: ทดสอบความถูกต้องของ Siena preprocessing
# ใช้กับ: Siena preprocessing

import tempfile
import unittest
from pathlib import Path

from prepare_dataset.Siena import (
    _extract_clock,
    _relative_seconds,
    parse_subject_annotations,
)


class SienaAnnotationTests(unittest.TestCase):
    def test_clock_formats_and_last_clock_choice(self):
        self.assertEqual(_extract_clock("Start time: 21:51:02"), 21 * 3600 + 51 * 60 + 2)
        self.assertEqual(
            _extract_clock("start 15.43.53 clinical; 15.43.59 electric"),
            15 * 3600 + 43 * 60 + 59,
        )
        self.assertEqual(_extract_clock("Registration start time:1 6.49.25"), 16 * 3600 + 49 * 60 + 25)

    def test_midnight_rollover(self):
        start = 23 * 3600 + 18 * 60 + 10
        clock = 5 * 3600 + 25 * 60 + 49
        self.assertEqual(_relative_seconds(clock, start), 22059.0)

    def test_explicit_pn00_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            subject = Path(tmp) / "PN00"
            subject.mkdir()
            (subject / "PN00-3.edf").touch()
            (subject / "Seizures-list-PN00.txt").write_text(
                "File name: PN00-3.edf\n"
                "Registration start time: 18.15.44\n"
                "Seizure start time: 18.28.29\n"
                "Seizure end time: 19.29.29\n",
                encoding="utf-8",
            )
            intervals = parse_subject_annotations(subject)
            self.assertEqual(intervals["PN00-3.edf"], [(765.0, 825.0)])


if __name__ == "__main__":
    unittest.main()

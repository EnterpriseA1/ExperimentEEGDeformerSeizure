# หน้าที่: ทดสอบความถูกต้องของ canonical evaluation
# ใช้กับ: canonical evaluation

import unittest

import numpy as np

from evaluation.compute_metrics import (
    aggregate_per_subject,
    count_false_alarm_episodes,
)


class FalseAlarmEpisodeTest(unittest.TestCase):
    def test_contiguous_false_positive_windows_are_one_alarm(self):
        preds = np.array([0, 1, 1, 0, 1])
        labels = np.zeros(5)
        seam = np.array([1, 0, 0, 0, 0])
        self.assertEqual(count_false_alarm_episodes(preds, labels, seam), (2, 2))

    def test_episode_overlapping_positive_label_is_not_false_alarm(self):
        preds = np.array([0, 1, 1, 0])
        labels = np.array([0, 0, 1, 0])
        seam = np.array([1, 0, 0, 0])
        self.assertEqual(count_false_alarm_episodes(preds, labels, seam), (0, 1))

    def test_recording_seam_splits_positive_predictions(self):
        preds = np.array([1, 1, 1, 1])
        labels = np.zeros(4)
        seam = np.array([1, 0, 1, 0])
        self.assertEqual(count_false_alarm_episodes(preds, labels, seam), (2, 2))


class PerSubjectAggregateTest(unittest.TestCase):
    def test_mean_and_sample_std_across_runs(self):
        base = {
            "subject_id": 7,
            "n_windows": 10,
            "n_positive_windows": 2,
            "duration_hours": 1.0,
            "seizure_f1": 0.5,
            "sensitivity": 0.5,
            "specificity": 0.9,
            "fa_per_24h": 24.0,
        }
        rows = []
        for value in (0.2, 0.4, 0.6):
            rows.append([dict(base, auprc=value)])
        result = aggregate_per_subject(rows)[0]
        self.assertAlmostEqual(result["auprc_mean"], 0.4)
        self.assertAlmostEqual(result["auprc_sample_std"], 0.2)
        self.assertEqual(result["auprc_n"], 3)

    def test_no_positive_subject_keeps_undefined_metrics(self):
        rows = []
        for specificity in (0.9, 1.0, 0.8):
            rows.append([{
                "subject_id": 1,
                "n_windows": 10,
                "n_positive_windows": 0,
                "duration_hours": 1.0,
                "auprc": float("nan"),
                "seizure_f1": float("nan"),
                "sensitivity": float("nan"),
                "specificity": specificity,
                "fa_per_24h": 24.0,
            }])
        result = aggregate_per_subject(rows)[0]
        self.assertTrue(np.isnan(result["auprc_mean"]))
        self.assertEqual(result["auprc_n"], 0)
        self.assertAlmostEqual(result["specificity_sample_std"], 0.1)

if __name__ == "__main__":
    unittest.main()

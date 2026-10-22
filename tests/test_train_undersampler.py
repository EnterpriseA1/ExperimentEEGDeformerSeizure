# หน้าที่: ทดสอบความถูกต้องของ C1/C2/T3/T4
# ใช้กับ: C1/C2/T3/T4

import unittest

import numpy as np

from utils import RotatingBinaryUnderSampler


class RotatingBinaryUnderSamplerTest(unittest.TestCase):
    def setUp(self):
        self.labels = [np.array([1, 1] + [0] * 20)]

    def test_keeps_all_positives_and_requested_ratio(self):
        sampler = RotatingBinaryUnderSampler(self.labels, neg_pos_ratio=3, seed=2024)
        indices = list(iter(sampler))
        selected_labels = self.labels[0][indices]
        self.assertEqual(len(indices), 8)
        self.assertEqual(int((selected_labels == 1).sum()), 2)
        self.assertEqual(int((selected_labels == 0).sum()), 6)
        self.assertEqual(len(indices), len(set(indices)))

    def test_rotates_negatives_between_epochs(self):
        sampler = RotatingBinaryUnderSampler(self.labels, neg_pos_ratio=3, seed=2024)
        first = {i for i in sampler if self.labels[0][i] == 0}
        second = {i for i in sampler if self.labels[0][i] == 0}
        self.assertTrue(first.isdisjoint(second))

    def test_wraparound_does_not_duplicate_within_epoch(self):
        labels = [np.array([1, 1, 1] + [0] * 10)]
        sampler = RotatingBinaryUnderSampler(labels, neg_pos_ratio=2, seed=2024)
        list(iter(sampler))
        second = list(iter(sampler))
        self.assertEqual(len(second), len(set(second)))

    def test_seed_is_reproducible(self):
        first = RotatingBinaryUnderSampler(self.labels, neg_pos_ratio=1, seed=7)
        second = RotatingBinaryUnderSampler(self.labels, neg_pos_ratio=1, seed=7)
        self.assertEqual(list(iter(first)), list(iter(second)))

    def test_rejects_non_binary_labels(self):
        with self.assertRaises(ValueError):
            RotatingBinaryUnderSampler([np.array([0, 1, 2])], 1, 1)


if __name__ == '__main__':
    unittest.main()

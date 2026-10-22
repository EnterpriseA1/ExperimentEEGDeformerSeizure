# หน้าที่: ทดสอบความถูกต้องของ Focal/CE/CB ทุกชุด
# ใช้กับ: Focal/CE/CB ทุกชุด

import unittest

import torch
from torch.nn import functional as F

from models.model_handler import FocalLoss, effective_number_class_weights


class FocalLossTest(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[2.0, 0.5], [-0.5, 1.5], [0.2, 0.7]])
        self.targets = torch.tensor([0, 1, 1])

    def test_gamma_zero_without_alpha_matches_cross_entropy(self):
        actual = FocalLoss(gamma=0.0)(self.logits, self.targets)
        expected = F.cross_entropy(self.logits, self.targets)
        torch.testing.assert_close(actual, expected)

    def test_alpha_weights_loss_but_not_probability_term(self):
        alpha = torch.tensor([1.0, 4.0])
        ce = F.cross_entropy(self.logits, self.targets, reduction='none')
        pt = torch.exp(-ce)
        expected = (alpha[self.targets] * (1 - pt).pow(2) * ce).mean()

        actual = FocalLoss(alpha=alpha, gamma=2.0)(self.logits, self.targets)
        torch.testing.assert_close(actual, expected)

    def test_alpha_length_must_match_number_of_classes(self):
        with self.assertRaisesRegex(ValueError, "alpha has 3 entries"):
            FocalLoss(alpha=[1.0, 2.0, 3.0])(self.logits, self.targets)


class EffectiveNumberWeightTest(unittest.TestCase):
    def test_tusz_beta_99999_has_expected_mild_ratio(self):
        weights = effective_number_class_weights([1545089, 87234], 0.99999)
        self.assertAlmostEqual(float(weights.sum()), 2.0, places=6)
        self.assertAlmostEqual(float(weights[1] / weights[0]), 1.7181257, places=6)

    def test_paper_large_scale_beta_saturates_for_tusz_windows(self):
        weights = effective_number_class_weights([1545089, 87234], 0.999)
        torch.testing.assert_close(weights, torch.ones(2))

    def test_rejects_invalid_beta_or_empty_class(self):
        with self.assertRaises(ValueError):
            effective_number_class_weights([10, 1], 1.0)
        with self.assertRaises(ValueError):
            effective_number_class_weights([10, 0], 0.9)


if __name__ == '__main__':
    unittest.main()

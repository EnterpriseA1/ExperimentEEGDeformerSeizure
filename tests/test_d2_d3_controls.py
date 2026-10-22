# หน้าที่: ทดสอบความถูกต้องของ D2/D3/D3.1
# ใช้กับ: D2/D3/D3.1

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from training.task import TUSZ_train
from models.model_handler import (
    DLModel,
    split_backbone_head_parameters,
)
from utils import PatientBalancedSampler


class PatientBalancedSamplerTest(unittest.TestCase):
    def test_equal_patient_quota_and_duplicate_case_pooling(self):
        labels = [
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([0, 0]),
            np.array([1, 0, 0, 0]),
        ]
        sampler = PatientBalancedSampler(labels, ["p1", "p1", "p2"], seed=7)
        indices = list(sampler)
        self.assertEqual(len(indices), 12)
        self.assertEqual(sum(index < 8 for index in indices), 6)
        self.assertEqual(sum(index >= 8 for index in indices), 6)
        self.assertEqual(set(sampler.patient_ids), {"p1", "p2"})

    def test_seed_is_reproducible_and_labels_do_not_change_sampling(self):
        labels_a = [np.array([0] * 5), np.array([1] * 3)]
        labels_b = [np.array([1] * 5), np.array([0] * 3)]
        first = PatientBalancedSampler(labels_a, ["p1", "p2"], seed=2024)
        second = PatientBalancedSampler(labels_b, ["p1", "p2"], seed=2024)
        self.assertEqual(list(first), list(second))

    def test_tempered_power_half_interpolates_patient_contribution(self):
        labels = [np.zeros(4, dtype=int), np.zeros(16, dtype=int)]
        sampler = PatientBalancedSampler(labels, ["small", "large"], seed=7, power=0.5)
        indices = list(sampler)
        # Patient mass is sqrt(n): 2:4, so the 20-sample epoch gets quotas 7:13.
        self.assertEqual(sum(index < 4 for index in indices), 7)
        self.assertEqual(sum(index >= 4 for index in indices), 13)

    def test_power_zero_reproduces_natural_patient_counts(self):
        labels = [np.zeros(4, dtype=int), np.zeros(16, dtype=int)]
        sampler = PatientBalancedSampler(labels, ["small", "large"], seed=7, power=0.0)
        indices = list(sampler)
        self.assertEqual(sum(index < 4 for index in indices), 4)
        self.assertEqual(sum(index >= 4 for index in indices), 16)
        self.assertEqual(len(set(indices)), 20)

    def test_patient_sampling_power_is_validated(self):
        with self.assertRaisesRegex(ValueError, "power must be in"):
            PatientBalancedSampler(
                [np.zeros(2, dtype=int)], ["p1"], seed=7, power=1.1
            )

    def test_retired_patient_sampling_cannot_silently_run_natural_sampling(self):
        with self.assertRaisesRegex(ValueError, 'has been removed'):
            TUSZ_train(SimpleNamespace(train_patient_balanced=True), 'unused', 'unused')


class GradualUnfreezingTest(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            model="Deformer", num_chan=16, num_time=800, kernel_length=21,
            T=64, num_class=2, num_layers=6, AT=32, dropout=0.35,
            focal_alpha=0.0, focal_gamma=2.0, cb_beta=0.0,
            head_warmup_epochs=1, backbone_lr_scale=0.1,
            lr=3e-4, weight_decay=1e-5,
        )

    def test_freeze_optimizer_groups_and_unfreeze(self):
        model = DLModel(self._config())
        backbone, head = split_backbone_head_parameters(model.net)
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone))
        self.assertTrue(all(parameter.requires_grad for parameter in head))

        model._trainer = SimpleNamespace(max_epochs=5, current_epoch=0)
        optimizers, _ = model.configure_optimizers()
        optimizer = optimizers[0]
        self.assertEqual([group["name"] for group in optimizer.param_groups], ["backbone", "head"])
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 3e-5)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 3e-4)

        model.on_train_epoch_start()
        self.assertFalse(model.net.cnn_encoder[2].training)
        self.assertTrue(model.net.mlp_head.training)

        model._trainer.current_epoch = 1
        model.on_train_epoch_start()
        self.assertTrue(all(parameter.requires_grad for parameter in backbone))
        # Regression check: the unfreeze must also put the backbone back into
        # train() mode in the same call, not just flip requires_grad. Missing
        # this left BatchNorm/Dropout in eval mode for the transition epoch.
        self.assertTrue(model.net.cnn_encoder[2].training)
        self.assertTrue(model.net.training)

    # ค่า default ต้องไม่เปลี่ยนพฤติกรรมของรันเดิมทุกชุด
    def test_new_options_are_off_by_default(self):
        model = DLModel(self._config())
        self.assertEqual(model.surgical_train, ())

        model._trainer = SimpleNamespace(max_epochs=5, current_epoch=0)
        optimizers, schedulers = model.configure_optimizers()
        self.assertIsInstance(
            schedulers[0], torch.optim.lr_scheduler.CosineAnnealingLR
        )
        self.assertEqual(schedulers[0].T_max, 5)

    def test_legacy_path_keeps_single_group_and_cosine(self):
        config = self._config()
        config.head_warmup_epochs = 0
        config.backbone_lr_scale = 1.0
        model = DLModel(config)
        model._trainer = SimpleNamespace(max_epochs=5, current_epoch=0)
        optimizers, schedulers = model.configure_optimizers()
        self.assertEqual(len(optimizers[0].param_groups), 1)
        self.assertIsInstance(
            schedulers[0], torch.optim.lr_scheduler.CosineAnnealingLR
        )


class SurgicalFineTuningTest(unittest.TestCase):
    def _config(self, **overrides):
        config = SimpleNamespace(
            model="Deformer", num_chan=16, num_time=800, kernel_length=21,
            T=64, num_class=2, num_layers=6, AT=32, dropout=0.35,
            focal_alpha=0.0, focal_gamma=2.0, cb_beta=0.0,
            head_warmup_epochs=0, backbone_lr_scale=1.0,
            lr=3e-4, weight_decay=1e-5,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_groups_partition_every_parameter(self):
        from models.model_handler import split_parameter_groups
        model = DLModel(self._config())
        groups = split_parameter_groups(model.net)
        grouped = sum(len(v) for v in groups.values())
        self.assertEqual(grouped, len(list(model.net.parameters())))

    def test_only_selected_blocks_train_and_frozen_bn_stays_eval(self):
        model = DLModel(self._config(surgical_train="cnn_encoder,mlp_head"))
        from models.model_handler import split_parameter_groups
        groups = split_parameter_groups(model.net)
        for name, params in groups.items():
            expected = name in ("cnn_encoder", "mlp_head")
            self.assertTrue(all(p.requires_grad == expected for p in params), name)

        model._trainer = SimpleNamespace(max_epochs=5, current_epoch=0)
        optimizers, _ = model.configure_optimizers()
        optimized = sum(len(g["params"]) for g in optimizers[0].param_groups)
        self.assertEqual(
            optimized, len(groups["cnn_encoder"]) + len(groups["mlp_head"])
        )

        model.net.train()  # what Lightning does before the hook
        model.on_train_epoch_start()
        self.assertTrue(model.net.cnn_encoder[2].training)  # trained block
        self.assertFalse(model.net.transformer.training)    # frozen block

    def test_frozen_encoder_keeps_its_batchnorm_in_eval(self):
        model = DLModel(self._config(surgical_train="transformer,mlp_head"))
        model._trainer = SimpleNamespace(max_epochs=5, current_epoch=0)
        model.net.train()
        model.on_train_epoch_start()
        self.assertFalse(model.net.cnn_encoder[2].training)

    def test_rejects_conflicting_options(self):
        with self.assertRaises(ValueError):
            DLModel(self._config(surgical_train="cnn_encoder", head_warmup_epochs=1))
        with self.assertRaises(ValueError):
            DLModel(self._config(surgical_train="decoder"))

    def test_off_by_default(self):
        model = DLModel(self._config())
        self.assertEqual(model.surgical_train, ())
        self.assertTrue(all(p.requires_grad for p in model.net.parameters()))


if __name__ == "__main__":
    unittest.main()

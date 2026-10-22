# หน้าที่: ทดสอบความถูกต้องของ L2-SP
# ใช้กับ: L2-SP

import tempfile
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from models.l2sp import L2SP
from models.model_handler import DLModel
from training.task import initialize_model_from_checkpoint


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 3)
        self.bn = nn.BatchNorm1d(3)
        self.mlp_head = nn.Linear(3, 2)

    def forward(self, x):
        return self.mlp_head(self.bn(self.encoder(x)))


def config(**extra):
    values = dict(num_class=2, focal_gamma=2., focal_alpha=0., cb_beta=0.,
                  lr=1e-3, weight_decay=1e-5, l2sp_lambda=.2)
    values.update(extra)
    return SimpleNamespace(**values)


@patch.dict(os.environ, {'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1'})
class L2SPTests(unittest.TestCase):
    def test_exact_penalty_gradient_and_exclusions(self):
        net = TinyNet()
        sp = L2SP(net)
        with self.assertRaises(RuntimeError):
            sp(net)
        sp.capture(net)
        self.assertEqual(sp(net).item(), 0.)
        self.assertFalse(list(sp.parameters()))
        self.assertFalse(any('mlp_head' in n or 'running' in n for n in sp.names))
        with torch.no_grad():
            net.encoder.weight.add_(.5)
            net.mlp_head.weight.add_(100)
            net.bn.running_mean.add_(100)
        penalty = sp(net)
        self.assertAlmostEqual(penalty.item(), 6 * .25, places=5)
        (.5 * .2 * penalty).backward()
        torch.testing.assert_close(net.encoder.weight.grad, torch.full_like(net.encoder.weight, .1))
        self.assertIsNone(net.mlp_head.weight.grad)
        self.assertTrue(all(b.grad is None for b in sp.buffers()))
        self.assertEqual(penalty.dtype, torch.float32)
        with self.assertRaises(RuntimeError):
            sp.capture(net)

    @patch('models.model_handler.init_model', side_effect=lambda _: TinyNet())
    def test_initialization_and_checkpoint_restore(self, _):
        with tempfile.TemporaryDirectory() as d:
            source = DLModel(config(l2sp_lambda=0))
            with torch.no_grad():
                source.net.encoder.weight.fill_(.75)
            ck = Path(d)/'source.ckpt'
            torch.save({'state_dict': source.state_dict()}, ck)
            model = DLModel(config())
            audit = initialize_model_from_checkpoint(model, str(ck), 'network_strict', str(Path(d)/'audit.json'))
            self.assertTrue(audit['l2sp_reference_captured'])
            self.assertEqual(model.l2sp(model.net).item(), 0.)
            with torch.no_grad():
                model.net.encoder.weight.add_(.5)
            expected = model.l2sp(model.net).item()
            target_ck = Path(d)/'target.ckpt'
            torch.save({'state_dict': model.state_dict(), 'hyper_parameters': model.hparams,
                        'pytorch-lightning_version': pl.__version__}, target_ck)
            ck.unlink()
            restored = DLModel.load_from_checkpoint(target_ck)
            self.assertEqual(restored.l2sp(restored.net).item(), expected)

    @patch('models.model_handler.init_model', side_effect=lambda _: TinyNet())
    def test_optimizer_defaults_and_disabled_state(self, _):
        for coefficient, override, expected in [(0., None, [1e-5]), (.2, None, [0., 1e-5]), (0., 0., [0., 1e-5])]:
            model = DLModel(config(l2sp_lambda=coefficient, backbone_weight_decay=override))
            model._trainer = SimpleNamespace(max_epochs=2)
            optimizers, _ = model.configure_optimizers()
            self.assertEqual([g['weight_decay'] for g in optimizers[0].param_groups], expected)
            if coefficient == 0:
                self.assertFalse(any(k.startswith('l2sp.') for k in model.state_dict()))

    @patch('models.model_handler.init_model', side_effect=lambda _: TinyNet())
    def test_training_step_adds_penalty(self, _):
        model = DLModel(config())
        model.l2sp.capture(model.net)
        with torch.no_grad():
            model.net.encoder.weight.add_(.5)
        batch = (torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
        model.eval()
        task = model.criterion(model(batch[0]), batch[1])
        with patch.object(model, 'log'):
            total = model.training_step(batch, 0)
        torch.testing.assert_close(total, task + .15)

    @patch('models.model_handler.init_model', side_effect=lambda _: TinyNet())
    def test_lightning_cpu_fit_and_resume(self, _):
        with tempfile.TemporaryDirectory() as d:
            model = DLModel(config())
            model.l2sp.capture(model.net)
            anchor = model.l2sp.ref_0.clone()
            data = DataLoader(TensorDataset(torch.randn(8, 2), torch.tensor([0, 1]*4)), batch_size=4)
            options = dict(accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
                           enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
            trainer = pl.Trainer(max_epochs=1, **options)
            trainer.fit(model, data, data)
            ck = str(Path(d)/'resume.ckpt')
            trainer.save_checkpoint(ck)
            restored = DLModel(config())
            resumed = pl.Trainer(max_epochs=2, **options)
            resumed.fit(restored, data, data, ckpt_path=ck)
            torch.testing.assert_close(restored.l2sp.ref_0, anchor)
            self.assertTrue(bool(restored.l2sp.ready))
            self.assertTrue(torch.isfinite(restored.l2sp(restored.net)))


if __name__ == '__main__':
    unittest.main()

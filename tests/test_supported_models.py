"""Check the thesis model factory without optional/retired architectures."""
from types import SimpleNamespace
import unittest

import torch
from models.model_handler import init_model


class SupportedModelTests(unittest.TestCase):
    def test_thesis_models_forward_and_reload(self):
        torch.set_num_threads(1)
        torch.manual_seed(2024)
        for channels in (16, 20):
            data = torch.randn(2, channels, 800)
            for name in ('Deformer', 'EEGNet', 'TSception', 'Conformer', 'EEGViT'):
                with self.subTest(model=name, channels=channels):
                    args = SimpleNamespace(
                        model=name, num_chan=channels, num_time=800,
                        kernel_length=21, T=64, num_class=2, num_layers=6,
                        AT=32, dropout=0.5, sampling_rate=200,
                    )
                    model = init_model(args).eval()
                    restored = init_model(args).eval()
                    restored.load_state_dict(model.state_dict(), strict=True)
                    with torch.inference_mode():
                        result = model(data)
                        restored_result = restored(data)
                    self.assertEqual(tuple(result.shape), (2, 2))
                    self.assertTrue(torch.isfinite(result).all().item())
                    torch.testing.assert_close(result, restored_result)

    def test_unknown_architecture_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported model: retired_model'):
            init_model(SimpleNamespace(model='retired_model'))


if __name__ == '__main__':
    unittest.main()

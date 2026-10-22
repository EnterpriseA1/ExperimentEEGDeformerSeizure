# หน้าที่: ทดสอบความถูกต้องของ float32 normalization controls
# ใช้กับ: float32 normalization controls

import unittest

import numpy as np

from utils import (
    EEGDataset,
    TRAIN_CHANNEL_F32_NORM_MODE,
    compute_train_channel_stats,
)


class TrainChannelF32Test(unittest.TestCase):
    def test_stats_match_all_train_samples(self):
        train = [
            np.asarray(
                [
                    [[1.0, 3.0], [10.0, 14.0]],
                    [[5.0, 7.0], [18.0, 22.0]],
                ],
                dtype=np.float16,
            )
        ]

        mean, std, count = compute_train_channel_stats(train, chunk_size=1)

        self.assertEqual(count, 4)
        self.assertEqual(mean.dtype, np.float32)
        self.assertEqual(std.dtype, np.float32)
        np.testing.assert_allclose(mean[:, 0], [4.0, 16.0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            std[:, 0], [np.sqrt(5.0), np.sqrt(20.0)], rtol=1e-6
        )

    def test_eval_uses_fixed_train_stats_in_float32(self):
        mean = np.asarray([[4.0], [16.0]], dtype=np.float32)
        std = np.asarray([[2.0], [4.0]], dtype=np.float32)
        eval_data = [
            np.asarray([[[4.0, 6.0], [16.0, 20.0]]], dtype=np.float16)
        ]
        dataset = EEGDataset(
            eval_data,
            [np.asarray([1])],
            norm_mode=TRAIN_CHANNEL_F32_NORM_MODE,
            norm_stats=(mean, std),
        )

        normalized, label = dataset[0]

        self.assertEqual(normalized.numpy().dtype, np.float32)
        np.testing.assert_allclose(normalized.numpy(), [[0.0, 1.0], [0.0, 1.0]])
        self.assertEqual(label.item(), 1)


if __name__ == '__main__':
    unittest.main()

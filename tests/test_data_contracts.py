# หน้าที่: ทดสอบความถูกต้องของ shared data pipeline
# ใช้กับ: shared data pipeline

import json
import pickle
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from prepare_dataset.prepare_data import PrepareData
from utils import LEGACY_NORM_MODE, canonical_norm_mode, log_run_manifest


def make_args(root):
    return Namespace(
        ROOT=str(root),
        data_path='/raw/tusz',
        label_type='seizure_train',
        sampling_rate=200,
        data_format='eeg',
        dataset='TUSZ',
        load_path=str(root / 'processed'),
    )


class DataContractTest(unittest.TestCase):
    def test_old_segment_name_maps_to_explicit_legacy_mode(self):
        self.assertEqual(canonical_norm_mode('segment'), LEGACY_NORM_MODE)
        self.assertEqual(canonical_norm_mode(LEGACY_NORM_MODE), LEGACY_NORM_MODE)
        self.assertEqual(canonical_norm_mode('segment_f32'), 'segment_f32')

    def test_subject_pickle_preserves_optional_window_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prep = PrepareData(make_args(root))
            metadata = {
                'records': [{'record_id': 'record_a', 'duration_seconds': 8.0}],
                'record_index': np.array([0, 0], dtype=np.int32),
                'window_index': np.array([0, 1], dtype=np.int32),
            }
            prep.save(
                np.zeros((2, 1, 4), dtype=np.float32),
                np.array([0, 1]),
                sub=0,
                metadata=metadata,
            )

            output = root / 'data_processed/data_eeg_TUSZ_seizure_train/sub0.pkl'
            with output.open('rb') as file:
                saved = pickle.load(file)
            self.assertEqual(saved['metadata']['records'][0]['record_id'], 'record_a')
            np.testing.assert_array_equal(saved['metadata']['window_index'], [0, 1])

    def test_run_manifest_records_legacy_dataset_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root)
            run_dir = root / 'save/logs_TUSZ_Deformer/run_1'
            log_run_manifest(str(run_dir), args, 'run_1', status='started')

            with (run_dir / 'run_manifest.json').open() as file:
                manifest = json.load(file)
            self.assertEqual(manifest['status'], 'started')
            self.assertEqual(
                manifest['datasets']['train']['status'],
                'legacy_dataset_without_manifest',
            )


if __name__ == '__main__':
    unittest.main()

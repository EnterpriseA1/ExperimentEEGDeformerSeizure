"""Equivalence of legacy split sources and the role-free LOSO input pool."""
import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from prepare_dataset import Siena as siena_module, CHBMIT as chb_module
from prepare_dataset.target_cohort import CASE_ORDER, cohort_cases
from training import loso as common


def payload(obj, case, overlap):
    channels = obj.original_order
    step = int(800 * (1 - overlap))
    starts = np.arange(0, 1601, step)
    rec = np.repeat(np.arange(2, dtype=np.int32), len(starts))
    start = np.tile(starts, 2)
    windows = (np.arange(len(channels)*800, dtype=np.float32).reshape(1, len(channels), 800)
               + start[:, None, None] + rec[:, None, None]*3000) * np.float32(1e-6)
    labels = ((start + 400) / 200 <= 3).astype(np.int64)
    meta = dict(patient_id=common.patient_key(case), case_id=case,
                channels=list(channels), sampling_rate=200, segment_seconds=4.0,
                overlap=overlap, step_samples=step, record_index=rec,
                window_index=np.tile(np.arange(len(starts), dtype=np.int32), 2),
                records=[dict(seizure_intervals_seconds=[[1., 3.]]) for _ in range(2)])
    return windows.astype(np.float32), labels, meta


class TargetCohortTests(unittest.TestCase):
    def test_defaults_need_no_split_json_and_reject_overlap(self):
        for module in (siena_module, chb_module):
            args = module.build_parser().parse_args(["--data-path", "/unused/raw"])
            self.assertFalse(hasattr(args, 'split_manifest'))
            self.assertFalse(hasattr(args, 'split'))
            self.assertEqual(args.overlap, 0.0)
            args.overlap = .5
            with self.assertRaisesRegex(ValueError, 'requires --overlap 0'):
                getattr(module, args.dataset)(args)

    def test_inventory_rejects_missing_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in CASE_ORDER['CHBMIT'][:-1]: (Path(tmp)/case).mkdir()
            with self.assertRaisesRegex(ValueError, 'cohort mismatch'):
                cohort_cases('CHBMIT', tmp)

    def test_legacy_and_all_pools_are_exact_for_every_case_and_fold(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            for ds, module, train_count, dev_count in (
                    ('Siena', siena_module, 10, 2), ('CHBMIT', chb_module, 18, 3)):
                oldroot, newroot, raw = tmp/ds/'old', tmp/ds/'new', tmp/ds/'raw'
                raw.mkdir(parents=True)
                (raw/'.download_complete').touch()
                for case in CASE_ORDER[ds]: (raw/case).mkdir()
                args = module.build_parser().parse_args(['--ROOT', str(newroot), '--data-path', str(raw)])
                obj = getattr(module, ds)(args)
                loader = 'load_subject_data' if ds == 'Siena' else 'load_case_data'
                annotations = {case: {} for case in CASE_ORDER[ds]}
                with patch.object(module, 'audit_annotations', return_value=(annotations, {})), \
                     patch.object(obj, loader, side_effect=lambda case, _: payload(obj, case, 0.0)):
                    obj.create_dataset()
                # Construct old half-overlap train and non-overlap dev/eval inputs.
                groups = [CASE_ORDER[ds][:train_count],
                          CASE_ORDER[ds][train_count:train_count+dev_count],
                          CASE_ORDER[ds][train_count+dev_count:]]
                for role, cases in zip(('train', 'dev', 'eval'), groups):
                    directory = oldroot/f'data_processed/data_eeg_{ds}_seizure_{role}'
                    directory.mkdir(parents=True)
                    with (directory/'dataset_info.pkl').open('wb') as f:
                        pickle.dump({'original channel':obj.original_order,'BL':obj.BL,'TS':obj.TS}, f)
                    subjects = []
                    for i, case in enumerate(cases):
                        x,y,meta = payload(obj, case, .5 if role == 'train' else 0.)
                        with (directory/f'sub{i}.pkl').open('wb') as f:
                            pickle.dump(dict(data=x,label=y,metadata=meta),f)
                        subjects.append(dict(subject_file=f'sub{i}.pkl', patient_id=common.patient_key(case),case_id=case))
                    (directory/'dataset_manifest.json').write_text(json.dumps(dict(sampling_rate=200,segment_seconds=4,subjects=subjects)))
                with patch.object(common,'ROOT',oldroot): old = common.inventory(ds)
                with patch.object(common,'ROOT',newroot): new = common.inventory(ds)
                self.assertEqual([e['case'] for e in old],[e['case'] for e in new])
                folds = common.make_folds([e['patient'] for e in old])
                self.assertEqual(folds, common.make_folds([e['patient'] for e in new]))
                for fold in folds:
                    for role in ('train','dev','test'):
                        self.assertEqual([e['case'] for e in old if e['patient'] in fold[role]],
                                         [e['case'] for e in new if e['patient'] in fold[role]])
                with patch.object(common,'OUT',oldroot/'pool_output'): common.prepare_pool(ds,old)
                with patch.object(common,'OUT',newroot/'pool_output'): common.prepare_pool(ds,new)
                for case in CASE_ORDER[ds]:
                    for name in ('x.npy','y.npy','seam.npy','start_sample.npy','record_index.npy'):
                        a=oldroot/'pool_output'/ds/'pool'/case/name
                        b=newroot/'pool_output'/ds/'pool'/case/name
                        # Byte equality includes dtype, shape, sequence and actual samples.
                        self.assertEqual(a.read_bytes(),b.read_bytes(),f'{ds}/{case}/{name}')
                if ds == 'CHBMIT':
                    # A resume must not accept an old overlap grid as the new pool.
                    args.resume=True
                    target=newroot/'data_processed/data_eeg_CHBMIT_seizure_all/sub0.pkl'
                    with target.open('rb') as f: cached=pickle.load(f)
                    cached['metadata']['step_samples']=400
                    with target.open('wb') as f:pickle.dump(cached,f)
                    with patch.object(module,'audit_annotations',return_value=(annotations,{})):
                        with self.assertRaisesRegex(ValueError,'Resume preprocessing mismatch'):
                            obj.create_dataset()


if __name__ == '__main__': unittest.main()

# หน้าที่: ทดสอบความถูกต้องของ baseline transfer LOSO
# ใช้กับ: baseline transfer LOSO

import unittest
from training.baseline_transfer import job_plan, MODELS, target_root, source_dir
from training.loso import make_folds


class BaselineTransferLOSOTests(unittest.TestCase):
    def setUp(self):
        self.folds = {'Siena': make_folds([f'PN{i:02d}' for i in range(14)]),
                      'CHBMIT': make_folds(['chb01','chb21'] + [f'chb{i:02d}' for i in range(2,25) if i != 21])}

    def test_count_and_source_before_target(self):
        jobs = job_plan(self.folds)
        self.assertEqual(len(jobs), 152)
        self.assertTrue(all(j['stage'] == 'source_pretrain' for j in jobs[:4]))
        self.assertTrue(all(j['stage'] == 'target_transfer' for j in jobs[4:]))

    def test_each_model_dataset_has_exactly_one_job_per_patient(self):
        for model in MODELS:
            for dataset, n in [('Siena',14),('CHBMIT',23)]:
                jobs = [j for j in job_plan(self.folds) if j['stage'] == 'target_transfer' and j['model'] == model and j['dataset'] == dataset]
                self.assertEqual(len(jobs), n)
                self.assertEqual(len({j['fold'] for j in jobs}), n)
                self.assertEqual([j['fold'] for j in jobs], [f['id'] for f in self.folds[dataset]])

    def test_all_output_namespaces_unique(self):
        paths = [source_dir(m) for m in MODELS] + [target_root(m,d) for m in MODELS for d in self.folds]
        self.assertEqual(len(set(paths)), 12)
        self.assertTrue(all('baselines_transfer' in str(p) for p in paths))

    def test_target_budget_matches_deformer(self):
        for j in job_plan(self.folds)[4:]:
            self.assertEqual(j['epochs'], 80 if j['dataset'] == 'Siena' else 5)

    def test_reject_unknown_model_namespace(self):
        with self.assertRaises(AssertionError):
            source_dir('Deformer')


if __name__ == '__main__':
    unittest.main()

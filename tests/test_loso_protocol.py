# หน้าที่: ทดสอบความถูกต้องของ LOSO ทั้งสาม runner
# ใช้กับ: LOSO ทั้งสาม runner

import unittest
import numpy as np
from training.loso import make_folds, canonical_indices, patient_key, score


class LOSOProtocolTests(unittest.TestCase):
    def test_each_patient_is_test_once_and_roles_are_disjoint(self):
        patients = ['p0', 'p1', 'p2', 'p3', 'p4']
        folds = make_folds(patients)
        self.assertEqual([f['test'][0] for f in folds], patients)
        for f in folds:
            self.assertEqual(len(f['dev']), 2)
            self.assertEqual(set(f['train'] + f['dev'] + f['test']), set(patients))
            self.assertEqual(len(f['train'] + f['dev'] + f['test']), len(patients))

    def test_same_patient_chb01_chb21_never_split(self):
        folds = make_folds(['chb01', 'chb21', 'chb02', 'chb03', 'chb04'])
        self.assertEqual(len(folds), 4)
        self.assertEqual(patient_key('chb01'), patient_key('chb21'))
        self.assertEqual(sum(f['test'] == ['chb01_21'] for f in folds), 1)

    def test_half_overlap_reset_at_every_record(self):
        idx = canonical_indices([0,0,0,1,1,1,1], [0,1,2,0,1,2,3], 400)
        np.testing.assert_array_equal(idx, [0,2,3,5])

    def test_nonoverlap_is_unchanged(self):
        idx = canonical_indices([0,0,1,1], [0,1,0,1], 800)
        np.testing.assert_array_equal(idx, np.arange(4))

    def test_missing_window_rejected(self):
        with self.assertRaises(AssertionError):
            canonical_indices([0,0], [0,2], 400)

    def test_record_reordering_rejected(self):
        with self.assertRaises(AssertionError):
            canonical_indices([1,0], [0,0], 800)

    def test_confusion_and_seams(self):
        r = score(np.array([.9,.9,.1,.1]), np.array([0,0,1,0]), np.array([1,1,0,0],dtype=bool), .5)
        self.assertEqual([r[k] for k in ['tn','fp','fn','tp']], [1,2,1,0])
        self.assertEqual(r['false_alarm_episodes'], 2)

    def test_single_class_auc_is_not_fabricated(self):
        r = score(np.array([.1,.2]), np.array([0,0]), np.array([1,0],dtype=bool), .5)
        self.assertIsNone(r['auroc'])
        self.assertIsNone(r['auprc'])


if __name__ == '__main__':
    unittest.main()

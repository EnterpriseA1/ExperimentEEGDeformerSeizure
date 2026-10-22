# หน้าที่: ทดสอบความถูกต้องของ target CBCE LOSO
# ใช้กับ: target CBCE LOSO

import json
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from training.cbce import beta_for_ratio,weight_ratio,job_plan,OUT,common
from models.model_handler import FocalLoss,effective_number_class_weights

class CBCETargetTests(unittest.TestCase):
    def test_weight_ratio_and_ce(self):
        for neg,pos in [(89499,545),(749993,2634),(10000,100)]:
            beta=beta_for_ratio(neg,pos)
            self.assertAlmostEqual(weight_ratio(neg,pos,beta),2,places=8)
            weights=effective_number_class_weights([neg,pos],beta)
            torch.testing.assert_close(weights,torch.tensor([2/3,4/3]))
            logits=torch.tensor([[.1,.7],[1.,-.5]],requires_grad=True)
            targets=torch.tensor([1,0])
            loss=FocalLoss(alpha=weights,gamma=0)(logits,targets)
            expected=(F.cross_entropy(logits,targets,reduction='none')*weights[targets]).mean()
            torch.testing.assert_close(loss,expected)
            loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_reject_impossible_ratio(self):
        for neg,pos in [(0,10),(10,0),(10,10),(15,10)]:
            with self.assertRaises(ValueError):beta_for_ratio(neg,pos)

    def test_all222_unique_jobs(self):
        from prepare_dataset.target_cohort import CASE_ORDER
        p={'folds': {ds: common.make_folds(cases) for ds,cases in CASE_ORDER.items()}}
        jobs=job_plan(p['folds'])
        self.assertEqual(len(jobs),222)
        self.assertEqual(len({(j['model'],j['dataset'],j['fold'],j['arm']) for j in jobs}),222)
        self.assertEqual(sum(j['arm']=='scratch' for j in jobs),37)
        self.assertTrue(all(j['model']=='Deformer' for j in jobs if j['arm']=='scratch'))
    @unittest.skipUnless((OUT/'protocol.json').exists(), 'Historical weight audit requires saved experiment outputs')
    def test_saved_fold_weights_match_training_counts(self):
        p=json.loads((OUT/'protocol.json').read_text())
        for ds,folds in p['folds'].items():
            for fold in folds:
                self.assertFalse(set(fold['train']) & set(fold['test']+fold['dev']))
                old=common.OUT/ds/fold['id']/'transfer/data_counts.json'
                c=json.loads(old.read_text())['train']
                expected=[c['windows']-c['positive_windows'],c['positive_windows']]
                record=p['fold_losses'][ds][fold['id']]
                self.assertEqual(record['class_counts'],expected)
                self.assertAlmostEqual(weight_ratio(*expected,record['beta']),2,places=8)

if __name__=='__main__':unittest.main()

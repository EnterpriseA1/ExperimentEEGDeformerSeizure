"""Focused checks: pairing errors, Holm closed testing, t-test direction/interval."""
import importlib.util
import itertools
import math
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'analysis/compare_experiment_auprc.py'
spec=importlib.util.spec_from_file_location('paired_analysis',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PairedAnalysisTests(unittest.TestCase):
    def test_holm_against_closed_bonferroni(self):
        # Independent definition: all intersection hypotheses containing each test.
        p=np.array([.01,.04,.03,.2]); expected=[]
        for i in range(len(p)):
            values=[]
            for size in range(1,len(p)+1):
                for subset in itertools.combinations(range(len(p)),size):
                    if i in subset: values.append(min(1,size*min(p[list(subset)])))
            expected.append(max(values))
        np.testing.assert_allclose(m.holm(p),expected)
        np.testing.assert_allclose(m.holm(p),[.04,.09,.09,.2])
    def test_pairing_uses_ids(self):
        a={'p2':dict(value=.3,signature='s2',fold='f2'),'p1':dict(value=.1,signature='s1',fold='f1')}
        b={'p1':dict(value=.4,signature='s1',fold='f1'),'p2':dict(value=.5,signature='s2',fold='f2')}
        ids,av,bv=m.align(a,b)
        self.assertEqual(ids,['p1','p2']);np.testing.assert_allclose(bv-av,[.3,.2])
        del b['p1']
        with self.assertRaises(ValueError):m.align(a,b)
    def test_population_mismatch_fails(self):
        a={'p':dict(value=.1,signature='a',fold='f')};b={'p':dict(value=.2,signature='b',fold='f')}
        with self.assertRaises(ValueError):m.align(a,b)
    def test_t_and_interval_reference(self):
        a=np.ones(5)*.5;b=a+np.array([-1,0,1,2,3])*.1
        x=m.paired_summary(a,b)
        self.assertAlmostEqual(x['t'],math.sqrt(2));self.assertAlmostEqual(x['delta'],.1)
        self.assertEqual(x['df'],4)
        # Student t .975 quantile with df4 = 2.7764451052.
        self.assertAlmostEqual(x['ci95_low'],.1-2.7764451052*math.sqrt(.005),places=10)
        y=m.paired_summary(b,a)
        self.assertAlmostEqual(x['p'],y['p']);self.assertAlmostEqual(x['ci95_low'],-y['ci95_high'])
    def test_nonfinite_and_degenerate_fail(self):
        with self.assertRaises(ValueError):m.holm([.1,float('nan')])
        with self.assertRaises(ValueError):m.paired_summary(np.ones(4),np.ones(4))

if __name__=='__main__':unittest.main()

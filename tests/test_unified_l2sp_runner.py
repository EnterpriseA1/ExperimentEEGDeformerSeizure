"""Unified L2-SP runner tests; optional real CPU/synthetic-data training."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import training.l2sp as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        # Plan-only checks need a protocol fixture, not private experiment output.
        if self._testMethodName not in {
                'test_plans_separate_loss_run_and_evaluation',
                'test_plan_does_not_write_or_train'}:
            return
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        from prepare_dataset.target_cohort import CASE_ORDER
        ref = {'source_sha256': '0' * 64,
               'folds': {ds: runner.common.make_folds(cases)
                         for ds, cases in CASE_ORDER.items()}}
        directory = Path(tmp.name)
        (directory / 'protocol.json').write_text(json.dumps(ref))
        patched = patch.object(runner.common, 'OUT', directory)
        patched.start()
        self.addCleanup(patched.stop)

    def args(self, *extra):
        return runner.parse_args(['--dataset', 'CHBMIT', '--fold', '0',
                                  '--l2sp-lambda', '0.001', *extra])

    def test_loss_defaults(self):
        a = self.args()
        self.assertEqual(a.loss, 'focal')
        self.assertIsNone(a.cbce_ratio)
        self.assertFalse(a.evaluate_test)
        self.assertEqual(self.args('--loss', 'cbce').cbce_ratio, 2)

    def test_invalid_options(self):
        for flags in [('--cbce-ratio', '2'), ('--run-id', '../old'),
                      ('--l2sp-lambda', 'nan'), ('--l2sp-lambda', '-1'),
                      ('--loss', 'cbce', '--cbce-ratio', '1'),
                      ('--loss', 'cbce', '--cbce-ratio', 'inf')]:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args(*flags)

    def test_command_roundtrip(self):
        for loss in ('focal', 'cbce'):
            a = self.args('--loss', loss, '--run-id', 'rerun-2', '--evaluate-test', '--smoke')
            b = runner.parse_args(runner.worker_command(a, 0)[3:])
            for key in ('loss', 'run_id', 'evaluate_test', 'smoke', 'cbce_ratio', 'l2sp_lambda'):
                self.assertEqual(getattr(a, key), getattr(b, key))
            self.assertTrue(b._worker and b.execute)

    def test_dispatch_exact_loss_and_dev_only(self):
        for loss in ('focal', 'cbce'):
            with patch.object(runner.common, 'worker') as worker:
                runner.dispatch_worker(self.args('--loss', loss), Path('/tmp/unused'))
                kw = worker.call_args.kwargs
                self.assertEqual(kw['target_cbce_ratio'], 2 if loss == 'cbce' else None)
                self.assertFalse(kw['evaluate_test'])
                self.assertEqual(kw['l2sp_lambda'], .001)
                self.assertEqual(kw['backbone_weight_decay'], 0)

    def test_fresh_and_resume_safety(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'run1'
            protocol = {'loss': 'cbce', 'code_sha256': {'a': 'old'}}
            runner.prepare_output(out, protocol, False)
            result = out / 'result.json'; result.write_text('old result')
            with self.assertRaises(RuntimeError):
                runner.prepare_output(out, protocol, False)
            runner.prepare_output(out, protocol, True)
            with self.assertRaises(RuntimeError):
                runner.prepare_output(out, dict(protocol, loss='focal'), True)
            with self.assertRaises(RuntimeError):
                runner.prepare_output(out, dict(protocol, code_sha256={'a': 'new'}), True)
            with self.assertRaises(RuntimeError):
                runner.prepare_output(Path(td) / 'missing', protocol, True)
            runner.prepare_output(Path(td) / 'run2', protocol, False)
            self.assertEqual(result.read_text(), 'old result')

    def test_plans_separate_loss_run_and_evaluation(self):
        paths = []
        for flags in [(), ('--loss', 'cbce'), ('--run-id', 'second'),
                      ('--evaluate-test',), ('--smoke',)]:
            _, indices, out, protocol = runner.build_plan(self.args(*flags))
            self.assertEqual(indices, [0])
            self.assertEqual(protocol['seed'], 2024)
            self.assertNotIn('surgical_train', protocol)
            paths.append(out)
        self.assertEqual(len(set(paths)), 5)

    def test_plan_does_not_write_or_train(self):
        with patch.object(runner, 'prepare_output') as prep, patch.object(runner, 'validate_inputs') as valid, patch.object(runner.common, 'worker') as worker, contextlib.redirect_stdout(io.StringIO()):
            runner.main(['--dataset', 'Siena', '--all-folds', '--loss', 'cbce', '--l2sp-lambda', '.001'])
            prep.assert_not_called(); valid.assert_not_called(); worker.assert_not_called()

    @unittest.skipUnless(os.environ.get('EEG_RUN_CPU_INTEGRATION') == '1', 'opt-in CPU training')
    def test_real_cpu_training_two_fresh_ids_each_loss(self):
        import numpy as np
        import torch
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
        torch.set_num_threads(1)
        real_trainer = pl.Trainer
        real_plan = runner.build_plan
        real_subprocess_run = runner.subprocess.run
        fit_count = []
        array_requests = []
        def arrays(dataset, entries, patients):
            array_requests.append(list(patients))
            rng = np.random.default_rng(2024)
            return ([rng.normal(size=(8, 16, 800)).astype(np.float32)],
                    [np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int64)],
                    [np.arange(8)])
        def loader(*a, **kw):
            kw.update(num_workers=0, persistent_workers=False, pin_memory=False, batch_size=4)
            return DataLoader(*a, **kw)
        def trainer(**kw):
            kw.update(accelerator='cpu', devices=1, enable_model_summary=False)
            obj = real_trainer(**kw)
            original_fit = obj.fit
            def fit(*a, **k):
                fit_count.append(1)
                return original_fit(*a, **k)
            obj.fit = fit
            return obj
        # Exercise the parent/worker argument round-trip in-process so CPU and
        # synthetic-data substitutions never change production source files.
        def child(cmd, **kwargs):
            if isinstance(cmd, list) and len(cmd) > 2 and cmd[2] == str(Path(runner.__file__).resolve()):
                runner.main(cmd[3:])
            else:
                return real_subprocess_run(cmd, **kwargs)
        with tempfile.TemporaryDirectory() as td:
            def plan(args):
                ref, indices, _, protocol = real_plan(args)
                return ref, indices, Path(td) / args.loss / args.run_id, protocol
            with patch.object(runner, 'build_plan', side_effect=plan), patch.object(runner, 'validate_inputs'), patch.object(runner.common, 'arrays', side_effect=arrays), patch('torch.utils.data.DataLoader', side_effect=loader), patch('pytorch_lightning.Trainer', side_effect=trainer), patch.object(runner.subprocess, 'run', side_effect=child), contextlib.redirect_stdout(io.StringIO()):
                for loss in ('focal', 'cbce'):
                    for run_id in ('first', 'second'):
                        argv = ['--dataset', 'CHBMIT', '--fold', '0', '--loss', loss,
                                '--l2sp-lambda', '.001', '--run-id', run_id, '--smoke', '--execute']
                        runner.main(argv)
                        out = Path(td) / loss / run_id
                        results = list(out.rglob('dev_result.json'))
                        self.assertEqual(len(results), 1)
                        result = json.loads(results[0].read_text())
                        self.assertFalse(result['test_evaluated'])
                        self.assertTrue(Path(result['checkpoint']).is_file())
                        config = json.loads((results[0].parent / 'config.json').read_text())
                        self.assertEqual(config['focal_gamma'], 2 if loss == 'focal' else 0)
                        self.assertEqual(config['l2sp_lambda'], .001)
                        audit = results[0].parent / 'loss_audit.json'
                        self.assertEqual(audit.exists(), loss == 'cbce')
                        if audit.exists():
                            self.assertAlmostEqual(json.loads(audit.read_text())['positive_negative_weight_ratio'], 2)
                        runner.main(argv + ['--resume'])
                self.assertEqual(len(fit_count), 4)
                self.assertEqual(len(array_requests), 8)  # train + dev only, no test


if __name__ == '__main__':
    unittest.main()

"""Dependency-free check of the suite's G2C defaults against the recent SLURM."""
import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Protocol(unittest.TestCase):
    def test_g2c_defaults_follow_latest_source_init_slurm(self):
        reference = (ROOT / 'gta5_to_cityscapes_srmp.slurm').read_text(encoding='utf-8')
        src = re.search(r'SRC="\$\{SRC:-([^}]+)\}"', reference).group(1)
        budget = re.search(r'ADAPT_ITERS="\$\{ADAPT_ITERS:-([^}]+)\}"', reference).group(1)
        batch = re.search(r'BATCH_SIZE="\$\{BATCH_SIZE:-([^}]+)\}"', reference).group(1)
        period = re.search(r'CHECKPOINT_PERIOD="\$\{CHECKPOINT_PERIOD:-([^}]+)\}"', reference).group(1)
        self.assertIn('if [ "${AMP:-0}" = "1" ]', reference)
        bash = shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file():
            self.skipTest('bash unavailable')
        subprocess.run([bash, '-n', str(ROOT / 'run_source_aux_suite.sh')], check=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log, stub = root / 'calls.tsv', root / 'python-stub.sh'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\t" "$@" >> "$SUITE_STUB_LOG"\nprintf "\\n" >> "$SUITE_STUB_LOG"\n', encoding='utf-8')
            stub.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if not key.startswith(('G2C_', 'VH_', 'AUX_'))}
            env.update(PYTHON=stub.as_posix(), G2C_REPO=root.as_posix(),
                G2C_SOURCE_CKPT=src, SUITE_STUB_LOG=log.as_posix(), SEED='1', EXPERIMENT_ID='stub_protocol')
            for key in ('PREFLIGHT_ONLY', 'EVAL_PERIOD', 'DIFFUSION_UPDATES', 'DIFFUSION_EPOCHS'):
                env.pop(key, None)
            subprocess.run([bash, str(ROOT / 'run_source_aux_suite.sh')], cwd=ROOT,
                env=env, check=True, capture_output=True)
            calls = [line.strip().split('\t') for line in log.read_text().splitlines()]
            runs = [call for call in calls if call[:2] == ['train_DGLE_SRMP_remote.py', 'adapt'] and '--preflight' not in call]
            self.assertEqual(len(runs), 4)
            self.assertNotIn('--remask-phase', runs[0])
            for run in runs[1:]:
                self.assertEqual(run[run.index('--remask-phase')+1], 'entropy_kl')
            for call in calls:
                if '--preflight' in call:
                    self.assertEqual(call[call.index('--remask-phase')+1], 'entropy_kl')
            g2c = runs[-1]
            for option, expected in (('--task', 'g2c'), ('--source', src), ('--updates', budget),
                ('--schedule-updates', budget), ('--batch-size', batch), ('--eval-period', period),
                ('--refine-lr', '2.5e-4'), ('--ema', '.999'), ('--ema-period', '1')):
                self.assertEqual(g2c[g2c.index(option)+1], expected, option)
            self.assertIn('--from-source', g2c)
            self.assertNotIn('--bf16', g2c)
            self.assertNotIn('model_iter019000.pth', '\t'.join(g2c))
        tree = ast.parse((ROOT / 'scripts/g2c_source_aux.py').read_text(encoding='utf-8'))
        sha = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'SOURCE_SHA' for target in node.targets))
        self.assertEqual(sha, 'be51c0a1132255e124f7539cd85139d3c762e28723a666b108a35cef6ccf7cde')

    def test_g2c_noaux_control_uses_same_protocol_and_runs_no_vh_stages(self):
        bash = shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file():
            self.skipTest('bash unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log, stub = root / 'calls.tsv', root / 'python-stub.sh'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\t" "$@" >> "$SUITE_STUB_LOG"\nprintf "\\n" >> "$SUITE_STUB_LOG"\n', encoding='utf-8')
            stub.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if not key.startswith(('G2C_', 'VH_', 'AUX_'))}
            env.update(PYTHON=stub.as_posix(), G2C_REPO=root.as_posix(),
                G2C_SOURCE_CKPT='pretrained/G2C_model_iter020000.pth', SUITE_STUB_LOG=log.as_posix(),
                SEED='1', EXPERIMENT_ID='source_aux_suite_v2', AUX_WEIGHT='.9')
            for key in ('WAIT_FOR_PID', 'PREFLIGHT_ONLY', 'EVAL_PERIOD'):
                env.pop(key, None)
            subprocess.run([bash, str(ROOT / 'run_source_aux_suite.sh'), 'g2c-control'], cwd=ROOT,
                env=env, check=True, capture_output=True)
            calls = [line.strip().split('\t') for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 2)
            self.assertIn('--preflight', calls[0])
            self.assertNotIn('--preflight', calls[1])
            for call in calls:
                self.assertEqual(call[:2], ['train_DGLE_SRMP_remote.py', 'adapt'])
                for option, expected in (('--task', 'g2c'), ('--method', 'remask'), ('--remask-phase', 'entropy_kl'),
                    ('--aux-weight', '0'), ('--updates', '30000'), ('--schedule-updates', '30000'),
                    ('--batch-size', '4'), ('--refine-lr', '2.5e-4'), ('--ema', '.999'), ('--ema-period', '1'),
                    ('--output', 'results_gta5_to_cityscapes/source_aux/source_aux_suite_v2/remask/seed1')):
                    self.assertEqual(call[call.index(option)+1], expected, option)
                self.assertIn('--from-source', call)
                self.assertNotIn('--bf16', call)


    def test_g2c_aux_strength_only_runs_g2c_with_explicit_budget_and_cap(self):
        bash = shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file():
            self.skipTest('bash unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log, stub = root / 'calls.tsv', root / 'python-stub.sh'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\t" "$@" >> "$SUITE_STUB_LOG"\nprintf "\\n" >> "$SUITE_STUB_LOG"\n', encoding='utf-8')
            stub.chmod(0o755)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(('G2C_', 'VH_', 'AUX_'))}
            env.update(PYTHON=stub.as_posix(), G2C_REPO=root.as_posix(),
                G2C_SOURCE_CKPT='pretrained/G2C_model_iter020000.pth', SUITE_STUB_LOG=log.as_posix(),
                SEED='1', EXPERIMENT_ID='g2c_aux_lambda1_cap1_v1', AUX_WEIGHT='1.0',
                ENCODER_GRAD_CAP='1.0', G2C_UPDATES='7000', G2C_SCHEDULE_UPDATES='30000')
            for key in ('WAIT_FOR_PID', 'PREFLIGHT_ONLY', 'EVAL_PERIOD'):
                env.pop(key, None)
            subprocess.run([bash, str(ROOT / 'run_source_aux_suite.sh'), 'g2c-aux'],
                cwd=ROOT, env=env, check=True, capture_output=True)
            calls = [line.strip().split('\t') for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 2)
            self.assertIn('--preflight', calls[0])
            self.assertNotIn('--preflight', calls[1])
            for call in calls:
                for option, expected in (('--task', 'g2c'), ('--method', 'remask-relation'),
                    ('--aux-weight', '1.0'), ('--encoder-grad-cap', '1.0'),
                    ('--updates', '7000'), ('--schedule-updates', '30000'),
                    ('--remask-phase', 'entropy_kl'), ('--ema', '.999'), ('--ema-period', '1')):
                    self.assertEqual(call[call.index(option)+1], expected, option)


if __name__ == '__main__':
    unittest.main()

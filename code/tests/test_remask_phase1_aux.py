"""Runnable CPU checks for source-free candidate access and exact resume batches."""
import random
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from train_ReMask_phase1_aux import (
    ROOT, CandidateImages, ChunkBatches, auxiliary_targets, historical_config)


class FullPhase1Checks(unittest.TestCase):
    def test_suite_stops_at_7000_with_unchanged_30000_lr_horizon(self):
        bash = shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stub, log = root/'python.sh', root/'calls.tsv'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\t" "$@" >> "$CALL_LOG"\nprintf "\\n" >> "$CALL_LOG"\n')
            stub.chmod(0o755)
            env = dict(os.environ, PYTHON=stub.as_posix(), CALL_LOG=log.as_posix(), G2C_REPO=root.as_posix())
            for key in ('UPDATES','SCHEDULE_UPDATES','WAIT_FOR_PID','PREFLIGHT_ONLY'):
                env.pop(key,None)
            subprocess.run([bash,str(ROOT/'run_remask_phase1_aux_g2c.sh'),'all'],
                env=env,capture_output=True,check=True)
            calls = [line.strip().split('\t') for line in log.read_text().splitlines()]
            self.assertEqual(len(calls),6)
            for call in calls:
                self.assertEqual(call[call.index('--updates')+1],'7000')
                self.assertEqual(call[call.index('--schedule-updates')+1],'30000')
            self.assertEqual([c[c.index('--aux-weight')+1] for c in calls if '--preflight' not in c],['0','0.1','1.0'])

    def test_candidates_never_open_training_gt_and_refresh_uses_previous_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root/'leftImg8bit/train/demo/demo_000000_000000_leftImg8bit.png'
            image.parent.mkdir(parents=True)
            Image.new('RGB', (64,32), (70,100,130)).save(image)
            listing = root/'train.txt'
            listing.write_text('demo/'+image.name+'\n')
            previous = root/'cache/CTR_O'/image.name
            previous.parent.mkdir(parents=True)
            Image.new('L', (64,32), 7).save(previous)
            args = SimpleNamespace(city_root=root, city_train_list=listing,
                city_config=ROOT/'configs/deeplabv2_r101_SND_G2C.yaml',
                output=root/'cache', updates=4, schedule_updates=4, batch_size=1,
                tcr_period=2, smoke=True, data_limit=4)
            cfg = historical_config(args)
            self.assertEqual(cfg.REMASK.LAMBDA,.1)
            self.assertTrue(cfg.TCR.OPEN and cfg.REMASK.USE_CONTRASTIVE)
            self.assertEqual((cfg.DTU.EMA_WEIGHT,cfg.DTU.FIX_ITERATION),(.999,1))
            open_image = Image.open
            def guarded(path, *a, **kw):
                self.assertNotIn('gtFine',str(path))
                return open_image(path,*a,**kw)
            with patch.object(Image,'open',guarded):
                initial = CandidateImages(args,cfg,True)[0]
                refresh = CandidateImages(args,cfg,False)[0]
            self.assertEqual(tuple(initial[0].shape),(3,64,128))
            self.assertTrue(torch.all(initial[1]==255))
            self.assertTrue(torch.all(refresh[1]==7))
            self.assertEqual(tuple(refresh[1].shape),(32,64))

    def test_resume_batches_and_rng_are_identical(self):
        state = torch.get_rng_state().clone()
        random_state = random.getstate()
        whole = list(ChunkBatches(101,4,3000,3000,3010,1))
        resumed = list(ChunkBatches(101,4,3000,3006,3010,1))
        self.assertEqual(resumed,whole[6:])
        self.assertEqual(len({i for batch in whole for i,_ in batch}),40)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(random_state,random.getstate())

    def test_aux_never_scores_copy_paste_or_its_halo(self):
        pseudo = torch.ones(1,32,32,dtype=torch.long)
        mixed = torch.full_like(pseudo,255)
        mixed[:,16,16] = 7
        masked = auxiliary_targets(pseudo,mixed)
        self.assertTrue(torch.all(masked[:,8:25,8:25]==255))
        self.assertEqual(masked[0,0,0].item(),1)
        self.assertTrue(torch.all(pseudo==1))


if __name__ == '__main__':
    unittest.main()

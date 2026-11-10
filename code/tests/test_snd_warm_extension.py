"""Run with python -m tests.test_snd_warm_extension (torch, CPU)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from train_SND_remote import B2_PROTOCOL, configuration, prepare_aux_warm_extension, sha256_file, write_json


def check_extension():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        warm = root/'seed1_warm/warm'
        warm.mkdir(parents=True)
        source = root/'source.pth'
        source.write_bytes(b'source')
        state = {'iteration': 6000, 'stage': 'warm', 'protocol_signature': 'same',
                 'source_checkpoint_sha256': sha256_file(source), 'smoke_only': False,
                 'optimizer': {'momentum': torch.ones(3)}, 'rng': torch.get_rng_state()}
        torch.save(state, warm/'resume.pth')
        (warm/'last.pth').write_bytes(b'EMA weights')
        hashes = [sha256_file(warm/name) for name in ('resume.pth', 'last.pth')]
        stop = {'iteration': 6000, 'protocol_signature': 'same',
                'resume_checkpoint_sha256': hashes[0], 'checkpoint_sha256': hashes[1]}
        write_json(warm/'stopped_early.json', stop)
        args = SimpleNamespace(output=warm.parent, source_checkpoint=source, smoke_only=False,
                               extend_warm_to=9000, warm_stop_updates=6000, device='cpu')
        assert prepare_aux_warm_extension(args, 'same')
        assert not (warm/'stopped_early.json').exists()
        assert json.loads((warm/'stopped_early_step006000.json').read_text()) == stop
        assert hashes == [sha256_file(warm/name) for name in ('resume.pth', 'last.pth')]
        assert prepare_aux_warm_extension(args, 'same')  # Interrupted extension can resume.
        args.extend_warm_to = 8500
        try:
            prepare_aux_warm_extension(args, 'same')
        except ValueError:
            pass
        else:
            raise AssertionError('Changing an unfinished extension budget must fail')
        args.extend_warm_to = 9000
        state['iteration'] = stop['iteration'] = 9000
        torch.save(state, warm/'resume.pth')
        stop['resume_checkpoint_sha256'] = sha256_file(warm/'resume.pth')
        write_json(warm/'stopped_early.json', stop)
        assert not prepare_aux_warm_extension(args, 'same')  # Repeating the completed screen is a no-op.
        (root/'seed1').mkdir()
        write_json(root/'seed1/protocol.json', {})
        try:
            prepare_aux_warm_extension(args, 'same')
        except ValueError:
            pass
        else:
            raise AssertionError('A referenced warm must remain fixed')
        recipe = dict(protocol=B2_PROTOCOL, size=512, batch_size=6, base_lr=2e-5,
                      momentum=.9, weight_decay=.01, lr_power=1., checkpoint_period=500,
                      candidate_top_k=.5, candidate_update=500,
                      warm=dict(steps=20000, lr_max_iter=32000, meta=False, meta_top_k=.25,
                                meta_update=500, ema=.99, ema_interval=12))
        cfg = configuration(recipe, 'warm', args, warm)
        assert cfg.SOLVER.STOP_ITER == 9000 and cfg.SOLVER.MAX_ITER == 32000
        assert cfg.DTU.EMA_WEIGHT == .99 and cfg.DTU.FIX_ITERATION == 12
        assert recipe['warm']['steps'] == 20000 and args.warm_stop_updates == 6000
    print('Warm extension: preserved checkpoints/protocol/schedule, archive, interrupted resume, no-op and continuation guard OK')


if __name__ == '__main__':
    check_extension()

"""Run with python -m tests.test_snd_interrupted_init (torch, CPU)."""
import json
from pathlib import Path
import tempfile

import torch

from train_SND_remote import prepare_stage_output


def check():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for stage in ('warm', 'snd', 'meta-off'):
            output = root/stage
            prepare_stage_output(output, 'test')  # Fresh run.
            assert not output.exists()
            output.mkdir()
            started = output/'started.json'
            started.write_text(json.dumps({'protocol_signature': 'test'}))
            # Covers cancellation both before and after the initial checkpoint.
            if stage != 'meta-off':
                torch.save({'iteration': 0, 'protocol_signature': 'test', 'stage': stage}, output/'last.pth')
                (output/'evaluation_curve.jsonl').write_text('{"iteration": 0}\n')
            (output/'CTR').mkdir()
            (output/'CTR'/'partial-bank.txt').write_text('preserve')
            prepare_stage_output(output, 'test')
            archives = list(root.glob(f'{stage}_interrupted_init_*'))
            assert len(archives) == 1 and not output.exists()
            assert (archives[0]/'CTR'/'partial-bank.txt').read_text() == 'preserve'
            assert (archives[0]/'started.json').read_text() == '{"protocol_signature": "test"}'

        output = root/'warm'
        output.mkdir()
        (output/'started.json').write_text('{"protocol_signature": "test"}')
        cases = [
            ('last.pth', {'iteration': 500, 'protocol_signature': 'test', 'stage': 'warm'}),
            ('training_history.jsonl', '{"iteration": 1}\n'),
            ('evaluation_curve.jsonl', '{"iteration": 500}\n'),
            ('started.json', '{"protocol_signature": "other"}'),
        ]
        for name, value in cases:
            path = output/name
            if name.endswith('.pth'):
                torch.save(value, path)
            else:
                path.write_text(value)
            try:
                prepare_stage_output(output, 'test')
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'Unsafe automatic restart accepted: {name}')
            assert output.is_dir() and path.is_file()
            path.unlink()
        (output/'started.json').write_text('{"protocol_signature": "test"}')
        (output/'resume.pth').write_bytes(b'existing training state')
        prepare_stage_output(output, 'test')
        assert (output/'resume.pth').read_bytes() == b'existing training state'
        assert len(list(root.glob('warm_interrupted_init_*'))) == 1
    print('Interrupted warm/SND/meta-off init preserved; trained/protocol-mismatch runs blocked; resume unchanged OK')


if __name__ == '__main__':
    check()

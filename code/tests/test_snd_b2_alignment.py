"""One tiny check for four-scale SND feature alignment and gradients."""
import torch
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.snd_remote import SNDRemoteRuntime
from train_SND_remote import selected_warm_iteration
from train_SRMP_remote import sha256_file


def test_alignment():
    runtime = object.__new__(SNDRemoteRuntime)
    runtime.size = 64
    features = tuple(torch.randn(1, channels, 64//stride, 64//stride, requires_grad=True)
                     for channels, stride in zip((64, 128, 320, 512), (4, 8, 16, 32)))
    params = {'rescale_size': [torch.tensor([64]), torch.tensor([64])],
              'padding': [torch.tensor([0]) for _ in range(4)],
              'random_crop_axis': [torch.tensor([0]), torch.tensor([64]),
                                   torch.tensor([0]), torch.tensor([64])],
              'RandomHorizontalFlip': torch.tensor([False])}
    aligned = runtime.align_features(features, params)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(aligned, features))
    params['rescale_size'] = [torch.tensor([65]), torch.tensor([65])]
    params['random_crop_axis'] = [torch.tensor([1]), torch.tensor([65]),
                                  torch.tensor([1]), torch.tensor([65])]
    params['RandomHorizontalFlip'] = torch.tensor([True])
    shifted = runtime.align_features(features, params)
    assert all(a.shape == b.shape and torch.isfinite(a).all() for a, b in zip(shifted, features))
    sum(a.square().mean() for a in shifted).backward()
    assert all(feature.grad is not None and feature.grad.abs().sum() > 0 for feature in features)


def test_warm_peak_selection():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        warm = root/'warm'
        warm.mkdir()
        source = root/'source.pth'
        source.write_bytes(b'source')
        for name, iteration in (('last.pth', 20), ('best_target_val.pth', 5)):
            torch.save({'stage': 'warm', 'protocol_signature': 'fixed',
                        'source_checkpoint_sha256': sha256_file(source),
                        'smoke_only': False, 'iteration': iteration}, warm/name)
        (warm/'complete.json').write_text(json.dumps({
            'protocol_signature': 'fixed', 'iteration': 20,
            'checkpoint_sha256': sha256_file(warm/'last.pth')}))
        (warm/'observed_peak.json').write_text(json.dumps({
            'protocol_signature': 'fixed', 'iteration': 5}))
        args = SimpleNamespace(output=root, source_checkpoint=source,
                               warm_checkpoint=warm/'best_target_val.pth', smoke_only=False)
        assert selected_warm_iteration(args, 'fixed') == 5
        assert json.loads((root/'warm_selection.json').read_text())['iteration'] == 5
        args.warm_checkpoint = warm/'last.pth'
        try:
            selected_warm_iteration(args, 'fixed')
        except ValueError as error:
            assert 'same warm checkpoint' in str(error)
        else:
            raise AssertionError('SND and meta-off accepted different warm checkpoints')


if __name__ == '__main__':
    test_alignment()
    test_warm_peak_selection()
    print('SND alignment and warm peak selection OK')

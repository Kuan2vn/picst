"""One bounded check of SND recipe and auxiliary gradient routing."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from PIL import Image
from train_SND_g2c_aux import ROOT, Runtime
from train_ReMask_phase1_aux import CandidateImages, auxiliary_targets
from train_DGLE_SRMP_remote import add_aux_gradients
from scripts.g2c_source_aux import config


def test_gradient_routing_and_source_free_candidates():
    encoder = torch.nn.Linear(2,2,bias=False)
    head = torch.nn.Linear(2,1,bias=False)
    rebuilder = torch.nn.Linear(2,2,bias=False)
    x = torch.ones(1,2)
    features = encoder(x)
    head(features).square().sum().backward(retain_graph=True)
    head_grad = head.weight.grad.clone()
    main = encoder.weight.grad.clone()
    loss = rebuilder(features).square().sum()
    record = add_aux_gradients(loss,encoder,SimpleNamespace(rebuilder=rebuilder),1.,False,1.)
    assert torch.equal(head.weight.grad,head_grad)
    assert torch.isfinite(encoder.weight.grad).all() and rebuilder.weight.grad.abs().sum()>0
    assert (encoder.weight.grad-main).norm() <= main.norm()*1.00001
    assert record['aux_grad_ratio'] <= 1.00001
    pasted = torch.full((1,32,32),255,dtype=torch.long); pasted[:,16,16]=3
    assert (auxiliary_targets(torch.ones_like(pasted),pasted)[:,8:25,8:25]==255).all()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        image = root/'leftImg8bit/train/demo/demo_000000_000000_leftImg8bit.png'
        image.parent.mkdir(parents=True)
        Image.new('RGB',(64,32)).save(image)
        listing = root/'train.txt'; listing.write_text('demo/'+image.name+'\n')
        args = SimpleNamespace(city_root=root,city_train_list=listing,city_config=ROOT/'configs/deeplabv2_r101_SND_G2C.yaml',output=root/'cache',data_limit=0,smoke=True)
        cfg=config(args)
        original=Image.open
        def guarded(path,*a,**kw):
            assert 'gtFine' not in str(path)
            return original(path,*a,**kw)
        with patch.object(Image,'open',guarded):
            _, labels, _ = CandidateImages(args,cfg,True)[0]
        assert (labels==255).all()
    print('SND G2C aux routing/source-free checks passed')


if __name__ == '__main__':
    test_gradient_routing_and_source_free_candidates()

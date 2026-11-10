"""Wider/cross-layer context and closed-form, bounded residual readouts."""
import torch
from torch.nn import functional as F


def context(probability,q,bandwidth,radius,low=None,low_bandwidth=None):
    h,w=q.shape[-2:]; raw=probability.T.reshape(-1,h,w)
    pp=F.pad(raw[None],(radius,)*4,mode='replicate')[0]
    qp=F.pad(q[None],(radius,)*4,mode='replicate')[0] if bandwidth is not None else None
    lp=F.pad(low[None],(radius,)*4,mode='replicate')[0] if low is not None else None
    numerator=torch.zeros_like(raw); denominator=torch.zeros(h,w)
    for y in range(2*radius+1):
        for x in range(2*radius+1):
            distance=torch.zeros(h,w)
            if qp is not None: distance+=(qp[:,y:y+h,x:x+w]-q).square().sum(0)/bandwidth
            if lp is not None:
                distance+=(lp[:,y:y+h,x:x+w]-low).square().sum(0)/low_bandwidth
                if qp is not None: distance*=.5
            weight=(-distance).exp()
            numerator+=weight[None]*pp[:,y:y+h,x:x+w]; denominator+=weight
    return .5*probability+.5*(numerator/denominator).flatten(1).T


def design(blocks,statistics=None):
    if statistics is None:
        statistics=[]
        for block in blocks:
            mean=block.double().mean(0)
            scale=max(float((block.double()-mean).square().mean().sqrt()),1e-6)
            statistics.append((mean,scale))
    columns=[(block.double()-mean)/scale for block,(mean,scale) in zip(blocks,statistics)]
    x=torch.cat(columns+[torch.ones(len(blocks[0]),1,dtype=torch.float64)],1)
    return x,statistics


def ridge(blocks,target,weights,cal_blocks,cal_target,cal_weights):
    x,statistics=design(blocks); cx,_=design(cal_blocks,statistics)
    weights=weights.double()/weights.sum(); cal_weights=cal_weights.double()/cal_weights.sum()
    target=target.double(); target-=target.mean(1,keepdim=True)
    cal_target=cal_target.double(); cal_target-=cal_target.mean(1,keepdim=True)
    gram=x.T@(x*weights[:,None]); rhs=x.T@(target*weights[:,None])
    options=[]
    for strength in (.01,.1,1.):
        system=gram+strength*torch.eye(x.shape[1],dtype=torch.float64)
        beta=torch.linalg.solve(system,rhs); beta-=beta.mean(1,keepdim=True)
        error=float(((cx@beta-cal_target).square().mean(1)*cal_weights).sum())
        stationarity=float((system@beta-rhs).abs().max())
        assert stationarity<1e-7 and torch.isfinite(beta).all()
        options.append((error,strength,beta,stationarity))
    error,strength,beta,residual=min(options,key=lambda v:v[0])
    return {'statistics':statistics,'beta':beta}, {'ridge':strength,'cal_teacher_mse':error,
        'stationarity_residual':residual,'parameters_stored':beta.numel(),
        'ridge_candidates':{str(s):e for e,s,_,_ in options},'selection':'unlabeled calibration soft-teacher MSE'}


def predict_readout(blocks,model,alpha=.5):
    x,_=design(blocks,model['statistics']); delta=(x@model['beta']).float()
    delta-=delta.mean(1,keepdim=True)
    delta*=torch.minimum(torch.ones(len(delta),1),2./delta.abs().amax(1,keepdim=True).clamp_min(1e-12))
    return alpha*delta


def selfcheck():
    from scripts.training_free_local_subspace import refine
    torch.manual_seed(83); q=F.normalize(torch.randn(12,5,7),dim=0); p=torch.randn(35,4).softmax(1)
    assert torch.allclose(context(p,q,.3,1),refine(q,p,{'kind':'full'},.3),atol=2e-7)
    u=context(p,q,None,4)
    expected=.5*p+.5*F.avg_pool2d(F.pad(p.T.reshape(1,4,5,7),(4,)*4,mode='replicate'),9,stride=1)[0].flatten(1).T
    assert torch.allclose(u,expected,atol=2e-7)
    low=q[:3]; value=context(p,q,.3,4,low,.2)
    assert (value>=0).all() and torch.allclose(value.sum(1),torch.ones(35),atol=1e-6)
    x=torch.randn(100,5); target=torch.randn(100,4)*.1; target-=target.mean(1,keepdim=True)
    weights=torch.ones(100); model,audit=ridge([x],target,weights,[x],target,weights)
    rotation=torch.linalg.qr(torch.randn(5,5))[0]
    rotated,_=ridge([x@rotation],target,weights,[x@rotation],target,weights)
    assert torch.allclose(predict_readout([x],model),predict_readout([x@rotation],rotated),atol=1e-6)
    zero,_=ridge([x],torch.zeros_like(target),weights,[x],torch.zeros_like(target),weights)
    assert torch.equal(predict_readout([x],zero),torch.zeros_like(target))
    delta=predict_readout([x],model); assert delta.abs().max()<=1. and delta.sum(1).abs().max()<1e-6
    raw=torch.randn(100,4)*10; margin=raw.topk(2,dim=1).values.diff(dim=1).abs().flatten()
    assert torch.equal(raw.argmax(1)[margin>2],(raw+delta).argmax(1)[margin>2])
    print('PASS: context3 parity/context9 pooling, cross-layer simplex, ridge stationarity/rotation/identity and bounded-margin protection',flush=True)

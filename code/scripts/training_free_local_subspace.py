"""Local posterior averaging with frozen semantic/nuisance distance metrics."""
from __future__ import annotations

import torch
from torch.nn import functional as F


OFFSETS=tuple((y,x) for y in (-1,0,1) for x in (-1,0,1))


def neighbor_vectors(grid, sites):
    d,h,w=grid.shape; padded=F.pad(grid[None],(1,1,1,1),mode='replicate')[0]
    return torch.stack([padded[:,1+y:1+y+h,1+x:1+x+w].flatten(1).T[sites] for y,x in OFFSETS if (y,x)!=(0,0)],1)


def coordinates(z, metric):
    if metric['kind']=='project': return z@metric['basis']
    if metric['kind']=='suppress':
        u=metric['basis']; return z-.5*(z@u)@u.T
    return z


def refine(grid, conditional, metric, bandwidth):
    h,w=grid.shape[-2:]
    flat=grid.flatten(1).T
    q=coordinates(flat,metric).T.reshape(-1,h,w)
    padded=F.pad(q[None],(1,1,1,1),mode='replicate')[0]
    probability=conditional.T.reshape(-1,h,w)
    padded_p=F.pad(probability[None],(1,1,1,1),mode='replicate')[0]
    numerator=torch.zeros_like(probability); denominator=torch.zeros(h,w)
    for y,x in OFFSETS:
        neighbor=padded[:,1+y:1+y+h,1+x:1+x+w]
        if metric['kind']=='uniform': weight=torch.ones(h,w)
        else: weight=torch.exp(-(neighbor-q).square().sum(0)/bandwidth)
        numerator+=weight[None]*padded_p[:,1+y:1+y+h,1+x:1+x+w]; denominator+=weight
    value=(numerator/denominator).flatten(1).T
    return .5*conditional+.5*value


def fit_nuisance(response, tile_ids):
    unique=torch.unique(tile_ids)
    if not len(response) or float(response.square().mean())<1e-10:
        return response.new_empty(response.shape[1],0),{'active':False,'rank':0,'reason':'zero response'}
    _,singular,vectors=torch.linalg.svd(response.double(),full_matrices=False)
    rank=min(8,int((singular>max(float(singular[0])*1e-6,1e-8)).sum()))
    bases=[]
    for tiles in (unique[::2],unique[1::2]):
        selected=response[torch.isin(tile_ids,tiles)]
        if len(selected)<rank: raise ValueError('Insufficient nuisance half support')
        _,_,half_vectors=torch.linalg.svd(selected.double(),full_matrices=False); bases.append(half_vectors[:rank].T)
    overlap=float(torch.linalg.svdvals(bases[0].T@bases[1]).square().mean())
    active=overlap>=.5
    basis=vectors[:rank].T.float() if active else response.new_empty(response.shape[1],0)
    return basis,{'active':active,'rank':basis.shape[1],'requested_rank':rank,'tile_half_projection_overlap':overlap,
        'basis_fitted_from':'all_fit_tiles; halves only test stability',
        'response_energy':float(response.square().mean()),'explained_response_energy':float(singular[:rank].square().sum()/singular.square().sum()),
        'reason':'stable paired response' if active else 'half overlap below0.5; identity fallback'}


def selfcheck():
    from scripts.training_free_graph import replace_conditional,semantic_basis
    torch.manual_seed(8); grid=F.normalize(torch.randn(12,5,7),dim=0)
    probabilities=torch.randn(35,3).softmax(1)
    uniform=refine(grid,probabilities,{'kind':'uniform'},1.)
    raw=probabilities.T.reshape(1,3,5,7)
    expected=.5*probabilities+.5*F.avg_pool2d(F.pad(raw,(1,1,1,1),mode='replicate'),3,stride=1)[0].flatten(1).T
    assert torch.allclose(uniform,expected,atol=1e-7)
    metric={'kind':'suppress','basis':torch.empty(12,0)}
    assert torch.equal(coordinates(grid.flatten(1).T,metric),grid.flatten(1).T)
    basis=semantic_basis(torch.randn(3,12)); metric={'kind':'project','basis':basis}
    rotated={'kind':'project','basis':basis@torch.linalg.qr(torch.randn(basis.shape[1],basis.shape[1]))[0]}
    value=refine(grid,probabilities,metric,.2)
    assert torch.allclose(value,refine(grid,probabilities,rotated,.2),atol=2e-7)
    assert (value>=0).all() and torch.allclose(value.sum(1),torch.ones(35),atol=1e-6)
    delta=neighbor_vectors(grid,torch.arange(35))-grid.flatten(1).T[:,None]
    assert delta.shape==(35,8,12)
    source=torch.randn(35,4).log_softmax(1); result=replace_conditional(source,value,[0,1,3])
    assert torch.equal(source[:,2],result[:,2]) and torch.allclose(result.exp().sum(1),torch.ones(35),atol=1e-6)
    lowrank=torch.randn(100,2)@torch.randn(2,12); tiles=torch.arange(100)%10
    nuisance,audit=fit_nuisance(lowrank,tiles)
    assert nuisance.shape[1]==2 and audit['active'] and audit['tile_half_projection_overlap']>.99
    mixed=lowrank+.2*torch.randn_like(lowrank)
    pooled,audit=fit_nuisance(mixed,tiles)
    _,_,expected_vectors=torch.linalg.svd(mixed.double(),full_matrices=False)
    expected_basis=expected_vectors[:8].T.float()
    assert audit['active'] and torch.allclose(pooled@pooled.T,expected_basis@expected_basis.T,atol=2e-6)
    zero,audit=fit_nuisance(torch.zeros(100,12),tiles)
    assert zero.shape[1]==0 and not audit['active']
    print('PASS: local uniform parity, simplex/unsupported mass, basis rotation, zero-rank identity, pooled nuisance basis and fallback',flush=True)

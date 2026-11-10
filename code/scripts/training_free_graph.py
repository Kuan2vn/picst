"""Small source-anchored bank graph; CPU only, no optimizer or dense pixel graph."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def semantic_basis(means, head_rows=None):
    rows = [means.double() - means.double().mean(0)]
    if head_rows is not None:
        rows.append(head_rows.double() - head_rows.double().mean(0))
    _, singular, vectors = torch.linalg.svd(torch.cat(rows), full_matrices=False)
    rank = int((singular > singular[0] * 1e-6).sum())
    if rank < 1:
        raise ValueError('No semantic contrast directions')
    return vectors[:rank].T.float()


def projected_prototype_scores(z, model, basis):
    # Original norm offsets preserve class odds; refitting projected means would not.
    means = model['means']
    return ((z @ basis) @ (means @ basis).T - .5 * means.square().sum(1)) / model['floor']


def neighbors(query, bank, k=5, query_ids=None, bank_ids=None, chunk=256):
    if k < 1 or len(bank) <= k or chunk < 1:
        raise ValueError('Insufficient bank/invalid neighbor parameters')
    if not torch.isfinite(query).all() or not torch.isfinite(bank).all():
        raise FloatingPointError('Non-finite retrieval coordinates')
    indices, distances = [], []
    norm = bank.square().sum(1)
    for start in range(0, len(query), chunk):
        q = query[start:start+chunk]
        distance = (q.square().sum(1)[:, None] + norm - 2 * q @ bank.T).clamp_min(0)
        if query_ids is not None:
            distance.masked_fill_(query_ids[start:start+chunk, None] == bank_ids[None], float('inf'))
        values, ids = distance.topk(k, largest=False)
        if not torch.isfinite(values).all():
            raise ValueError('Insufficient cross-image neighbors')
        indices.append(ids); distances.append(values)
    return torch.cat(indices), torch.cat(distances)


def anchored_graph(coordinates, anchors, image_ids, tile_ids, k=5):
    ids, distance = neighbors(coordinates, coordinates, k, image_ids, image_ids)
    scale = max(float(distance[:, -1].median()), 1e-8)
    directed = torch.zeros(len(coordinates), len(coordinates))
    directed.scatter_(1, ids, torch.exp(-distance/scale))
    weight = torch.where(directed.T > 0, directed, 0)
    weight = (weight + weight.T)/2
    degree = weight.sum(1)
    if (degree > 0).any():
        weight /= degree[degree > 0].median()
    weight = weight.double()
    degree = weight.sum(1)
    edge = weight.nonzero().T
    sparse = torch.sparse_coo_tensor(edge, weight[edge[0], edge[1]], weight.shape).coalesce()
    value = anchors.double(); a = value.clone()
    for iteration in range(1, 501):
        value = (a + torch.sparse.mm(sparse, value))/(1 + degree[:, None])
        residual = (value*(1+degree[:, None]) - torch.sparse.mm(sparse, value) - a).abs().max()
        if residual < 1e-8:
            break
    else:
        raise RuntimeError('Anchored graph failed residual tolerance')
    assert value.min() >= 0 and torch.allclose(value.sum(1), torch.ones(len(value), dtype=torch.float64), atol=1e-7)
    counts = torch.bincount(ids.flatten(), minlength=len(coordinates)).float()
    audit = {'nodes':len(value), 'undirected_edges':int(edge.shape[1]//2),
        'isolated_fraction':float((degree == 0).float().mean()), 'iterations':iteration,
        'stationarity_residual':float(residual), 'bandwidth_squared':scale,
        'cross_image_edge_fraction':float((image_ids[edge[0]] != image_ids[edge[1]]).float().mean()) if edge.shape[1] else None,
        'cross_tile_edge_fraction':float((tile_ids[edge[0]] != tile_ids[edge[1]]).float().mean()) if edge.shape[1] else None,
        'neighbor_max_indegree':int(counts.max()), 'neighbor_indegree_cv':float(counts.std()/counts.mean())}
    return value.float(), scale, audit


def retrieve(query, bank, probabilities, scale, radius, fallback):
    ids, distance = neighbors(query, bank)
    weights = (-distance/scale).softmax(1)
    value = (probabilities[ids]*weights[..., None]).sum(1)
    covered = distance[:, 0] <= radius
    value = torch.where(covered[:, None], .5*fallback + .5*value, fallback)
    return value, covered, ids


def replace_conditional(log_probabilities, conditional, classes):
    result = log_probabilities.clone()
    mass = log_probabilities[:, classes].logsumexp(1, keepdim=True)
    result[:, classes] = conditional.clamp_min(torch.finfo(conditional.dtype).tiny).log() + mass
    return result


def selfcheck():
    from scripts.training_free_subspace import blend_log_probabilities, scores
    torch.manual_seed(13)
    z = F.normalize(torch.randn(31, 12), dim=1)
    means = torch.randn(3,12)*.1
    model = {'active':True,'means':means,'classes':[0,2,3],'floor':.05,
        'bases':[torch.empty(12,0)]*3, 'amplitudes':[torch.empty(0)]*3}
    basis = semantic_basis(means, torch.randn(4,12))
    old = scores(z,model,'B1').log_softmax(1)
    new = projected_prototype_scores(z,model,basis).log_softmax(1)
    assert torch.allclose(old,new,atol=2e-5)
    image = torch.arange(31)//2
    ids,distance = neighbors(z,z,5,image,image,chunk=3)
    ids2,distance2 = neighbors(z,z,5,image,image,chunk=31)
    assert torch.equal(ids,ids2) and torch.allclose(distance,distance2,atol=1e-6)
    assert (image[:,None] != image[ids]).all()
    anchors = torch.randn(31,3).softmax(1)
    value,scale,audit = anchored_graph(z,anchors,image,image)
    adjacency = torch.zeros(31,31); adjacency.scatter_(1,ids,torch.exp(-distance/scale))
    adjacency = torch.where(adjacency.T > 0,adjacency,0); adjacency=(adjacency+adjacency.T)/2
    degree=adjacency.sum(1); adjacency/=degree[degree>0].median(); degree=adjacency.sum(1)
    exact=torch.linalg.solve(torch.eye(31).double()+torch.diag(degree).double()-adjacency.double(),anchors.double())
    assert torch.allclose(value.double(),exact,atol=2e-7)
    assert torch.equal(value[degree==0],anchors[degree==0])
    isolated_coordinates=torch.cat([torch.randn(6,2)*.001,torch.full((1,2),100.)])
    isolated_anchors=torch.randn(7,3).softmax(1); distinct=torch.arange(7)
    isolated,_,isolated_audit=anchored_graph(isolated_coordinates,isolated_anchors,distinct,distinct)
    assert isolated_audit['isolated_fraction']>0 and torch.equal(isolated[-1],isolated_anchors[-1])
    logits=torch.randn(31,4); b1=blend_log_probabilities(logits,scores(z,model,'B1'),model['classes'],1.)
    conditional=b1[:,model['classes']].softmax(1)
    fallback,covered,_=retrieve(z,z,value,scale,-1.,conditional)
    assert torch.equal(fallback,conditional) and not covered.any()
    result=replace_conditional(b1,fallback,model['classes'])
    assert torch.equal(result[:,1],b1[:,1]) and torch.allclose(result.exp().sum(1),torch.ones(31),atol=1e-6)
    assert audit['stationarity_residual']<1e-8
    print('PASS: prototype odds, chunked cross-image retrieval, exact graph solve, isolation/fallback, unsupported mass',flush=True)

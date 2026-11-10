"""Target-only semantic seed memory. Frozen DINO is optional; student remains MiT-B2."""
import hashlib
import math
from pathlib import Path
import torch
import torch.nn.functional as F

NAME = 'dtst_semantic_recovery_20261016_v1'


def validate_recovery(config):
    if config.get('name') != NAME or config.get('features') not in ('ema', 'dinov2'):
        raise ValueError('Unknown semantic recovery')
    for key in ('seed_confidence', 'vote_confidence', 'min_cosine', 'correction_weight', 'max_native_confidence'):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('Invalid recovery setting: '+key)
    for key in ('grid', 'memory_per_class', 'seeds_per_image_class', 'min_seeds', 'neighbors', 'required_votes', 'max_age'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError('Invalid recovery count: '+key)
    if config['required_votes'] > config['neighbors'] or config['min_seeds'] > config['memory_per_class']:
        raise ValueError('Invalid seed/vote support')
    if config['grid'] != 32:
        raise ValueError('These pilots use32x32 feature grids')
    if config['features'] == 'dinov2':
        if not config.get('model_path') or len(config.get('model_sha256','')) != 64:
            raise ValueError('DINO requires pinned local weights')
    return config


def name_id(name):
    return int.from_bytes(hashlib.sha256(str(name).encode()).digest()[:7], 'big')


class SeedMemory:
    def __init__(self, config):
        self.config = validate_recovery(config)
        self.features = self.labels = self.image_ids = self.steps = None

    def state_dict(self):
        return {key: None if getattr(self,key) is None else getattr(self,key).cpu()
                for key in ('features','labels','image_ids','steps')}

    def load_state_dict(self, state, device):
        for key in ('features','labels','image_ids','steps'):
            value = state[key]
            setattr(self,key,None if value is None else value.to(device))
        if self.features is not None:
            if not torch.isfinite(self.features).all() or not (self.labels.ge(0)&self.labels.lt(6)).all():
                raise ValueError('Invalid recovery memory state')

    @torch.no_grad()
    def query(self, features, image_ids, step):
        count=features.shape[0];device=features.device
        labels=torch.full((count,),255,device=device,dtype=torch.long)
        accepted=torch.zeros(count,device=device,dtype=torch.bool)
        if self.features is None:return labels,accepted
        live=(step-self.steps <= self.config['max_age'])
        counts=torch.bincount(self.labels[live],minlength=6)
        live &= counts[self.labels] >= self.config['min_seeds']
        if int(live.sum()) < self.config['neighbors']:return labels,accepted
        keys=self.features[live];classes=self.labels[live];ids=self.image_ids[live]
        # Chunking bounds dense-query memory without changing the neighbor search.
        for start in range(0,count,2048):
            stop=min(start+2048,count)
            similarity=features[start:stop]@keys.T
            similarity.masked_fill_(image_ids[start:stop,None] == ids[None],-2.)
            chosen_values=[];chosen_indices=[];chosen_ids=[]
            # At most two tokens per image: repeated pixels are not independent semantic votes.
            for _ in range(self.config['neighbors']):
                value,idx=similarity.max(1)
                chosen_values.append(value);chosen_indices.append(idx);chosen_ids.append(ids[idx])
                hits=(torch.stack(chosen_ids,1)==ids[idx,None]).sum(1)
                similarity.scatter_(1,idx[:,None],-2.)
                similarity.masked_fill_((hits>=2)[:,None] & (ids[None]==ids[idx,None]),-2.)
            values=torch.stack(chosen_values,1);index=torch.stack(chosen_indices,1)
            nearby=classes[index];near_ids=ids[index]
            weights=((values-values[:,0:1])/.1).exp()
            vote=torch.zeros(stop-start,6,device=device).scatter_add_(1,nearby,weights)
            vote=vote/vote.sum(1,keepdim=True).clamp_min(1e-8)
            confidence,winner=vote.max(1)
            supporters=nearby==winner[:,None]
            votes=supporters.sum(1)
            mean_cosine=(values*supporters).sum(1)/votes.clamp_min(1)
            low=near_ids.masked_fill(~supporters,torch.iinfo(torch.long).max).min(1).values
            high=near_ids.masked_fill(~supporters,-1).max(1).values
            good=((confidence>=self.config['vote_confidence']) &
                  (votes>=self.config['required_votes']) &
                  (mean_cosine>=self.config['min_cosine']) & (low!=high) & (values[:,-1]>-1))
            labels[start:stop]=winner;accepted[start:stop]=good
        return labels,accepted

    @torch.no_grad()
    def add(self, features, probability, image_ids, step):
        confidence,labels=probability.max(1)
        all_features=[];all_labels=[];all_ids=[];all_steps=[]
        for cls in range(6):
            take=[]
            for image in image_ids.unique():
                idx=torch.where((image_ids==image)&(labels==cls)&(confidence>=self.config['seed_confidence']))[0]
                if idx.numel():
                    positions=torch.linspace(0,idx.numel()-1,min(idx.numel(),self.config['seeds_per_image_class']),
                                             device=idx.device).round().long()
                    take.append(idx[positions])
            idx=torch.cat(take) if take else torch.empty(0,device=features.device,dtype=torch.long)
            chunks=[features[idx]];id_chunks=[image_ids[idx]]
            time_chunks=[torch.full((len(idx),),step,device=features.device,dtype=torch.long)]
            if self.features is not None:
                old=(self.labels==cls)&(step-self.steps<=self.config['max_age'])
                # Replace this batch's old seed copies rather than count repeated images as new evidence.
                old &= ~torch.isin(self.image_ids,image_ids.unique())
                chunks.insert(0,self.features[old]);id_chunks.insert(0,self.image_ids[old]);time_chunks.insert(0,self.steps[old])
            values=torch.cat(chunks)[-self.config['memory_per_class']:]
            ids=torch.cat(id_chunks)[-self.config['memory_per_class']:]
            times=torch.cat(time_chunks)[-self.config['memory_per_class']:]
            all_features.append(values);all_ids.append(ids);all_steps.append(times)
            all_labels.append(torch.full((len(values),),cls,device=features.device,dtype=torch.long))
        self.features=torch.cat(all_features).detach()
        self.labels=torch.cat(all_labels);self.image_ids=torch.cat(all_ids);self.steps=torch.cat(all_steps)


class SemanticRecovery:
    def __init__(self, runtime, config):
        self.runtime=runtime;self.config=validate_recovery(config)
        self.memory=SeedMemory(config);self.last_record={};self.forward_images=0;self.cache_hits=0
        self.model=None;self.corrected=None;self.protected=None
        if config['features']=='dinov2':
            from transformers import AutoModel
            from train_SRMP_remote import rng_state,restore_rng
            path=Path(config['model_path'])
            if hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest()!=config['model_sha256']:
                raise ValueError('DINO weight hash mismatch')
            rng=rng_state()
            try:
                self.model=AutoModel.from_pretrained(str(path),local_files_only=True).to(runtime.device).eval()
                self.model.requires_grad_(False)
            finally:restore_rng(rng)
            self.cache=Path(config['cache_root'])/config['model_sha256'][:16]/runtime.source_sha[:16]
            self.cache.mkdir(parents=True,exist_ok=True)

    def state_dict(self):
        return dict(memory=self.memory.state_dict(),forward_images=self.forward_images,cache_hits=self.cache_hits)

    def load_state_dict(self,state):
        self.memory.load_state_dict(state['memory'],self.runtime.device)
        self.forward_images=state['forward_images'];self.cache_hits=state['cache_hits']

    @torch.no_grad()
    def extract(self,images,teacher_features,names):
        grid=self.config['grid']
        if self.model is None:
            features=torch.cat([F.adaptive_avg_pool2d(f,(grid,grid)) for f in teacher_features[-2:]],1)
        else:
            rows=[]
            for index,name in enumerate(names):
                cache=self.cache/(hashlib.sha256(str(name).encode()).hexdigest()+'.pth')
                if cache.exists():
                    feat=torch.load(cache,map_location=self.runtime.device,weights_only=True).float()
                    self.cache_hits+=1
                else:
                    # Keep all IRRG planes and full-image extent; only frozen feature input is resized.
                    x=F.interpolate(images[index:index+1],size=(448,448),mode='bilinear',align_corners=False)
                    out=self.model(pixel_values=x).last_hidden_state[:,1:]
                    if out.shape[1]!=grid*grid:raise ValueError('Unexpected DINO token grid')
                    feat=out.transpose(1,2).reshape(1,-1,grid,grid).float()
                    temp=cache.with_suffix('.tmp')
                    torch.save(feat.cpu().half(),temp);temp.replace(cache)
                    feat=feat.half().float()  # Same numerical path on the first and cached uses.
                    self.forward_images+=1
                if tuple(feat.shape)!=(1,384,grid,grid) or not torch.isfinite(feat).all():
                    raise ValueError('Invalid DINO feature cache')
                rows.append(feat)
            features=torch.cat(rows)
        return F.normalize(features.float(),dim=1)

    @torch.no_grad()
    def prepare(self,images,teacher_features,native_logits,names,params,step):
        features=self.extract(images,teacher_features,names)
        n,channels,h,w=features.shape
        flat=features.permute(0,2,3,1).reshape(-1,channels)
        probability=F.adaptive_avg_pool2d(native_logits.softmax(1),(h,w)).permute(0,2,3,1).reshape(-1,6)
        ids=torch.tensor([name_id(name) for name in names],device=flat.device,dtype=torch.long)
        ids=ids[:,None].expand(n,h*w).reshape(-1)
        labels,accepted=self.memory.query(flat,ids,step)
        self.memory.add(flat,probability,ids,step)
        full_size=native_logits.shape[-2:]
        labels=F.interpolate(labels.reshape(n,1,h,w).float(),size=full_size,mode='nearest')
        mask=F.interpolate(accepted.reshape(n,1,h,w).float(),size=full_size,mode='nearest')
        self.candidate=self.runtime.full2weak(labels,params,nearest=True)[:,0].long()
        self.accepted=self.runtime.full2weak(mask,params,nearest=True)[:,0]>.5
        self.last_record=dict(recovery_features=self.config['features'],
            recovery_memory_by_class=torch.bincount(self.memory.labels,minlength=6).cpu().tolist(),
            recovery_query_grid_fraction=float(accepted.float().mean()),
            recovery_feature_forward_images=self.forward_images,recovery_feature_cache_hits=self.cache_hits)

    @torch.no_grad()
    def apply(self,labels,pasted,teacher_probability,valid):
        original=labels.clone()
        mask=(self.accepted & valid.bool() & (pasted==255) & (self.candidate!=labels) &
              (teacher_probability.max(1).values <= self.config['max_native_confidence']))
        labels=torch.where(mask,self.candidate,labels)
        self.corrected=mask
        self.protected=torch.where(mask,labels,torch.full_like(labels,255))
        self.last_record.update(
            recovery_changed_fraction=float(mask.float().mean()),
            recovery_changed_by_class=torch.bincount(labels[mask],minlength=6).cpu().tolist(),
            recovery_filled_ignore_by_class=torch.bincount(labels[mask&(original==255)],minlength=6).cpu().tolist(),
            recovery_relabel_by_original_class=torch.bincount(original[mask&(original!=255)],minlength=6).cpu().tolist())
        return labels

    def positive_loss(self,loss_map,weights):
        coefficient=torch.where(self.corrected,self.config['correction_weight'],1.)
        return (loss_map*weights*coefficient).mean()

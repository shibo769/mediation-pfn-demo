"""Shared-scale error experts and genuinely continuous CDF mixture routing.

Uses a v9 training cache; no sealed test rows are accepted while fitting.
All three experts model Z=(consensus EIF estimate - NIE)/consensus EIF SE.
Offline truth is used to train and select; deployment takes observables only.
"""
from pathlib import Path
import argparse, copy, hashlib, json, math, os, sys, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .posttrain import (Cache, ErrorDensity, logpdf, crps, quantile, grouped_indices,
                       group_risk, sha)

class Router(nn.Module):
    def __init__(self,dim,width=64):
        super().__init__(); self.dim=dim;self.width=width
        self.net=nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.LayerNorm(width),nn.Linear(width,3))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def concentration(self,x):return .5+F.softplus(self.net(x))
    def forward(self,x):
        a=self.concentration(x);return a/a.sum(-1,keepdim=True)

def mixture(parts,weights,top2=False):
    if top2:
        ix=weights.topk(2,dim=-1).indices
        mask=torch.zeros_like(weights).scatter_(-1,ix,1.)
        weights=weights*mask;weights=weights/weights.sum(-1,keepdim=True)
    w=torch.cat([weights[:,i,None]*p[0] for i,p in enumerate(parts)],-1)
    return w,torch.cat([p[1] for p in parts],-1),torch.cat([p[2] for p in parts],-1)

def parts_at(heads,x):return [head(x) for head in heads]
def dev_mask(cache,role):
    m=cache.mask(role,'dev')&cache.good
    # A role-labelled new DEV bank is not pooled with the other model selector.
    if np.any(cache.bank[m] != 'all'):
        m &= cache.bank==role
        if not m.any():raise ValueError('Missing DEV bank '+role)
    return m
def arrays(cache,device):
    a=cache.a;mask=cache.mask('head')&cache.good
    mean=a['x_multi'][mask].mean(0);std=np.maximum(a['x_multi'][mask].std(0),.01)
    x=torch.as_tensor(np.nan_to_num((a['x_multi']-mean)/std),dtype=torch.float32,device=device)
    z=torch.as_tensor(np.nan_to_num((a['estimates'][:,1]-a['truth'])/a['ses'][:,1]),dtype=torch.float32,device=device)
    return x,z,mean,std

def intervals(part,estimate,se):
    lo=quantile(part,.025).cpu().numpy();hi=quantile(part,.975).cpu().numpy()
    return np.stack([estimate-se*hi,estimate-se*lo],-1)

def score(part,ids,a,groups):
    with torch.no_grad():ci=intervals(part,a['estimates'][ids,1],a['ses'][ids,1])
    index={int(v):i for i,v in enumerate(ids)}
    risk=[]
    for group in groups:
        ii=np.array([index[int(v)] for v in group])
        risk.append(group_risk(ci[ii,0],ci[ii,1],a['truth'][group],a['ses'][group,1]))
    return float(np.mean(risk))

def fit_heads(cache,x,z,args,logs):
    a=cache.a; train=cache.mask('head')&cache.good;dev=dev_mask(cache,'head')
    all_head_groups=grouped_indices(a['mechanism_id'],train,args.min_group,args.max_group)
    if len(all_head_groups)<4:raise ValueError('At least four independent head mechanisms are required')
    permutation=np.random.default_rng(args.seed).permutation(len(all_head_groups))
    all_head_groups=[all_head_groups[i] for i in permutation]
    boundary=len(all_head_groups)//2
    global_groups=all_head_groups[:boundary];specialist_groups=all_head_groups[boundary:]
    train_ids=np.concatenate(global_groups);dev_ids=np.flatnonzero(dev)
    groups=grouped_indices(a['mechanism_id'],dev,args.min_group,args.max_group)
    model=ErrorDensity(x.shape[1],args.width).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    best=math.inf;state=None;epoch_selected=0
    for ep in range(args.head_epochs+1):
        if ep:
            for ii in torch.as_tensor(train_ids,device=args.device)[torch.randperm(len(train_ids),device=args.device)].split(args.batch):
                p=model(x[ii]);loss=(-logpdf(p,z[ii])+.2*crps(p,z[ii])).mean();opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():risk=score(model(x[dev_ids]),dev_ids,a,groups)
        logs.append(dict(stage='global',epoch=ep,dev_risk=risk))
        if risk<best:best=risk;state=copy.deepcopy(model.state_dict());epoch_selected=ep
    if state is None:raise ValueError('No finite global head')
    model.load_state_dict(state);model.eval()
    # Classification uses only training mechanisms and their synthetic truth.
    # Under/over groups select a tail-oriented bias without reading DEV/test.
    train_groups=specialist_groups
    # Split repeated tables within each mechanism: label tables and specialist
    # fitting tables are disjoint, and the split is fixed by cache order.
    label_groups=[g[:len(g)//2] for g in train_groups]
    fit_groups=[g[len(g)//2:] for g in train_groups]
    gaps=[]
    with torch.no_grad():
        for group in label_groups:
            ci=intervals(model(x[group]),a['estimates'][group,1],a['ses'][group,1])
            gaps.append(float(((ci[:,0]<=a['truth'][group])&(a['truth'][group]<=ci[:,1])).mean()))
    # Rank rather than absolute threshold so both types are represented even for
    # tiny simulation banks; the rule and group membership are logged.
    ordered=np.argsort(gaps,kind='stable');half=max(1,len(ordered)//2)
    memberships=[np.concatenate([fit_groups[i] for i in ordered[:half]]),
                 np.concatenate([fit_groups[i] for i in ordered[-half:]])]
    heads=[model];scales=(1.25,.8)
    for name,subset,scale in zip(('under','over'),memberships,scales):
        specialist=copy.deepcopy(model);opt=torch.optim.AdamW(specialist.parameters(),lr=args.lr/2)
        best=math.inf;state=None;chosen=0
        for ep in range(args.expert_epochs+1):
            if ep:
                ii=torch.as_tensor(subset,device=args.device)
                for b in ii[torch.randperm(len(ii),device=args.device)].split(args.batch):
                    p=specialist(x[b]);target=z[b]/scale
                    loss=(-logpdf(p,target)+.2*crps(p,target)).mean()
                    # An L2 anchor regularizes against catastrophic departure.
                    anchor=sum((v-model.state_dict()[k]).square().mean() for k,v in specialist.state_dict(keep_vars=True).items())
                    loss=loss+.001*anchor;opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                w,m,s=specialist(x[dev_ids]);risk=score((w,m*scale,s*scale),dev_ids,a,groups)
            logs.append(dict(stage=name,epoch=ep,dev_risk=risk,scale=scale))
            if risk<best:best=risk;state=copy.deepcopy(specialist.state_dict());chosen=ep
        if state is None:raise ValueError('No finite specialist')
        specialist.load_state_dict(state);heads.append(Scaled(specialist,scale))
    for head in heads:
        head.eval()
        for p in head.parameters():p.requires_grad_(False)
    logs.append(dict(stage='specialist_groups',global_epoch=epoch_selected,coverage=gaps,under_group_indices=ordered[:half].tolist(),over_group_indices=ordered[-half:].tolist(),label_rows=[g.tolist() for g in label_groups],fit_rows=[g.tolist() for g in fit_groups]))
    return heads

class Scaled(nn.Module):
    def __init__(self,base,scale):super().__init__();self.base=base;self.scale=scale
    def forward(self,x):
        w,m,s=self.base(x);return w,m*self.scale,s*self.scale

def fit_sft(router,heads,cache,x,z,args,logs,roles,arm,initial=None,eligible_dpo=None):
    if initial is not None:router.load_state_dict(initial)
    a=cache.a;dev=dev_mask(cache,'sft');ids=np.flatnonzero(dev)
    groups=grouped_indices(a['mechanism_id'],dev,args.min_group,args.max_group)
    opt=torch.optim.AdamW(router.parameters(),lr=args.lr);best=math.inf;state=None
    stages=[role for role,epochs in roles for _ in range(epochs)];updates=0
    for ep in range(len(stages)+1):
        if ep:
            role=stages[ep-1];trainmask=cache.mask(role)&cache.good
            if arm=='matched_sft':
                groups_for_role=grouped_indices(a['mechanism_id'],trainmask,args.min_group,args.max_group)
                batches=[torch.as_tensor(groups_for_role[i],device=args.device) for i in np.random.permutation(len(groups_for_role)) if role!='dpo' or i in eligible_dpo]
            else:
                train=np.flatnonzero(trainmask)
                batches=torch.as_tensor(train,device=args.device)[torch.randperm(len(train),device=args.device)].split(args.batch)
            for b in batches:
                with torch.no_grad():parts=parts_at(heads,x[b])
                p=mixture(parts,router(x[b]),args.top2);loss=(-logpdf(p,z[b])+.2*crps(p,z[b])).mean()
                opt.zero_grad();loss.backward();opt.step();updates+=1
        with torch.no_grad():risk=score(mixture(parts_at(heads,x[ids]),router(x[ids]),args.top2),ids,a,groups)
        logs.append(dict(stage=arm,epoch=ep,dev_risk=risk,updates=updates))
        if risk<best:best=risk;state=copy.deepcopy(router.state_dict())
    if state is None:raise ValueError('No finite SFT router')
    router.load_state_dict(state);return router

def candidate_rewards(parts,weights,ids,a,top2):
    rewards=[]
    with torch.no_grad():
        for w in weights:
            ci=intervals(mixture(parts,w,top2),a['estimates'][ids,1],a['ses'][ids,1])
            rewards.append(-group_risk(ci[:,0],ci[:,1],a['truth'][ids],a['ses'][ids,1]))
    return torch.tensor(rewards,dtype=torch.float32,device=weights.device)

def fit_preference(initial,heads,cache,x,args,logs,mode,arm=None):
    model=copy.deepcopy(initial);reference=copy.deepcopy(initial).eval();opt=torch.optim.AdamW(model.parameters(),lr=args.lr/2)
    a=cache.a;train=cache.mask(mode)&cache.good;dev=dev_mask(cache,'sft')
    groups=grouped_indices(a['mechanism_id'],train,args.min_group,args.max_group)
    devgroups=grouped_indices(a['mechanism_id'],dev,args.min_group,args.max_group);dev_ids=np.flatnonzero(dev)
    # DPO pairs are sampled once from the frozen reference and never regenerated.
    pairs={}
    if mode=='dpo':
        for gi,ids in enumerate(groups):
            with torch.no_grad():
                actions=torch.distributions.Dirichlet(reference.concentration(x[ids])).sample((args.draws,))
                reward=candidate_rewards(parts_at(heads,x[ids]),actions,ids,a,args.top2)
            if torch.isfinite(reward).all() and float(reward.max()-reward.min())>1e-8:
                pairs[gi]=(actions[reward.argmax()].detach(),actions[reward.argmin()].detach())
    best=math.inf;state=None;steps=0
    for ep in range(args.preference_epochs+1):
        if ep:
            for gi in np.random.permutation(len(groups)):
                ids=groups[gi];dist=torch.distributions.Dirichlet(model.concentration(x[ids]));ref=torch.distributions.Dirichlet(reference.concentration(x[ids]).detach())
                if mode=='dpo':
                    if gi not in pairs:continue
                    win,lose=pairs[gi]
                    delta=(dist.log_prob(win)-dist.log_prob(lose)-ref.log_prob(win)+ref.log_prob(lose)).sum()
                    loss=-F.logsigmoid(.1*delta)
                else:
                    with torch.no_grad():
                        actions=dist.sample((args.draws,));reward=candidate_rewards(parts_at(heads,x[ids]),actions,ids,a,args.top2)
                    if not torch.isfinite(reward).all():continue
                    advantage=reward-(reward.sum()-reward)/(len(reward)-1)
                    loss=-(dist.log_prob(actions).sum(-1)*advantage.detach()).mean()+.02*torch.distributions.kl_divergence(dist,ref).mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1
        with torch.no_grad():risk=score(mixture(parts_at(heads,x[dev_ids]),model(x[dev_ids]),args.top2),dev_ids,a,devgroups)
        logs.append(dict(stage=arm or mode,epoch=ep,dev_risk=risk,updates=steps,eligible_pairs=len(pairs)))
        if risk<best:best=risk;state=copy.deepcopy(model.state_dict())
    if state is None:raise ValueError('No finite preference router')
    model.load_state_dict(state);return model,set(pairs)

def expert_distributions(heads,mean,std,x_multi,theta,se,device='cpu'):
    """Observable-only expert outputs on absolute NIE scale, shape n x 3 x K."""
    x=torch.as_tensor(np.nan_to_num((np.asarray(x_multi)-mean)/std),dtype=torch.float32,device=device)
    theta=np.asarray(theta);se=np.asarray(se)
    with torch.no_grad():parts=parts_at(heads,x)
    return dict(weights=np.stack([p[0].cpu().numpy() for p in parts],1),
                means=np.stack([theta[:,None]-se[:,None]*p[1].cpu().numpy() for p in parts],1),
                scales=np.stack([se[:,None]*p[2].cpu().numpy() for p in parts],1))

def train(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    cache=Cache(args.cache);x,z,mean,std=arrays(cache,args.device);logs=[]
    start=time.perf_counter();heads=fit_heads(cache,x,z,args,logs)
    init=Router(x.shape[1],args.router_width).to(args.device)
    sft=fit_sft(init,heads,cache,x,z,args,logs,[('sft',args.router_epochs)],'sft')
    dpo,eligible=fit_preference(sft,heads,cache,x,args,logs,'dpo')
    rlvr,_=fit_preference(dpo,heads,cache,x,args,logs,'rlvr',arm='sft_dpo_rlvr')
    rlvr_only,_=fit_preference(sft,heads,cache,x,args,logs,'rlvr',arm='sft_rlvr')
    control=fit_sft(copy.deepcopy(sft),heads,cache,x,z,args,logs,[('dpo',args.preference_epochs),('rlvr',args.preference_epochs)],'matched_sft',eligible_dpo=eligible)
    result=dict(format='v10-continuous-expert-mixture',cache_sha256=sha(args.cache),mean=torch.as_tensor(mean),std=torch.as_tensor(std),
      head_width=args.width,router_width=args.router_width,dim=x.shape[1],head_states=[heads[0].state_dict(),heads[1].base.state_dict(),heads[2].base.state_dict()],
      scales=[1.,1.25,.8],router_states={k:v.state_dict() for k,v in [('sft',sft),('matched_sft',control),('dpo',dpo),('rlvr',rlvr),('rlvr_only',rlvr_only)]},
      arguments=vars(args),reward='negative group interval score / (2 EIF SE), coverage shortfall and asymmetric tails penalties; fixed v9 group_risk')
    torch.save(result,out/'mixture.pt')
    (out/'training.json').write_text(json.dumps(dict(arguments=vars(args),cache_sha256=sha(args.cache),elapsed=time.perf_counter()-start,logs=logs),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(out/'mixture.pt'),sha256=sha(out/'mixture.pt'),elapsed=time.perf_counter()-start)),flush=True)

def load(path,device='cpu'):
    c=torch.load(path,map_location=device,weights_only=True);heads=[]
    if c['format']!='v10-continuous-expert-mixture':raise ValueError('wrong format')
    for state,scale in zip(c['head_states'],c['scales']):
        base=ErrorDensity(c['dim'],c['head_width']).to(device);base.load_state_dict(state);base.eval()
        heads.append(base if scale==1. else Scaled(base,scale).to(device).eval())
    routers={}
    for name,state in c['router_states'].items():
        r=Router(c['dim'],c['router_width']).to(device);r.load_state_dict(state);routers[name]=r.eval()
    mean,std=c['mean'],c['std']
    if isinstance(mean,torch.Tensor):mean=mean.detach().cpu().numpy()
    if isinstance(std,torch.Tensor):std=std.detach().cpu().numpy()
    return heads,routers,mean,std,c

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=int(os.environ.get('NSLOTS','1')))
    p.add_argument('--width',type=int,default=128);p.add_argument('--router-width',type=int,default=64);p.add_argument('--batch',type=int,default=128);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--head-epochs',type=int,default=12);p.add_argument('--expert-epochs',type=int,default=8);p.add_argument('--router-epochs',type=int,default=8);p.add_argument('--preference-epochs',type=int,default=4)
    p.add_argument('--draws',type=int,default=4);p.add_argument('--min-group',type=int,default=16);p.add_argument('--max-group',type=int,default=32);p.add_argument('--top2',action='store_true');return p
if __name__=='__main__':
    args=parser().parse_args()
    if args.draws<2:raise ValueError('RLVR requires at least two draws')
    train(args)

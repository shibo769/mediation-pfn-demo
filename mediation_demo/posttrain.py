"""Proper error-density learning and frozen-expert categorical post-training.

Only offline training/development access truth. Deployment accepts observable
features, canonical point estimates/SEs, and optionally measured action costs.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from statistics import NormalDist
from torch import nn
from torch.nn import functional as F
from .inference import metric_arrays

EXPERTS=('single_wald','multi_wald','generic_error','mediation_error')

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return str(x)
    return x
def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class ErrorDensity(nn.Module):
    def __init__(self,dim,width=128,components=5):
        super().__init__();self.dim=dim;self.width=width;self.components=components
        self.net=nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),nn.Linear(width,3*components))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():self.net[-1].bias[components:2*components].copy_(torch.linspace(-1.,1.,components))
    def forward(self,x):
        w,m,s=self.net(x).chunk(3,-1);return w.softmax(-1),m,.05+F.softplus(s)

def logpdf(p,y):
    w,m,s=p;return torch.logsumexp(w.clamp_min(1e-30).log()-s.log()-.5*((y[...,None]-m)/s).square()-.5*math.log(2*math.pi),-1)
def cdf(p,y):
    w,m,s=p;return (w*torch.special.ndtr((y[...,None]-m)/s)).sum(-1)
def _A(delta,scale):
    z=delta/scale;return 2*scale*torch.exp(-z.square()/2)/math.sqrt(2*math.pi)+delta*(2*torch.special.ndtr(z)-1)
def crps(p,y):
    w,m,s=p;return (w*_A(y[:,None]-m,s)).sum(-1)-.5*(w[:,:,None]*w[:,None,:]*_A(m[:,:,None]-m[:,None,:],(s[:,:,None].square()+s[:,None,:].square()).sqrt())).sum((-1,-2))
def density_loss(p,y):return (-logpdf(p,y)+.2*crps(p,y)).mean()
def quantile(p,prob,iterations=48):
    w,m,s=p
    if not all(torch.isfinite(a).all() for a in p):raise ValueError('Nonfinite error density')
    lo=(m-14*s).min(-1).values;hi=(m+14*s).max(-1).values
    if not bool(((cdf(p,lo)<=prob)&(cdf(p,hi)>=prob)).all()):raise ValueError('Error quantile bracket failed')
    for _ in range(iterations):
        mid=(lo+hi)/2;below=cdf(p,mid)<prob;lo=torch.where(below,mid,lo);hi=torch.where(below,hi,mid)
    return (lo+hi)/2

class Policy(nn.Module):
    def __init__(self,dim,width=128):
        super().__init__();self.dim=dim;self.width=width;self.net=nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.LayerNorm(width),nn.Linear(width,4))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)

def categorical_joint(logits,actions):
    """One joint action includes an independent categorical decision per table."""
    return torch.distributions.Categorical(logits=logits).log_prob(actions.detach()).sum(-1)
def dpo_loss(logits,reference_logits,winner,loser,beta=.1):
    current=categorical_joint(logits,winner)-categorical_joint(logits,loser)
    reference=categorical_joint(reference_logits.detach(),winner)-categorical_joint(reference_logits.detach(),loser)
    return -F.logsigmoid(beta*(current-reference)).mean()
def rloo_loss(logits,reference_logits,actions,rewards,kl_weight=.02):
    logprob=categorical_joint(logits[None],actions);rewards=rewards.detach()
    advantage=(rewards-(rewards.sum(0,keepdim=True)-rewards)/(len(rewards)-1)).detach()
    logp=logits.log_softmax(-1);p=logp.exp();kl=(p*(logp-reference_logits.detach().log_softmax(-1))).sum(-1).mean()
    return -(logprob*advantage).mean()+kl_weight*kl

def grouped_indices(mechanism_ids,mask,min_tables=16,max_tables=32):
    ids=np.asarray(mechanism_ids).astype(str);mask=np.asarray(mask,bool);groups=[]
    for mid in np.unique(ids[mask]):
        members=np.flatnonzero(mask&(ids==mid))
        if len(members)<min_tables:raise ValueError(f'Mechanism {mid} has {len(members)} tables; require >= {min_tables}')
        # Keep every row: an oversized group is split without discarding a tail.
        chunks=max(1,math.ceil(len(members)/max_tables));parts=np.array_split(members,chunks)
        if min(map(len,parts))<min_tables:raise ValueError('Episode partition would have fewer than minimum tables')
        groups.extend(parts)
    if not groups:raise ValueError('No mechanism groups')
    return groups

def group_risk(lo,hi,truth,base_se,cost=None,target=.95):
    m=metric_arrays(lo,hi,truth);den=2*np.asarray(base_se,float)
    if not m['valid'].all() or not np.isfinite(den).all() or (den<=0).any():return float('inf')
    nis=float(np.mean(m['IS']/den));coverage=float(m['covered'].mean());tail=float(max(0,m['lower'].mean()-.025)+max(0,m['upper'].mean()-.025))
    c=0. if cost is None else float(np.asarray(cost).mean())
    return nis+20*max(0,target-coverage)**2+5*tail+.01*c

def action_rewards(actions,intervals,truth,base_se,cost=None):
    a=np.asarray(actions,int);idx=np.arange(len(truth));out=[]
    for row in a:
        chosen=intervals[idx,row];cc=None if cost is None else cost[idx,row]
        out.append(-group_risk(chosen[:,0],chosen[:,1],truth,base_se,cc))
    return np.asarray(out,float)

class Cache:
    def __init__(self,path,allow_shared_bank_for_smoke=False):
        with np.load(path,allow_pickle=False) as z:self.a={k:z[k] for k in z.files}
        a=self.a;required=('x_single','x_multi','estimates','ses','truth','mechanism_id','split')
        if not all(k in a for k in required):raise ValueError('Cache requires '+str(required))
        n=len(a['truth'])
        if a['estimates'].shape!=(n,2) or a['ses'].shape!=(n,2):raise ValueError('Exactly single/consensus estimates and SEs required')
        if any(len(a[k])!=n for k in required):raise ValueError('Cache row counts differ')
        self.split=a['split'].astype(str);self.bank=a.get('bank',np.full(n,'all')).astype(str)
        if not allow_shared_bank_for_smoke and ('bank' not in a or (self.bank[self.split=='train']=='all').any()):raise ValueError('Production requires distinct head/sft/dpo/rlvr banks; shared bank allowed only with --allow-shared-bank-for-smoke')
        self.good=np.isfinite(a['x_single']).all(1)&np.isfinite(a['x_multi']).all(1)&np.isfinite(a['estimates']).all(1)&np.isfinite(a['ses']).all(1)&(a['ses']>0).all(1)&np.isfinite(a['truth'])
        # Leakage guard: mechanisms may repeat across training roles, never DEV/TEST.
        for left,right in (('train','dev'),('train','test'),('dev','test')):
            if set(a['mechanism_id'][self.split==left].astype(str))&set(a['mechanism_id'][self.split==right].astype(str)):raise ValueError('Mechanism split leakage')
        if not allow_shared_bank_for_smoke:
            roles=('head','sft','dpo','rlvr')
            for j,left in enumerate(roles):
                for right in roles[j+1:]:
                    lm=(self.split=='train')&(self.bank==left);rm=(self.split=='train')&(self.bank==right)
                    if set(a['mechanism_id'][lm].astype(str))&set(a['mechanism_id'][rm].astype(str)):raise ValueError('Training-role mechanism leakage')
        if (self.split=='test').any():raise ValueError('Training cache must not contain sealed test labels')
        for key in ('seeds','observation_seed','observation_seeds'):
            if key in a and len(np.unique(a[key]))!=n:raise ValueError('Repeated observation seed in training cache')
        self.path=Path(path)
    def mask(self,role,split='train'):
        m=self.split==split
        if split=='train' and not np.all(self.bank[m]=='all'):
            m=m&np.isin(self.bank,(role,'all'))
        if not m.any():raise ValueError('Missing data bank '+role+'/'+split)
        return m

def normalization(cache):
    a=cache.a;m=cache.mask('head')&cache.good;out={}
    if not m.any():raise ValueError('No valid head-training tables')
    for key in ('x_single','x_multi'):out[key]=(a[key][m].mean(0),np.maximum(a[key][m].std(0),.01))
    return out
def normalized(a,key,norm,device):
    center,scale=norm[key];x=(a[key]-center)/scale;return torch.as_tensor(np.nan_to_num(x,nan=0.,posinf=0.,neginf=0.),dtype=torch.float32,device=device)

@torch.no_grad()
def expert_intervals(heads,norm,x_single,x_multi,estimates,ses,alpha=.05,device='cpu'):
    """Observable-only deployment API: NO truth, family, or mechanism input."""
    xs=np.asarray(x_single);xm=np.asarray(x_multi);est=np.asarray(estimates,float);se=np.asarray(ses,float);n=len(est)
    if est.shape!=(n,2) or se.shape!=(n,2):raise ValueError('Point/SE shapes')
    valid=np.isfinite(xs).all(1)&np.isfinite(xm).all(1)&np.isfinite(est).all(1)&np.isfinite(se).all(1)&(se>0).all(1)
    ci=np.full((n,4,2),np.nan);qs=np.full((n,4,2),np.nan);z=NormalDist().inv_cdf(1-alpha/2);evidence={}
    for j in (0,1):ci[valid,j]=np.column_stack((est[valid,j]-z*se[valid,j],est[valid,j]+z*se[valid,j]));qs[valid,j]=[-z,z]
    a=dict(x_single=xs,x_multi=xm)
    for j,key in enumerate(('x_single','x_multi')):
        if not valid.any():continue
        x=normalized(a,key,norm,device)[valid];density=heads[j](x);qlo=quantile(density,alpha/2).cpu().numpy();qhi=quantile(density,1-alpha/2).cpu().numpy()
        ci[valid,j+2]=np.column_stack((est[valid,j]-se[valid,j]*qhi,est[valid,j]-se[valid,j]*qlo));qs[valid,j+2]=np.column_stack((qlo,qhi))
        evidence[EXPERTS[j+2]]={name:np.full((n,heads[j].components),np.nan) for name in ('weights','means','scales')}
        for name,tensor in zip(('weights','means','scales'),density):evidence[EXPERTS[j+2]][name][valid]=tensor.cpu().numpy()
    return dict(intervals=ci,quantiles=qs,mixture_evidence=evidence,valid=valid,expert_names=EXPERTS)

@torch.no_grad()
def predict(heads,policy,norm,x_single,x_multi,estimates,ses,alpha=.05,device='cpu'):
    ex=expert_intervals(heads,norm,x_single,x_multi,estimates,ses,alpha,device)
    a=dict(x_multi=np.asarray(x_multi));x=normalized(a,'x_multi',norm,device);prob=policy(x).softmax(-1).cpu().numpy();action=prob.argmax(-1);idx=np.arange(len(action))
    return dict(intervals=ex['intervals'][idx,action],quantiles=ex['quantiles'][idx,action],action=action,probabilities=prob,point_index=np.where(action%2==0,0,1),experts=ex)

def evaluate_policy(policy,x,ci,a,groups,cost):
    with torch.no_grad():action=policy(x).argmax(-1).cpu().numpy()
    risks=[]
    for ids in groups:
        selected=ci[ids,action[ids]];cc=None if cost is None else cost[ids,action[ids]]
        risks.append(group_risk(selected[:,0],selected[:,1],a['truth'][ids],a['ses'][ids,0],cc))
    return float(np.mean(risks))

def checkpoint(path,model,**extra):
    torch.save(dict(state_dict=model.state_dict(),dim=model.dim,width=model.width,**extra),path)

def train_heads(cache,norm,out,args,logs):
    a=cache.a;models=[];train=cache.mask('head')&cache.good;dev=cache.mask('head','dev')&cache.good
    devgroups=grouped_indices(a['mechanism_id'],cache.mask('head','dev'),args.min_group,args.max_group)
    for j,key in enumerate(('x_single','x_multi')):
        model=ErrorDensity(a[key].shape[1],args.head_width).to(args.device);x=normalized(a,key,norm,args.device);z=torch.as_tensor((a['estimates'][:,j]-a['truth'])/a['ses'][:,j],dtype=torch.float32,device=args.device)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);ids=torch.as_tensor(np.flatnonzero(train),device=args.device);best=math.inf;state=None;chosen=0;updates=0;forwards=0
        for epoch in range(args.head_epochs+1):
            if epoch:
                for ix in ids[torch.randperm(len(ids),device=args.device)].split(args.batch):
                    value=density_loss(model(x[ix]),z[ix]);opt.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();updates+=1;forwards+=1
            with torch.no_grad():
                dev_loss=float(density_loss(model(x[dev]),z[dev]));qlo=quantile(model(x[dev]),.025).cpu().numpy();qhi=quantile(model(x[dev]),.975).cpu().numpy()
            lo=np.full(len(x),np.nan);hi=lo.copy();lo[dev]=a['estimates'][dev,j]-a['ses'][dev,j]*qhi;hi[dev]=a['estimates'][dev,j]-a['ses'][dev,j]*qlo
            cost=a.get('cost');score=float(np.mean([group_risk(lo[ix],hi[ix],a['truth'][ix],a['ses'][ix,0],None if cost is None else cost[ix,j+2]) for ix in devgroups]))
            logs.append(dict(stage='head',arm=EXPERTS[j+2],epoch=epoch,dev_loss=dev_loss,dev_group_risk=score,optimizer_steps=updates,training_forward_calls=forwards,attempted_train=int(cache.mask('head').sum()),valid_train=int(train.sum())))
            if score<best:best=score;state=copy.deepcopy(model.state_dict());chosen=epoch
        if state is None:raise ValueError('No finite head checkpoint')
        model.load_state_dict(state);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        checkpoint(out/f'{EXPERTS[j+2]}.pt',model,components=model.components,epoch=chosen,dev_group_risk=best,parameters=sum(p.numel() for p in model.parameters()));models.append(model)
        print('HEAD',EXPERTS[j+2],chosen,best,flush=True)
    return models

def load_heads(out,device):
    heads=[]
    for arm in EXPERTS[2:]:
        ck=torch.load(Path(out)/f'{arm}.pt',map_location=device,weights_only=True);m=ErrorDensity(ck['dim'],ck['width'],ck['components']).to(device);m.load_state_dict(ck['state_dict']);m.eval()
        for p in m.parameters():p.requires_grad_(False)
        heads.append(m)
    return heads
def load_policy(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=True);m=Policy(ck['dim'],ck['width']).to(device);m.load_state_dict(ck['state_dict']);return m.eval()

def fit_policy(initial,reference,mode,roles,cache,norm,ci,out,args,logs,arm):
    a=cache.a;x=normalized(a,'x_multi',norm,args.device);cost=a.get('cost');devgroups=grouped_indices(a['mechanism_id'],cache.mask('sft','dev'),args.min_group,args.max_group)
    model=copy.deepcopy(initial);ref=copy.deepcopy(reference).eval()
    for p in ref.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);best=math.inf;state=None;chosen=0;updates=0;forwards=0;exposure=0;frozen_preferences={};epoch_roles=[]
    for role,epochs in roles:epoch_roles.extend([role]*epochs)
    with torch.no_grad():reference_logits=ref(x).detach()
    allgroups={role:grouped_indices(a['mechanism_id'],cache.mask(role),args.min_group,args.max_group) for role,_ in roles}
    # Offline SFT labels and DPO preferences are frozen before optimization.
    for role,groups in allgroups.items():
        for gi,ids in enumerate(groups):
            fixed=np.tile(np.arange(4)[:,None],(1,len(ids)));reward=action_rewards(fixed,ci[ids],a['truth'][ids],a['ses'][ids,0],None if cost is None else cost[ids]);label=int(np.argmax(reward))
            pref=dict(label=label,usable=bool(np.isfinite(reward).any()))
            if mode=='dpo':
                with torch.no_grad():actions=torch.distributions.Categorical(logits=reference_logits[ids]).sample((args.draws,)).detach();rewards=action_rewards(actions.cpu().numpy(),ci[ids],a['truth'][ids],a['ses'][ids,0],None if cost is None else cost[ids]);finite=np.flatnonzero(np.isfinite(rewards));order=finite[np.argsort(rewards[finite],kind='stable')]
                gap=float(rewards[order[-1]]-rewards[order[0]]) if len(order)>=2 else 0.
                pref['usable']=len(order)>=2 and gap>1e-10
                if pref['usable']:pref.update(winner=actions[int(order[-1])],loser=actions[int(order[0])],reward_gap=gap)
            frozen_preferences[role,gi]=pref
    preference_summary={role+':'+str(gi):dict(usable=p['usable'],label=p['label'],reward_gap=p.get('reward_gap'),winner=p['winner'].cpu().tolist() if 'winner' in p else None,loser=p['loser'].cpu().tolist() if 'loser' in p else None) for (role,gi),p in frozen_preferences.items()}
    save(out/f'preferences_{arm}.json',preference_summary)
    # The DPO-matched SFT control uses exactly the same eligible episode set;
    # exact/numerical reward ties never become invented preferences.
    dpo_eligibility=None
    if mode=='sft' and any(role=='dpo' for role,_ in roles) and (out/'preferences_sft_dpo.json').exists():dpo_eligibility=json.loads((out/'preferences_sft_dpo.json').read_text())
    for epoch in range(len(epoch_roles)+1):
        if epoch:
            role=epoch_roles[epoch-1];groups=allgroups[role]
            for gi in np.random.default_rng(args.seed+epoch).permutation(len(groups)):
                ids=groups[gi];logits=model(x[ids]);forwards+=1;pref=frozen_preferences[role,gi]
                if dpo_eligibility is not None and role=='dpo' and not dpo_eligibility[role+':'+str(gi)]['usable']:
                    logs.append(dict(stage='matched_DPO_tie_skip',arm=arm,epoch=epoch,attempts=len(ids)));continue
                if mode in ('sft','dpo') and not pref['usable']:
                    logs.append(dict(stage='invalid_offline_preference',arm=arm,epoch=epoch,attempts=len(ids)));continue
                if mode=='sft':value=F.cross_entropy(logits,torch.full((len(ids),),pref['label'],dtype=torch.long,device=args.device))
                elif mode=='dpo':value=dpo_loss(logits,reference_logits[ids],pref['winner'],pref['loser'],args.beta)
                else:
                    with torch.no_grad():actions=torch.distributions.Categorical(logits=logits.detach()).sample((args.draws,)).detach();reward=action_rewards(actions.cpu().numpy(),ci[ids],a['truth'][ids],a['ses'][ids,0],None if cost is None else cost[ids]);reward=torch.as_tensor(reward,dtype=torch.float32,device=args.device)
                    if not torch.isfinite(reward).all():
                        # All attempts retained in risk; a failed frozen expert
                        # cannot supply a finite REINFORCE gradient this update.
                        logs.append(dict(stage='invalid_reward',arm=arm,epoch=epoch,mechanism=str(a['mechanism_id'][ids[0]]),attempts=len(ids)));continue
                    value=rloo_loss(logits,reference_logits[ids],actions,reward)
                opt.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();updates+=1;exposure+=len(ids)
        score=evaluate_policy(model,x,ci,a,devgroups,cost);logs.append(dict(stage=mode,arm=arm,epoch=epoch,dev_group_risk=score,optimizer_steps=updates,training_forward_calls=forwards,table_exposures=exposure,reference='fixed SFT policy' if mode!='sft' else 'offline preferred frozen CI expert'))
        if score<best:best=score;state=copy.deepcopy(model.state_dict());chosen=epoch
        print('POLICY',arm,epoch,score,flush=True)
    if state is None:raise ValueError('No finite policy validation checkpoint')
    model.load_state_dict(state);model.eval();checkpoint(out/f'{arm}.pt',model,epoch=chosen,dev_risk=best,mode=mode,optimizer_steps=updates,training_forward_calls=forwards,table_exposures=exposure,reference_sha256=sha(out/'sft.pt') if (out/'sft.pt').exists() and mode!='sft' else None)
    return model

def train(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(args.seed);np.random.seed(args.seed);cache=Cache(args.cache,args.allow_shared_bank_for_smoke);logs=[]
    normpath=out/'normalization.npz'
    if normpath.exists():
        with np.load(normpath) as z:norm={k:(z[k+'_center'],z[k+'_scale']) for k in ('x_single','x_multi')}
    else:
        norm=normalization(cache);np.savez(normpath,**{k+s:v[i] for k,v in norm.items() for i,s in enumerate(('_center','_scale'))})
    save(out/'training_contract.json',dict(cache_sha256=sha(args.cache),source_hashes={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('inference.py'))},args=vars(args),experts=EXPERTS,error_target='(canonical estimate - known synthetic NIE)/canonical SE; truth offline only',interval='[estimate-SE*q(1-alpha/2),estimate-SE*q(alpha/2)]',policy='Independent categorical per-table actions, joint episode probability; SAME-mechanism16–32table rewards; RLOO reward/actions/advantage detached; fixedSFTreference',risk='mean IS/(2*singleSE)+20*(.95-coverage)_+^2+5*tail_excess+.01*measuredcost',failure='All attempted tables retained in DEV risk; failed expert selection => infinite risk. Invalid RL reward skipped/logged, never silently labeled success.',generic_vs_mediation='single pooled/generic observables vs extra mediation-score/overlap/ESS/crossview diagnostics',mechanism_splits='Disjoint train/dev; no test labels accepted',shared_bank_warning=bool(np.all(cache.bank=='all')),budget='One policy forward and one optimizer step per group for SFT/DPO/RL; extra SFT controls use identical role banks and epoch counts. Reference forward cached once per stage; teacher CI forwards cached separately.'))
    if args.stage in ('head','full'):heads=train_heads(cache,norm,out,args,logs)
    else:heads=load_heads(out,args.device)
    if args.stage=='head':save(out/'training_head.json',logs);return
    a=cache.a;ex=expert_intervals(heads,norm,a['x_single'],a['x_multi'],a['estimates'],a['ses'],device=args.device);ci=ex['intervals'];np.savez_compressed(out/'offline_expert_intervals.npz',intervals=ci,quantiles=ex['quantiles'],cache_hash=sha(args.cache))
    initial=Policy(a['x_multi'].shape[1],args.policy_width).to(args.device)
    if args.stage in ('sft','full'):sft=fit_policy(initial,initial,'sft',[('sft',args.sft_epochs)],cache,norm,ci,out,args,logs,'sft')
    else:sft=load_policy(out/'sft.pt',args.device)
    if args.stage in ('dpo','full'):
        dpo=fit_policy(sft,sft,'dpo',[('dpo',args.post_epochs)],cache,norm,ci,out,args,logs,'sft_dpo')
        fit_policy(sft,sft,'sft',[('dpo',args.post_epochs)],cache,norm,ci,out,args,logs,'extra_sft_dpo')
    elif (out/'sft_dpo.pt').exists():dpo=load_policy(out/'sft_dpo.pt',args.device)
    else:dpo=None
    if args.stage in ('rlvr','full'):
        fit_policy(sft,sft,'rlvr',[('rlvr',args.post_epochs)],cache,norm,ci,out,args,logs,'sft_rlvr')
        fit_policy(sft,sft,'sft',[('rlvr',args.post_epochs)],cache,norm,ci,out,args,logs,'extra_sft_rlvr')
        if dpo is not None:
            fit_policy(dpo,sft,'rlvr',[('rlvr',args.post_epochs)],cache,norm,ci,out,args,logs,'sft_dpo_rlvr')
            fit_policy(sft,sft,'sft',[('dpo',args.post_epochs),('rlvr',args.post_epochs)],cache,norm,ci,out,args,logs,'extra_sft_dpo_rlvr')
    save(out/f'training_{args.stage}.json',logs)
    save(out/'artifacts.json',dict(head_hashes={p.name:sha(p) for p in out.glob('*.pt')},normalization_hash=sha(normpath),cache_hash=sha(args.cache),complete=True,stage=args.stage))

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True);t=sub.add_parser('train');t.add_argument('--stage',choices=('head','sft','dpo','rlvr','full'),default='full');t.add_argument('--cache',required=True);t.add_argument('--output',required=True);t.add_argument('--seed',type=int,default=2026102691);t.add_argument('--device',default='cpu');t.add_argument('--head-width',type=int,default=128);t.add_argument('--policy-width',type=int,default=128);t.add_argument('--head-epochs',type=int,default=40);t.add_argument('--sft-epochs',type=int,default=20);t.add_argument('--post-epochs',type=int,default=10);t.add_argument('--batch',type=int,default=128);t.add_argument('--draws',type=int,default=4);t.add_argument('--beta',type=float,default=.1);t.add_argument('--min-group',type=int,default=16);t.add_argument('--max-group',type=int,default=32);t.add_argument('--allow-shared-bank-for-smoke',action='store_true');args=p.parse_args();train(args)
if __name__=='__main__':main()

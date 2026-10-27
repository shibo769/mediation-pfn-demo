"""Honest multi-view nuisance inference and the unchanged full-sample NIE EIF.

Rows are [X..., A, M, Y]. Context selection sees opposite-fold X only.
Every patient is queried once per view and enters the final canonical score.
"""
from __future__ import annotations
import math,time
from statistics import NormalDist
import numpy as np
import torch

VIEWS=('uniform','coverage','feature_coverage')
NUISANCE_KEYS=('g','q0','q1','u0','u1')

def canonical_nie(rows,nuisance,clip=.01,alpha=.05):
    r=np.asarray(rows,float);nu=np.asarray(nuisance,float);n=len(r)
    if r.ndim!=2 or r.shape[1]<4 or nu.shape!=(n,5) or n<4:raise ValueError('Rows/nuisance shape')
    if not np.isfinite(r).all() or not np.isfinite(nu).all():raise ValueError('Nonfinite table/nuisance')
    if not np.isin(r[:,-3:-1],(0,1)).all():raise ValueError('A and M must be binary')
    if not 0<clip<.5 or not 0<alpha<1:raise ValueError('Invalid clip/alpha')
    if not ((nu[:,:3]>=0)&(nu[:,:3]<=1)).all():raise ValueError('Probability outside unit interval')
    probs=np.clip(nu[:,:3],clip,1-clip);g,q0,q1=probs.T;u0,u1=nu[:,3:].T;a,m,y=r[:,-3:].T
    eta0=u0+q0*(u1-u0);eta1=u0+q1*(u1-u0);mu=np.where(m==1,u1,u0);ratio=np.where(m==1,q0/q1,(1-q0)/(1-q1))
    s11=eta1.copy();s10=eta0.copy();treated=a==1;control=~treated
    s11[treated]+=(y[treated]-eta1[treated])/g[treated]
    s10[treated]+=ratio[treated]*(y[treated]-mu[treated])/g[treated]
    s10[control]+=(mu[control]-eta0[control])/(1-g[control])
    score=s11-s10;estimate=float(score.mean());se=float(score.std(ddof=1)/math.sqrt(n));z=NormalDist().inv_cdf(1-alpha/2)
    if not np.isfinite(score).all() or not math.isfinite(se) or se<=0:raise ValueError('Nonregular score variance')
    return dict(estimate=estimate,se=se,lo=estimate-z*se,hi=estimate+z*se,score=score,clipping_rate=float(np.any(probs!=nu[:,:3],axis=1).mean()),attempted_rows=n,used_rows=n,failed=False)

def select_indices(support_x,budget,seed,view):
    x=np.asarray(support_x,float);n,p=x.shape;k=min(int(budget),n)
    if k<2 or not np.isfinite(x).all() or view not in VIEWS:raise ValueError('Invalid context selection')
    rng=np.random.default_rng(int(seed))
    if view=='uniform':return np.sort(rng.choice(n,k,replace=False))
    columns=np.arange(p) if view=='coverage' else np.sort(rng.choice(p,max(1,(p+1)//2),replace=False))
    xx=x[:,columns];z=(xx-xx.mean(0))/np.maximum(xx.std(0),1e-6)
    # Greedy farthest-point coverage: support X only, no A/M stratification.
    selected=np.empty(k,int);distance=np.full(n,np.inf);chosen=np.zeros(n,bool);j=int(rng.integers(n))
    for t in range(k):
        selected[t]=j;chosen[j]=True;distance=np.minimum(distance,((z-z[j])**2).sum(1));distance[chosen]=-np.inf;j=int(np.argmax(distance))
    return np.sort(selected)

def _sigmoid(x):return np.exp(-np.logaddexp(0.,-x))
def raw_to_nuisance(raw):
    p=np.asarray(raw,float)
    if p.ndim!=2 or p.shape[1]!=7 or not np.isfinite(p).all():raise ValueError('Expected finite raw7 nuisance predictions')
    return np.column_stack((_sigmoid(p[:,:3]),p[:,5:7]))

def _moments(values):
    v=np.asarray(values,float).ravel();s=max(float(v.std()),1e-12);z=(v-v.mean())/s
    return [float(v.mean()),float(v.std()),float(np.mean(z**3)),float(np.mean(z**4))]

def generic_features(rows,summary,pooled=None):
    r=np.asarray(rows,float);x=r[:,:-3];n,p=x.shape;a,m,y=r[:,-3:].T
    xm=x.mean(0);xs=x.std(0);yc=y-y.mean();corr=(x-xm).T@yc/(len(x)*np.maximum(xs,1e-12)*max(y.std(),1e-12))
    f=[math.log(n),math.log(p),summary['estimate'],math.log(summary['se']),float(a.mean()),float(m.mean())]
    f+=_moments(y)+_moments(xm)+_moments(xs)+_moments(corr)
    if pooled is not None and np.asarray(pooled).size:
        pool=np.asarray(pooled,float);f.extend(pool.mean(0));f.extend(pool.std(0))
    return np.asarray(f,np.float32)

def mediation_features(rows,nu,summaries,view_metadata,generic):
    r=np.asarray(rows,float);nu=np.asarray(nu,float);g,q0,q1=np.clip(nu[:,:3],.01,.99).T;a,m=r[:,-3:-1].T
    ratio=np.where(m==1,q0/q1,(1-q0)/(1-q1));w1=np.where(a==1,ratio/g,0.);w0=np.where(a==0,1/(1-g),0.);s=summaries[-1];score=s['score'];se=s['se']
    f=[]
    for v in (g,q0,q1,w1,w0):
        f.extend(np.quantile(v,[0,.01,.05,.5,.95,.99,1]));f.append(float(np.sum(v)**2/max(np.sum(v*v),1e-12)/len(v)))
    f+=_moments(score)+[s['clipping_rate']]
    for other in summaries[:-1]:f.extend([(other['estimate']-s['estimate'])/se,other['se']/se,other['clipping_rate']])
    estimates=np.array([v['estimate'] for v in summaries[:-1]]);f.extend([float(estimates.std()/se),float(np.ptp(estimates)/se)])
    for v in view_metadata:f.extend([v['support_fraction'],v['selected_x_distance_mean'],v['selected_x_distance_max']])
    return np.r_[generic,np.asarray(f,np.float32)].astype(np.float32)

@torch.inference_mode()
def crossfit_views(model,rows,fold_ids,seed=2026102601,context_size=256,device=None,views=VIEWS,query_batch=128,include_pooled=True):
    score_rows=np.asarray(rows,np.float64);r=np.asarray(rows,np.float32);ff=np.asarray(fold_ids);n,p=r.shape[0],r.shape[1]-3
    if r.ndim!=2 or p<1 or ff.shape!=(n,) or set(np.unique(ff))!={0,1}:raise ValueError('Two-fold row assignment required')
    if min(np.sum(ff==f) for f in (0,1))<2:raise ValueError('Each fold requires at least two observations')
    if not np.isfinite(r).all():raise ValueError('Nonfinite data must not be dropped')
    if device is None:
        try:device=next(model.parameters()).device
        except (AttributeError,StopIteration):device='cpu'
    started=time.perf_counter();allnu=[];pooled=[];metadata=[];support_indices=[]
    for vi,view in enumerate(views):
        nu=np.full((n,5),np.nan);pool=[]
        for fold in (0,1):
            ids=np.flatnonzero(ff!=fold);query_ids=np.flatnonzero(ff==fold)
            selected=select_indices(score_rows[ids,:p],context_size,int(seed)+1009*vi+37*fold,view);chosen=ids[selected];support_indices.append(dict(view=view,fold=fold,indices=chosen.copy()))
            support=torch.as_tensor(r[chosen][None],device=device);context=model.encode_support(support) if hasattr(model,'encode_support') else None
            if context is not None and include_pooled and 'pooled' in context:pool.append(context['pooled'].detach().cpu().numpy()[0])
            for start in range(0,len(query_ids),query_batch):
                q=query_ids[start:start+query_batch];query=torch.as_tensor(r[q,:p][None],device=device)
                prediction=model.predict_nuisance(context,query) if context is not None else model(support,query)
                nu[q]=raw_to_nuisance(prediction.detach().cpu().numpy()[0])
            sx=score_rows[ids,:p];mean=sx.mean(0);scale=np.maximum(sx.std(0),1e-6);distance=np.sqrt(np.sum(((score_rows[chosen,:p]-mean)/scale)**2,1))
            metadata.append(dict(view=view,fold=fold,support_fraction=len(chosen)/len(ids),selected_x_distance_mean=float(distance.mean()),selected_x_distance_max=float(distance.max())))
        allnu.append(nu);pooled.append(np.asarray(pool))
    allnu=np.asarray(allnu);consensus=allnu.mean(0);summaries=[canonical_nie(score_rows,nu) for nu in allnu]+[canonical_nie(score_rows,consensus)]
    generic=generic_features(score_rows,summaries[0],pooled[0] if include_pooled else None)
    med=mediation_features(score_rows,consensus,summaries,metadata,generic)
    return dict(nuisance=consensus,views=allnu,summaries=summaries,x_single=generic,x_multi=med,estimates=np.array([summaries[0]['estimate'],summaries[-1]['estimate']]),ses=np.array([summaries[0]['se'],summaries[-1]['se']]),metadata=metadata,support_indices=support_indices,seconds=time.perf_counter()-started,attempted_rows=n,used_rows=n,error='')

def metric_arrays(lo,hi,truth,failed=None):
    lo,hi,y=[np.asarray(v,float) for v in (lo,hi,truth)];failed=np.zeros(len(y),bool) if failed is None else np.asarray(failed,bool)
    valid=(~failed)&np.isfinite(lo)&np.isfinite(hi)&(hi>=lo);length=np.where(valid,hi-lo,np.inf)
    return dict(valid=valid,covered=valid&(lo<=y)&(y<=hi),length=length,IS=np.where(valid,length+40*np.maximum(lo-y,0)+40*np.maximum(y-hi,0),np.inf),lower=valid&(y<lo),upper=valid&(y>hi))

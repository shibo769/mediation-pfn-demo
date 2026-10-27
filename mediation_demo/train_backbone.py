"""Factual-only training with a fresh seen-family mechanism per table/batch."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,json,math,time,os,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from .data import TRAIN_FAMILIES,sample_mechanism,pad_table,derived_seed
from .model import MediationPFN,SIZE_CONFIGS,save_checkpoint

def make_pool(data_seed,count):
    if count<1:raise ValueError('Positive pool size required')
    rng=np.random.default_rng(derived_seed(data_seed,2026102616));pool=[]
    for i in range(count):
        p=int(rng.choice([3,10,30]));family=TRAIN_FAMILIES[i%len(TRAIN_FAMILIES)]
        pool.append(sample_mechanism(derived_seed(data_seed,i,2026102617),p,family))
    return pool

def make_batch(data_seed,step,batch_size,n=None,dev=False,query_budget=None,mechanism_pool=None):
    rng=np.random.default_rng(derived_seed(data_seed,step,2026102610 if dev else 2026102611))
    n=int(rng.choice([200,600,1000])) if n is None else int(n);s=n//2;q=n-s
    if query_budget is not None:q=min(q,int(query_budget))
    support=[];query=[];masks=[];provenance=[]
    for index in range(batch_size):
        p=int(rng.choice([3,10,30]));family=str(rng.choice(TRAIN_FAMILIES));mseed=derived_seed(data_seed,step,index,2026102612 if dev else 2026102613);dseed=derived_seed(data_seed,step,index,2026102614 if dev else 2026102615)
        if mechanism_pool is None:mechanism=sample_mechanism(mseed,p,family)
        else:
            mechanism=mechanism_pool[int(rng.integers(len(mechanism_pool)))];mseed,p,family=mechanism.seed,mechanism.p,mechanism.family
        table=pad_table(mechanism.draw(n,dseed,include_truth=False));order=rng.permutation(n)
        rows=np.column_stack((table['x'],table['a'],table['m'],table['y'])).astype(np.float32)
        support.append(rows[order[:s]]);query.append(rows[order[s:s+q]]);masks.append(table['feature_mask'])
        provenance.append(dict(mechanism_seed=mseed,draw_seed=dseed,p=p,family=family,n=n))
    return dict(support=torch.from_numpy(np.stack(support)),query=torch.from_numpy(np.stack(query)),mask=torch.from_numpy(np.stack(masks)),provenance=provenance)

def factual_loss(model,batch,device):
    support=batch['support'].to(device);query=batch['query'].to(device);mask=batch['mask'].to(device)
    context=model.encode_support(support,feature_mask=mask);pred=model.predict_nuisance(context,query[:,:,:30]);a=query[:,:,-3];m=query[:,:,-2];y=query[:,:,-1]
    q=pred[:,:,1:3].gather(-1,a.long()[...,None]).squeeze(-1)
    mu=pred[:,:,3:].gather(-1,(2*a.long()+m.long())[...,None]).squeeze(-1)
    g_loss=F.binary_cross_entropy_with_logits(pred[:,:,0],a);m_loss=F.binary_cross_entropy_with_logits(q,m)
    y_loss=torch.mean(((mu-y)/context['y_scale'])**2)
    return g_loss+m_loss+y_loss,dict(bce_a=g_loss.detach(),bce_m=m_loss.detach(),scaled_mse_y=y_loss.detach())

def train(args):
    output=Path(args.output)
    if output.suffix.lower()!='.pt':output=output/'model.pt'
    if output.exists():raise FileExistsError('Refusing to overwrite a completed backbone checkpoint')
    output.parent.mkdir(parents=True,exist_ok=True)
    cpu_threads=args.cpu_threads if args.cpu_threads is not None else int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else int(os.environ['NSLOTS']) if 'NSLOTS' in os.environ else None
    if cpu_threads is not None:torch.set_num_threads(cpu_threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed%(2**32));torch.set_float32_matmul_precision('high')
    model=MediationPFN(size=args.size).to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    pool=make_pool(args.data_seed,args.mechanism_pool_size)
    # Independent train-seen development laws only. No OOD generation/selection.
    development=[make_batch(args.data_seed,k,args.dev_batch_size,n=n,dev=True,query_budget=args.query_budget) for k,n in enumerate([200,600,1000])]
    history=[];best=float('inf');best_state=None;best_step=None;start=time.perf_counter()
    for step in range(args.steps+1):
        if step:
            model.train();batch=make_batch(args.data_seed,step,args.batch_size,query_budget=args.query_budget,mechanism_pool=pool)
            optimizer.zero_grad(set_to_none=True);loss,parts=factual_loss(model,batch,args.device)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite backbone training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite backbone gradient')
            optimizer.step()
        if step==0 or step%args.dev_every==0 or step==args.steps:
            model.eval()
            with torch.inference_mode():dev=float(np.mean([float(factual_loss(model,b,args.device)[0]) for b in development]))
            if not math.isfinite(dev):raise FloatingPointError('Nonfinite seen-development loss')
            history.append(dict(step=step,dev_factual_loss=dev,elapsed_seconds=time.perf_counter()-start))
            if dev<best:best=dev;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(json.dumps(dict(step=step,size=args.size,parameters=model.parameter_count,dev_factual_loss=dev,best=best)),flush=True)
    model.load_state_dict(best_state);metadata=dict(seed=args.seed,data_seed=args.data_seed,steps=args.steps,selected_step=best_step,selected_dev_factual_loss=best,
        objective='Observed BCE(A,g)+BCE(M,q_A)+MSE(Y,mu_AM)/opposite-support Yscale^2',training_families=TRAIN_FAMILIES,
        selection='Minimum independent train-seen development factual loss; epoch0 eligible; no OOD data',
        dev_provenance=[r for b in development for r in b['provenance']],arguments=vars(args),created_utc=datetime.now(timezone.utc).isoformat())
    metadata.update(training_mechanism_pool=dict(size=len(pool),seeds=[m.seed for m in pool],population_rows_per_mechanism=8192,
        strategy='Fixed independent seeded train-seen mechanism pool reused with fresh iid row draws/noise every step; exact truth lazy and unused by factual training',
        unique_train_families=sorted({m.family for m in pool}),finite_population_bytes=sum(m.population.nbytes for m in pool)))
    log_path=output.with_suffix('.training.json')
    log_path.write_text(json.dumps(dict(**metadata,parameter_count=model.parameter_count,history=history),indent=2)+'\n',encoding='utf-8')
    metadata['training_log_sha256']=hashlib.sha256(log_path.read_bytes()).hexdigest()
    save_checkpoint(output,model,**metadata)
    return output

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--size',choices=tuple(SIZE_CONFIGS),default='small');p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--data-seed',type=int,required=True);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--dev-batch-size',type=int,default=4);p.add_argument('--dev-every',type=int,default=100);p.add_argument('--query-budget',type=int,default=64);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--mechanism-pool-size',type=int,default=256);p.add_argument('--cpu-threads',type=int,default=None)
    return p

if __name__=='__main__':
    args=parser().parse_args()
    if args.steps<0 or args.dev_every<1 or args.batch_size<1 or args.dev_batch_size<1 or args.query_budget<1:raise ValueError('Invalid training budget')
    train(args)

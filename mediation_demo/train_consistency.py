"""Synthetic-oracle supervised nuisance adapters with optional mediation consistency.

This module never updates a PFN. Cache inputs are frozen, cross-fitted nuisance
predictions, not learned adjacency matrices. Training does not accept test rows.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import torch
from torch import nn

def load(path):
    with np.load(path,allow_pickle=False) as z: a={k:z[k] for k in z.files}
    need=['x','base','target','mechanism_id','split']
    if any(k not in a for k in need):raise ValueError('cache requires '+', '.join(need))
    n=len(a['x'])
    if a['x'].ndim!=2 or a['base'].shape!=(n,3) or a['target'].shape!=(n,4):raise ValueError('invalid arrays')
    if any(len(a[k])!=n for k in need):raise ValueError('unaligned arrays')
    if not np.isfinite(a['x']).all() or not np.isfinite(a['base']).all() or not np.isfinite(a['target']).all():raise ValueError('nonfinite')
    if not ((a['base'][:,0]>=0)&(a['base'][:,0]<=1)&(a['target'][:,0]>=0)&(a['target'][:,0]<=1)).all():raise ValueError('q0 must be a probability')
    split=a['split'].astype(str);mech=a['mechanism_id'].astype(str)
    if set(split)!={'train','dev'}:raise ValueError('cache must contain train/dev only; sealed test is prohibited')
    if set(mech[split=='train'])&set(mech[split=='dev']):raise ValueError('mechanisms overlap across train/dev')
    return a

class Adapter(nn.Module):
    def __init__(self,p,width=32):
        super().__init__();self.net=nn.Sequential(nn.Linear(p+3,width),nn.GELU(),nn.Linear(width,4))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,b):
        delta=self.net(torch.cat([x,b],1));q=torch.sigmoid(torch.logit(b[:,0].clamp(.001,.999))+delta[:,0]);m0=b[:,1]+delta[:,1];m1=b[:,2]+delta[:,2]
        # Direct eta has separately parameterized output; formula uses adjusted q and mu.
        direct=(1-b[:,0])*b[:,1]+b[:,0]*b[:,2]+delta[:,3]
        formula=(1-q)*m0+q*m1
        return torch.stack([q,m0,m1,direct],1),formula

def metrics(pred,formula,target):
    sq=(pred-target).square();return {'q0_mse':float(sq[:,0].mean()),'mu10_mse':float(sq[:,1].mean()),'mu11_mse':float(sq[:,2].mean()),'direct_eta_mse':float(sq[:,3].mean()),'formula_eta_mse':float((formula-target[:,3]).square().mean()),'consistency_mse':float((pred[:,3]-formula).square().mean()),'supervised_sum':float(sq.mean(0).sum())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--lambda-consistency',type=float,required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--seed',type=int,default=14);p.add_argument('--lr',type=float,default=.002);p.add_argument('--width',type=int,default=32);args=p.parse_args()
    if args.lambda_consistency<0 or args.epochs<0:raise ValueError('invalid hyperparameter')
    a=load(args.cache);split=a['split'].astype(str);train=np.flatnonzero(split=='train');dev=np.flatnonzero(split=='dev')
    # Normalize only on training data, and retain stats for deployment.
    center=a['x'][train].mean(0);scale=np.maximum(a['x'][train].std(0),1e-5)
    x=torch.tensor((a['x']-center)/scale,dtype=torch.float32);b=torch.tensor(a['base'],dtype=torch.float32);y=torch.tensor(a['target'],dtype=torch.float32)
    torch.set_num_threads(1);torch.manual_seed(args.seed);net=Adapter(x.shape[1],args.width);opt=torch.optim.Adam(net.parameters(),lr=args.lr);best=float('inf');state=None;selected=0
    for epoch in range(args.epochs+1):
        if epoch:
            for idx in torch.randperm(len(train)).split(256):
                j=torch.tensor(train)[idx];pred,formula=net(x[j],b[j]);loss=(pred-y[j]).square().mean(0).sum()+args.lambda_consistency*(pred[:,3]-formula).square().mean();opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            pred,formula=net(x[dev],b[dev]);v=metrics(pred,formula,y[dev]);risk=v['supervised_sum']
        if risk<best:best=risk;state={k:t.detach().clone() for k,t in net.state_dict().items()};selected=epoch
    net.load_state_dict(state);net.eval()
    with torch.no_grad(): pred,formula=net(x[dev],b[dev]);report=metrics(pred,formula,y[dev])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.save({'state_dict':state,'width':args.width,'input_dim':x.shape[1],'center':torch.tensor(center),'scale':torch.tensor(scale),'lambda_consistency':args.lambda_consistency},out/'adapter.pt')
    result={'args':vars(args),'cache_sha256':hashlib.sha256(Path(args.cache).read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256((out/'adapter.pt').read_bytes()).hexdigest(),'train_rows':len(train),'dev_rows':len(dev),'train_mechanisms':len(set(a['mechanism_id'][train].astype(str))),'dev_mechanisms':len(set(a['mechanism_id'][dev].astype(str))),'selected_epoch':selected,'dev':report,'supervision':'synthetic oracle row-level q0, mu10, mu11, eta10; not observed clinical data','target_uses_test':False,'backbone_updated':False}
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()

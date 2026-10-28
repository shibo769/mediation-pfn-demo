"""Fit a new error-density CI head on adapter-disjoint synthetic mechanisms."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .posttrain import ErrorDensity,density_loss

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--shards',nargs='+',required=True);p.add_argument('--freeze',required=True);p.add_argument('--candidate',choices=['zero','selected'],required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=140702);p.add_argument('--width',type=int,default=128);a=p.parse_args()
    fp=Path(a.freeze);f=json.loads(fp.read_text());entry=next(x for x in f['candidates'] if x['lambda']==0.) if a.candidate=='zero' else {'sha256':f['adapter_sha256']};adapter_sha=entry['sha256'];parts=[];metas=[]
    for s in a.shards:
        path=Path(s);m=json.loads(path.with_suffix('.manifest.json').read_text());
        if sha(path)!=m['sha256'] or m['candidate']!=a.candidate or m['adapter_sha256']!=adapter_sha or m['freeze_sha256']!=sha(fp) or m['split'] not in ('train','dev'):raise ValueError('Bad CI shard provenance')
        with np.load(path,allow_pickle=False) as z:part={k:z[k] for k in z.files}
        if set(part['split'].astype(str))!={m['split']} or set(part['mechanism_id'].astype(str))!=set(m['mechanisms']):raise ValueError('CI shard identity drift')
        parts.append(part);metas.append(m)
    for split in ('train','dev'):
        group=[m for m in metas if m['split']==split]
        if not group or len({x['shards'] for x in group})!=1 or sorted(x['shard'] for x in group)!=list(range(group[0]['shards'])):raise ValueError('Missing CI shard')
    data={k:np.concatenate([z[k] for z in parts]) for k in parts[0]};split=data['split'].astype(str);ids=data['mechanism_id'].astype(str);tr=np.flatnonzero(split=='train');dev=np.flatnonzero(split=='dev')
    if set(ids[tr])&set(ids[dev]) or set(ids)&set(f['train_dev_mechanisms']):raise ValueError('CI train/DEV/adapter mechanisms overlap')
    if not np.isfinite(data['x_multi']).all() or not np.isfinite(data['estimates']).all() or not np.isfinite(data['truth']).all() or not np.isfinite(data['ses']).all() or (data['ses']<=0).any():raise ValueError('Nonfinite CI attempt')
    center=data['x_multi'][tr].mean(0);scale=np.maximum(data['x_multi'][tr].std(0),.01);x=torch.tensor((data['x_multi']-center)/scale,dtype=torch.float32);z=torch.tensor((data['estimates']-data['truth'])/data['ses'],dtype=torch.float32)
    torch.set_num_threads(4);torch.manual_seed(a.seed);model=ErrorDensity(x.shape[1],a.width);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4);best=float('inf');state=None;chosen=0;logs=[]
    for epoch in range(a.epochs+1):
        if epoch:
            model.train()
            for j in torch.randperm(len(tr)).split(256):
                ix=torch.tensor(tr)[j];loss=density_loss(model(x[ix]),z[ix]);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        model.eval()
        with torch.no_grad():risk=float(density_loss(model(x[dev]),z[dev]));logs.append({'epoch':epoch,'dev_density_loss':risk})
        if np.isfinite(risk) and risk<best:best=risk;state=copy.deepcopy(model.state_dict());chosen=epoch
    if state is None:raise ValueError('No finite CI candidate')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.save({'state':state,'center':torch.tensor(center),'scale':torch.tensor(scale),'dim':x.shape[1],'width':a.width,'adapter_sha256':adapter_sha,'freeze_sha256':sha(fp)},out/'ci_head.pt')
    record={'candidate':a.candidate,'adapter_sha256':adapter_sha,'freeze_sha256':sha(fp),'head_sha256':sha(out/'ci_head.pt'),'selected_epoch':chosen,'dev_density_loss':best,'train_mechanisms':sorted(set(ids[tr])),'dev_mechanisms':sorted(set(ids[dev])),'train_rows':len(tr),'dev_rows':len(dev),'source_shards':[{'sha256':m['sha256'],'source_manifest_sha256':m['source_manifest_sha256'],'split':m['split'],'shard':m['shard']} for m in metas],'logs':logs,'target':'(EIF estimate - synthetic truth)/EIF standard error','status':'ci_head_frozen_before_test'}
    with (out/'ci_freeze.json').open('x') as w:json.dump(record,w,indent=2)
    print(json.dumps({'candidate':a.candidate,'head_sha256':record['head_sha256'],'epoch':chosen,'dev_density_loss':best,'train_rows':len(tr),'dev_rows':len(dev)}),flush=True)

if __name__=='__main__':main()

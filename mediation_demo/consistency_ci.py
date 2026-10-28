"""Apply independently trained error head to adapted EIF estimates."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from mediation_demo.posttrain import ErrorDensity,quantile

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_ci(path,adapter_sha):
    path=Path(path);meta=json.loads((path/'ci_freeze.json').read_text());weights=path/'ci_head.pt'
    if meta['status']!='ci_head_frozen_before_test' or meta['head_sha256']!=sha(weights) or meta['adapter_sha256']!=adapter_sha:raise ValueError('CI-head freeze mismatch')
    state=torch.load(weights,map_location='cpu',weights_only=True)
    if state['adapter_sha256']!=adapter_sha or state['freeze_sha256']!=meta['freeze_sha256']:raise ValueError('CI-head provenance mismatch')
    net=ErrorDensity(state['dim'],state['width']);net.load_state_dict(state['state']);net.eval()
    return net,np.asarray(state['center'],float),np.asarray(state['scale'],float),meta

def ci(features,estimate,se,head,alpha=.05):
    net,center,scale,meta=head;x=np.asarray(features,float)
    if x.shape!=center.shape or scale.shape!=center.shape or not np.isfinite(x).all() or (scale<=0).any() or not np.isfinite(estimate) or not np.isfinite(se) or se<=0:raise ValueError('Invalid CI-head input')
    with torch.inference_mode():
        parts=net(torch.as_tensor(((x-center)/scale)[None],dtype=torch.float32));lo=float(quantile(parts,alpha/2)[0]);hi=float(quantile(parts,1-alpha/2)[0]);bias=float((parts[0]*parts[1]).sum(-1)[0])
    return {'point':estimate-se*bias,'se':se,'interval':[estimate-se*hi,estimate-se*lo]}

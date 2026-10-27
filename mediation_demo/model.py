"""Trainable variable-p causal-role PFN with honest support/query separation."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import math
import torch
from torch import nn

QUERY_ORDER=('logit_g','logit_q0','logit_q1','mu00','mu01','mu10','mu11')

@dataclass
class ModelConfig:
    width:int=64
    heads:int=4
    inducing_tokens:int=16
    depth:int=1
    ff_multiplier:int=3
    max_p:int=30
    size:str='small'

SIZE_CONFIGS={
    'small':dict(width=64,heads=4,depth=2,ff_multiplier=3),
    'medium':dict(width=128,heads=4,depth=4,ff_multiplier=3),
    'large':dict(width=256,heads=8,depth=5,ff_multiplier=3),
    # Same architecture used by the 13,496,449-parameter research backbone.
    'xl_10m':dict(width=384,heads=12,depth=7,ff_multiplier=3),
}

class MediationPFN(nn.Module):
    query_order=QUERY_ORDER
    normalization=dict(x='Opposite-fold support-only feature mean and population std; floor1e-4; masked padded features',
        y='Opposite-fold support-only mean and population std; floor1e-3; all four outcome means restored to original units')
    def __init__(self,config:ModelConfig|dict|None=None,size='small'):
        super().__init__()
        if isinstance(config,str):size=config;config=None
        self.config=ModelConfig(size=size,**SIZE_CONFIGS[size]) if config is None else ModelConfig(**config) if isinstance(config,dict) else config
        c=self.config;w=c.width
        self.x_map=nn.Sequential(nn.Linear(2*c.max_p,w),nn.GELU(),nn.Linear(w,w))
        self.a_role=nn.Embedding(2,w);self.m_role=nn.Embedding(2,w);self.y_role=nn.Linear(1,w)
        self.row_fuse=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,c.ff_multiplier*w),nn.GELU(),nn.Linear(c.ff_multiplier*w,w))
        self.inducing=nn.Parameter(torch.randn(1,c.inducing_tokens,w)*.02)
        self.size_map=nn.Linear(2,w);self.support_attention=nn.MultiheadAttention(w,c.heads,dropout=0,batch_first=True)
        self.support_norm=nn.LayerNorm(w)
        self.token_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(w,c.heads,dim_feedforward=c.ff_multiplier*w,dropout=0,activation='gelu',batch_first=True,norm_first=True),num_layers=c.depth,enable_nested_tensor=False)
        self.type_embedding=nn.Embedding(7,w)
        self.query_attention=nn.MultiheadAttention(w,c.heads,dropout=0,batch_first=True)
        self.query_norm=nn.LayerNorm(w)
        self.query_ff=nn.Sequential(nn.Linear(w,c.ff_multiplier*w),nn.GELU(),nn.Linear(c.ff_multiplier*w,w))
        self.output_norm=nn.LayerNorm(w);self.output=nn.Linear(w,1)
        nn.init.normal_(self.output.weight,std=.02);nn.init.zeros_(self.output.bias)

    @property
    def parameter_count(self):return sum(p.numel() for p in self.parameters())
    def config_dict(self):return asdict(self.config)
    def _pad(self,x):
        if x.ndim!=3 or not 1<=x.shape[-1]<=self.config.max_p:raise ValueError('X must be batch by rows by p<=30, covariates only')
        return torch.nn.functional.pad(x,(0,self.config.max_p-x.shape[-1]))
    def encode_support(self,support_x,support_a=None,support_m=None,support_y=None,feature_mask=None):
        if support_a is None:
            if support_m is not None or support_y is not None or support_x.shape[-1]<4:raise ValueError('Packed support is [X...,A,M,Y]')
            support_x,support_a,support_m,support_y=support_x[...,:-3],support_x[...,-3],support_x[...,-2],support_x[...,-1]
        p=support_x.shape[-1];x=self._pad(support_x);b,n,_=x.shape
        if n<1 or any(t is None or t.shape!=(b,n) for t in (support_a,support_m,support_y)):raise ValueError('Support label shapes must match support rows')
        if not bool(torch.isfinite(x).all() and torch.isfinite(support_y).all()):raise ValueError('Nonfinite support')
        if not bool(((support_a==0)|(support_a==1)).all() and ((support_m==0)|(support_m==1)).all()):raise ValueError('Binary A/M required')
        if feature_mask is None:
            feature_mask=x.new_zeros((b,self.config.max_p));feature_mask[:,:p]=1
        if feature_mask.shape!=(b,self.config.max_p) or not bool(((feature_mask==0)|(feature_mask==1)).all()):raise ValueError('Feature mask must be batch by max_p binary')
        mask=feature_mask[:,None,:];center=x.mean(1,keepdim=True);scale=x.std(1,unbiased=False,keepdim=True).clamp_min(1e-4)
        normalized=((x-center)/scale)*mask
        y_center=support_y.mean(1,keepdim=True);y_scale=support_y.std(1,unbiased=False,keepdim=True).clamp_min(1e-3)
        y=(support_y-y_center)/y_scale
        role=self.x_map(torch.cat((normalized,mask.expand(-1,n,-1)),dim=-1))+self.a_role(support_a.long())+self.m_role(support_m.long())+self.y_role(y[...,None])
        rows=role+self.row_fuse(role)
        size=rows.new_tensor([math.log(n)/6,n**-.5]).expand(b,-1)
        query=self.inducing.expand(b,-1,-1)+self.size_map(size)[:,None,:]
        attended,_=self.support_attention(query,rows,rows,need_weights=False)
        tokens=self.token_encoder(self.support_norm(query+attended))
        return dict(tokens=tokens,pooled=tokens.mean(1),x_center=center,x_scale=scale,feature_mask=feature_mask,y_center=y_center,y_scale=y_scale,input_p=p)

    def predict_nuisance(self,context,query_x):
        if query_x.shape[-1] not in (context['input_p'],self.config.max_p):raise ValueError('Query must contain support covariates only, with matching p or max_p padding')
        x=self._pad(query_x);b,n,_=x.shape
        if b!=context['tokens'].shape[0] or n<1 or not bool(torch.isfinite(x).all()):raise ValueError('Invalid covariate query')
        mask=context['feature_mask'][:,None,:];z=(x-context['x_center'])/context['x_scale']*mask
        query=self.x_map(torch.cat((z,mask.expand(-1,n,-1)),dim=-1))[:,:,None,:]+self.type_embedding.weight[None,None,:,:]
        query=query.reshape(b,n*7,self.config.width)
        attended,_=self.query_attention(query,context['tokens'],context['tokens'],need_weights=False)
        hidden=self.query_norm(query+attended);hidden=self.output_norm(hidden+self.query_ff(hidden))
        raw=self.output(hidden).reshape(b,n,7)
        means=raw[:,:,3:]*context['y_scale'][:,None,:]+context['y_center'][:,None,:]
        return torch.cat((raw[:,:,:3],means),dim=-1)

    def forward(self,support_rows,query_x,feature_mask=None):
        return self.predict_nuisance(self.encode_support(support_rows,feature_mask=feature_mask),query_x)

def save_checkpoint(path,model,**metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(format='v9-mediation-pfn-1',config=model.config_dict(),normalization=model.normalization,parameter_count=model.parameter_count,
                    state_dict=model.state_dict(),**metadata),path)

def load_checkpoint(path,device='cpu'):
    # Public demo accepts safe tensor/primitive-only exports, not arbitrary pickle.
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('format')!='v9-mediation-pfn-1':raise ValueError('Unknown checkpoint format')
    model=MediationPFN(checkpoint['config']);model.load_state_dict(checkpoint['state_dict']);model.to(device).eval()
    if model.parameter_count!=checkpoint['parameter_count']:raise ValueError('Checkpoint parameter count mismatch')
    return model

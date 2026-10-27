"""Variable-p binary mediation SCMs with exact finite-population NIE.

Only X is a finite population. A/M/Y are drawn afresh and independently for
every sampled row; table rows are iid draws with replacement from that law.
No integration approximation or Monte Carlo estimate is used for truth.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.special import expit

MAX_P=30
TRAIN_FAMILIES=('linear','smooth','heterosked','weak_overlap','covariate_shift')
OOD_FAMILIES=('tree','rff','mlp','combinations')
FAMILIES=TRAIN_FAMILIES+OOD_FAMILIES
ALIASES={'heteroskedastic':'heterosked','covshift':'covariate_shift','weakoverlap':'weak_overlap','combined':'combinations'}
GENERATOR_VERSION='v9-finite-population-scm-1'

def derived_seed(seed,*indices):
    return int(np.random.SeedSequence([int(seed),*[int(i) for i in indices]]).generate_state(1,np.uint64)[0])

def make_folds(n,seed):
    if n<4:raise ValueError('Need at least four rows')
    order=np.random.default_rng(int(seed)).permutation(n);out=np.empty(n,np.int8);out[order]=np.arange(n)%2;return out
make_fold_ids=make_folds

def observed_rows(table,dtype=np.float32):
    return np.column_stack((table['x'],table['a'],table['m'],table['y'])).astype(dtype)

def pad_x(x,max_p=MAX_P):
    x=np.asarray(x,np.float32)
    if x.ndim!=2 or not 1<=x.shape[1]<=max_p:raise ValueError('Expected n by p, 1<=p<=max_p')
    padded=np.zeros((len(x),max_p),np.float32);padded[:,:x.shape[1]]=x
    mask=np.zeros(max_p,np.float32);mask[:x.shape[1]]=1
    return padded,mask

def pad_table(table,max_p=MAX_P):
    x,mask=pad_x(table['x'],max_p)
    return dict(x=x,a=np.asarray(table['a'],np.float32),m=np.asarray(table['m'],np.float32),y=np.asarray(table['y'],np.float32),feature_mask=mask)

@dataclass
class SCM:
    seed:int
    p:int
    family:str
    population_seed:int|None
    population:np.ndarray
    parameters:dict
    external_population:bool=False

    def __post_init__(self):
        self.population=np.asarray(self.population,np.float32).copy()
        if self.population.ndim!=2 or self.population.shape[1]!=self.p or not np.isfinite(self.population).all():raise ValueError('Invalid covariate population')
        if len(self.population)<2:raise ValueError('Population needs at least two covariate rows')
        self.population.flags.writeable=False
        x=self.population.astype(float);self.x_center=x.mean(0);self.x_scale=np.maximum(x.std(0),1e-6)
        self._theta_NIE=None
        self.population_hash=hashlib.sha256(self.population.tobytes()).hexdigest()

    @property
    def mechanism_seed(self):return self.seed
    @property
    def theta_NIE(self):
        if self._theta_NIE is None:
            nu=self.nuisance(self.population)
            self._theta_NIE=float(np.mean((nu['q1']-nu['q0'])*(nu['mu11']-nu['mu10'])))
        return self._theta_NIE
    @property
    def positivity_floor(self):return self.parameters['floor']
    @property
    def truth(self):return self.theta_NIE

    def _components(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=self.p or not np.isfinite(x).all():raise ValueError('Expected finite n by mechanism p covariates')
        z=(x-self.x_center)/self.x_scale;par=self.parameters
        linear=z@par['weights'].T
        pairs=par['pairs'];interaction=np.zeros(len(z))
        if len(pairs):interaction=np.mean(np.tanh(z[:,pairs[:,0]])*np.tanh(z[:,pairs[:,1]]),axis=1)
        smooth=np.sin(z@par['smooth_direction'])+.4*np.tanh(z[:,par['active'][0]])
        nonlinear=np.zeros(len(z))
        if self.family in ('smooth','heterosked','covariate_shift'):nonlinear=.45*smooth+.3*interaction
        if self.family in ('tree','combinations'):
            tree=np.mean(np.where(z[:,par['tree_columns']]>par['tree_thresholds'],par['tree_values'], -par['tree_values']),axis=1)
            nonlinear+=1.4*tree
        if self.family in ('rff','combinations'):
            nonlinear+=np.sin(z@par['rff_frequencies'].T+par['rff_phase'])@par['rff_weights']/np.sqrt(8)
        if self.family in ('mlp','combinations'):
            nonlinear+=np.tanh(z@par['mlp_weights'].T+par['mlp_bias'])@par['mlp_output']/np.sqrt(12)
        return z,linear,nonlinear,interaction

    def nuisance(self,x):
        z,linear,nl,interaction=self._components(x);par=self.parameters;floor=par['floor']
        g=floor+(1-2*floor)*expit(par['propensity_scale']*(par['intercepts'][0]+linear[:,0]+.5*nl))
        varying=self.family!='linear'
        l0=par['intercepts'][1]+linear[:,1]+.8*nl+(.2*interaction if varying else 0.)
        delta=par['gamma']+(.35*np.tanh(linear[:,2]) if varying else 0.)
        q0=floor+(1-2*floor)*expit(l0);q1=floor+(1-2*floor)*expit(l0+delta)
        baseline=par['intercepts'][2]+linear[:,3]+.7*nl
        treatment=par['treatment_effect']+(.25*np.tanh(linear[:,4]) if varying else 0.)
        d0=par['mediator_effect']+(.35*np.tanh(linear[:,2])+.25*nl if varying else 0.)
        d1=d0+par['am_effect']+(.25*np.tanh(linear[:,4]) if varying else 0.)
        if not varying:d0=np.full(len(z),d0);d1=np.full(len(z),d1)
        mu00=baseline;mu01=baseline+d0;mu10=baseline+treatment;mu11=mu10+d1
        if par['heteroskedastic']:
            sd=par['noise_scale']*(.65+.7*expit(linear[:,5]+.5*nl))
        else:sd=np.full(len(z),par['noise_scale'])
        return dict(g=g,q0=q0,q1=q1,mu00=mu00,mu01=mu01,mu10=mu10,mu11=mu11,u0=mu10,u1=mu11,
                    conditional_sd=sd,conditional_variance=sd**2)

    def draw(self,n,draw_seed,include_truth=True):
        if n<1:raise ValueError('Positive sample size required')
        rng=np.random.default_rng(int(draw_seed));indices=rng.integers(0,len(self.population),size=int(n));x=self.population[indices].copy();nu=self.nuisance(x)
        a=rng.binomial(1,nu['g']);q=np.where(a,nu['q1'],nu['q0']);m=rng.binomial(1,q)
        mu=np.where(a,np.where(m,nu['mu11'],nu['mu10']),np.where(m,nu['mu01'],nu['mu00']))
        df=self.parameters['noise_df'];noise=rng.normal(size=n) if df is None else rng.standard_t(df,size=n)*np.sqrt((df-2)/df)
        y=mu+nu['conditional_sd']*noise
        result=dict(x=x,a=a.astype(np.float32),m=m.astype(np.float32),y=y,p=self.p,mechanism_seed=self.seed,draw_seed=int(draw_seed),population_indices=indices)
        if include_truth:result['theta_NIE']=self.theta_NIE
        return result

    def to_dict(self,include_population=False):
        par={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in self.parameters.items()}
        result=dict(generator_version=GENERATOR_VERSION,seed=self.seed,p=self.p,family=self.family,population_seed=self.population_seed,
            population_size=len(self.population),population_hash=self.population_hash,external_population=self.external_population,
            theta_NIE=self.theta_NIE,positivity_floor=self.positivity_floor,parameters=par,
            truth_definition='Exact mean((q1-q0)*(mu11-mu10)) across the declared fixed covariate population; iid row sampling with replacement')
        if include_population:result['population']=self.population.tolist()
        return result

    @classmethod
    def from_dict(cls,record,population=None):
        if record.get('generator_version')!=GENERATOR_VERSION:raise ValueError('Generator version mismatch')
        if population is None and record.get('external_population') and 'population' in record:population=np.asarray(record['population'],np.float32)
        if record.get('external_population') and population is None:raise ValueError('External covariate population must be supplied')
        mechanism=sample_mechanism(record['seed'],record['p'],record['family'],record['population_size'],population)
        if mechanism.population_hash!=record['population_hash'] or mechanism.theta_NIE!=record['theta_NIE']:raise ValueError('Mechanism reconstruction mismatch')
        return mechanism

def sample_mechanism(seed,p,family,population_size=8192,population=None):
    p=int(p);family=ALIASES.get(family,family)
    if not 1<=p<=MAX_P or family not in FAMILIES:raise ValueError('Unknown p/family')
    rng=np.random.default_rng(int(seed));active=np.sort(rng.choice(p,size=min(p,max(3,int(np.ceil(np.sqrt(p))))),replace=False))
    weights=np.zeros((6,p));weights[:,active]=rng.normal(0,.65/np.sqrt(len(active)),size=(6,len(active)))
    pairs=np.array([(int(active[j]),int(active[(j+1)%len(active)])) for j in range(len(active)) if len(active)>1],dtype=int).reshape(-1,2)
    direction=np.zeros(p);direction[active]=rng.normal(size=len(active))/np.sqrt(len(active))
    rff=np.zeros((8,p));rff[:,active]=rng.normal(size=(8,len(active)))*rng.uniform(.8,2.5)/np.sqrt(len(active))
    mlp=np.zeros((12,p));mlp[:,active]=rng.normal(size=(12,len(active)))/np.sqrt(len(active))
    sign=float(rng.choice([-1.,1.]));floor=.01 if family in ('weak_overlap','combinations') else .03
    par=dict(weights=weights,active=active,pairs=pairs,smooth_direction=direction,rff_frequencies=rff,rff_phase=rng.uniform(-np.pi,np.pi,8),
        rff_weights=rng.normal(size=8),mlp_weights=mlp,mlp_bias=rng.normal(0,.5,12),mlp_output=rng.normal(size=12),
        tree_columns=rng.choice(active,size=6,replace=True),tree_thresholds=rng.uniform(-.8,.8,6),tree_values=rng.normal(size=6),
        intercepts=rng.normal(0,.35,3),floor=floor,propensity_scale=2.8 if family in ('weak_overlap','combinations') else 1.,
        gamma=sign*rng.uniform(.3,1.3),mediator_effect=float(rng.choice([-1.,1.]))*rng.uniform(.2,1.2),am_effect=rng.normal(0,.35),
        treatment_effect=rng.normal(.3,.5),noise_scale=rng.uniform(.45,1.2),heteroskedastic=family in ('heterosked','combinations'),
        noise_df=8 if family=='heterosked' else 6 if family=='combinations' else None)
    population_seed=derived_seed(seed,8192,2026102601)
    external=population is not None
    if not external:
        prng=np.random.default_rng(population_seed);common=prng.normal(size=(population_size,1));population=.35*common+np.sqrt(1-.35**2)*prng.normal(size=(population_size,p))
        if family=='covariate_shift':population=population*prng.uniform(.6,1.8,p)+prng.uniform(-1,1,p)
        population=population.astype(np.float32)
    elif np.asarray(population).shape!=(population_size,p):raise ValueError('External population shape must equal population_size by p')
    return SCM(int(seed),p,family,None if external else population_seed,population,par,external)

def exact_truth(mechanism):return mechanism.theta_NIE
def sample_mechanisms(count,seed,family,p=3,population_size=8192):
    return [sample_mechanism(derived_seed(seed,i,2026102602),p,family,population_size) for i in range(count)]

def generate_table(mechanism,n,seed):return mechanism.draw(n,seed)

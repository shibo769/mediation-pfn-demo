"""Select the adapter on DEV alone and freeze an immutable deployment bundle."""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
from .train_consistency import load

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--candidates',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    data=load(a.cache);cache_sha=sha(a.cache);records=[]
    for path in a.candidates:
        path=Path(path);r=json.loads((path/'report.json').read_text());adapter=path/'adapter.pt'
        if r['cache_sha256']!=cache_sha or r.get('adapter_sha256')!=sha(adapter):raise ValueError('Candidate/cache hash mismatch')
        if r['target_uses_test'] or r['backbone_updated']:raise ValueError('Candidate violates frozen training contract')
        risk=float(r['dev']['supervised_sum'])
        if not np.isfinite(risk):raise ValueError('Invalid DEV objective')
        records.append((risk,float(r['args']['lambda_consistency']),path,r))
    if len({v[1] for v in records})!=len(records):raise ValueError('Duplicate lambda candidates')
    # Predeclared deterministic rule: minimize same synthetic-oracle DEV risk,
    # smaller lambda breaks ties; no downstream outcomes influence selection.
    records.sort(key=lambda z:(z[0],z[1]));best=records[0];out=Path(a.output);out.mkdir(parents=True,exist_ok=False);shutil.copyfile(best[2]/'adapter.pt',out/'adapter.pt')
    meta_path=Path(a.cache).with_suffix('.manifest.json');metadata=json.loads(meta_path.read_text()) if meta_path.exists() else None
    if metadata and metadata['cache_sha256']!=cache_sha:raise ValueError('Cache manifest SHA mismatch')
    candidates=[]
    for i,v in enumerate(records):
        name='candidate_'+str(i)+'.pt';shutil.copyfile(v[2]/'adapter.pt',out/name)
        candidates.append({'lambda':v[1],'dev_supervised_sum':v[0],'sha256':v[3]['adapter_sha256'],'file':name})
    manifest={'selection_rule':'smallest DEV supervised_sum, lower lambda on ties','candidates':candidates, 'lambda_selected':best[1],'adapter_sha256':sha(out/'adapter.pt'),'train_dev_cache_sha256':cache_sha,'train_dev_mechanisms':sorted(set(data['mechanism_id'].astype(str))),'training_backbone_sha256':None if metadata is None else metadata['checkpoint_sha256'],'training_source':'unknown' if metadata is None else metadata['source'],'status':'frozen_before_calibration_and_test'}
    with (out/'freeze.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()

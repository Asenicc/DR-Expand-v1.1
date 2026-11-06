"""Non-overwriting, verified inheritance of the unaffected five warmup epochs."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from train_erm import ROOT,Model_thin,dump,file_sha,model_sha,now
from dre_v1p1_core import METHOD,CONFIG,MODEL_ARGS
from train_dre_v1p1 import preflight,read


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    preflight()
    old=ROOT/'runs/DRE_v1_N_seed0';new=ROOT/'runs/DRE_v1p1_N_seed0'
    if new.exists():raise FileExistsError(new)
    if read(ROOT/'runs/suite_dre_v1_seed0.json')['status']!='failed':raise RuntimeError('Expected preserved failed v1')
    ckpt=torch.load(old/'last.pt',map_location='cpu',weights_only=False)
    assert ckpt['epoch']==5 and ckpt['config']['method_config']==CONFIG
    with (old/'training.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    assert [int(r['epoch']) for r in rows]==[1,2,3,4,5]
    assert all(r['search_active']=='False' for r in rows)
    cfg=read(old/'config.json')
    assert cfg['source']=='N' and cfg['method_config']==CONFIG
    sha=file_sha(ROOT/'outputs/DRE_v1p1_seed0_protocol.json')
    cfg.update(method=METHOD,protocol_sha256=sha,
               inherited_warmup={'run':str(old),'epochs':[1,2,3,4,5],
                   'original_checkpoint_sha256':file_sha(old/'last.pt'),
                   'note':'No adversarial search occurred in these five completed epochs; failed partial epoch6 discarded'})
    new.mkdir()
    dump(new/'config.json',cfg)
    ckpt['config']=cfg;ckpt['protocol_sha256']=sha
    torch.save(ckpt,new/'last.pt')
    for name in ['training.csv','relation_weights.json']:
        shutil.copy2(old/name,new/name)
    saved=torch.load(new/'last.pt',map_location='cpu',weights_only=False)
    keys=['model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng',
          'loader_generator','search_rng','class_audit']
    equality={k:same(ckpt[k],saved[k]) for k in keys}
    assert all(equality.values())
    model=Model_thin(**MODEL_ARGS);model.load_state_dict(saved['model'])
    report={'passed':True,'inherited_epoch':5,'checked_at':now(),'source':'N',
            'origin':str(old),'original_checkpoint_sha256':file_sha(old/'last.pt'),
            'inherited_checkpoint_sha256':file_sha(new/'last.pt'),
            'inherited_model_sha256':model_sha(model),'exact_state_checks':equality,
            'inherited_csv_sha256':file_sha(new/'training.csv'),
            'original_config_sha256':file_sha(old/'config.json'),
            'new_config_sha256':file_sha(new/'config.json'),
            'target_data_opened':False,'old_files_modified':False}
    dump(new/'warmup_inheritance.json',report)
    dump(new/'progress.json',{'status':'inherited_warmup','source':'N','epoch':5,'epochs':50,'updated_at':now()})
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

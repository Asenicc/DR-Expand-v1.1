"""Exploratory DR-Expand v1.1; fixed protocol, source only, last-epoch targets."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset

from train_erm import ROOT,REPO,DATA,NAMES,Skeletons,Model_thin,dump,evaluate,file_sha,model_sha,now
from train_rar import build_recoverer,recovery_metadata
from dre_v1p1_core import (METHOD,MODEL_ARGS,CONFIG,fit_relation_weights,adversarial_expand,
                      supervised_contrastive,teacher_preservation)

FILES=['scripts/dre_v1p1_core.py','scripts/train_dre_v1p1.py','scripts/run_dre_v1p1_suite.py',
       'scripts/finalize_dre_v1p1_results.py','scripts/test_dre_v1p1.py','scripts/train_rar.py',
       'scripts/train_erm.py','scripts/finalize_results.py',
       'Recover-and-Resample/nets/agcn/agcn.py',
       'Recover-and-Resample/feeders/rotation_aug.py',
       'Recover-and-Resample/main/train_3domain_ours.py',
       'Recover-and-Resample/main/w_transforms.py',
       'Recover-and-Resample/main/linear_extrapolation.py']
MANIFEST=ROOT/'outputs/DRE_v1p1_seed0_protocol.json'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def preflight():
    p=read(MANIFEST)
    for rel,sha in p['files_sha256'].items():
        assert file_sha(ROOT/rel)==sha, f'Frozen code changed: {rel}'
    for source in ['N','EA']:
        for path,sha in p['teachers'][source]['files_sha256'].items():
            assert file_sha(path)==sha, f'Teacher changed: {path}'
    assert read(ROOT/'work/DRE_v1p1_checks.json')['passed']
    assert file_sha(ROOT/'work/DRE_v1p1_checks.json')==p['checks_sha256']
    for source in ['N','EA']:
        assert file_sha(ROOT/'work'/f'DRE_v1p1_smoke_{source}_seed0'/'smoke_result.json')==p['smoke_sha256'][source]
    return p


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',choices=['N','EA'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    out=a.output.resolve()
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise FileExistsError(f'Refusing to overwrite {out}')
    if a.smoke and a.resume:raise ValueError('No smoke resume')
    out.mkdir(parents=True,exist_ok=True)
    protocol=preflight() if not a.smoke else None
    teacher_dir=ROOT/'runs'/f'RAR_{a.source}_seed0'
    tc=read(teacher_dir/'config.json');tr=read(teacher_dir/'results.json')
    assert tc['source']==a.source and tc['epochs']==tr['epochs']==50 and tc['seed']==0
    assert tc['target_usage']=='final evaluation only; model.eval(); no adaptation'
    assert tr['selection']=='fixed final epoch; no target-based checkpoint selection'
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);np.random.seed(0);random.seed(0)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    model=Model_thin(**MODEL_ARGS).cuda()
    initial=model_sha(model)
    assert initial==read(teacher_dir/'environment.json')['initial_model_sha256']
    # Creating a frozen teacher must not perturb the student's augmentation RNG.
    with torch.random.fork_rng(devices=[0]):
        teacher=Model_thin(**MODEL_ARGS).cuda()
    checkpoint=torch.load(teacher_dir/'last.pt',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==50
    teacher.load_state_dict(checkpoint['model']);del checkpoint
    assert model_sha(teacher)==tr['checkpoint_model_sha256']
    teacher.eval().requires_grad_(False)
    teacher_sha=model_sha(teacher)
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,nesterov=True,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(0)
    search_rng=torch.Generator(device='cuda').manual_seed(91026)
    recovery=build_recoverer(a.source,out)
    domain='ntu' if a.source=='N' else 'etriA'
    ds=Skeletons(domain,'train',rotate=True)
    if a.smoke:
        chosen=np.concatenate([np.flatnonzero(ds.labels==k)[:8] for k in range(18)])
        train_ds=Subset(ds,chosen.tolist())
    else:train_ds=ds
    loader=DataLoader(train_ds,batch_size=64,shuffle=True,num_workers=0,pin_memory=True,
                      drop_last=True,generator=generator)
    config={'method':METHOD,'source':a.source,'seed':0,'smoke':a.smoke,'epochs':50,
            'batch_size':64,'drop_last':True,'source_samples':len(ds),
            'model':'unmodified author Model_thin; no inference branch or projection',
            'initial_model_sha256':initial,'inference_parameters':sum(p.numel() for p in model.parameters()),
            'optimizer':dict(lr=.1,momentum=.9,nesterov=True,weight_decay=1e-4),
            'schedule':dict(milestones=[30,40],gamma=.1),'method_config':CONFIG,
            'recovery':recovery_metadata(a.source),
            'teacher_checkpoint':str(teacher_dir/'last.pt'),
            'teacher_checkpoint_sha256':file_sha(teacher_dir/'last.pt'),
            'teacher_model_sha256':teacher_sha,'teacher_training_epochs':50,
            'precision':'FP32; AMP/TF32 disabled','selection':'fixed_final_epoch_50',
            'target_usage':'only final inference after source training; no target selection',
            'protocol_sha256':file_sha(MANIFEST) if protocol else None,
            'loss':'.5*(CE(original,y)+CE(expanded,y)) + .1*SupCon(two views) + .25*correct-teacher KL(T=2)',
            'status_note':'exploratory candidate; not validated novelty or identified content/style disentanglement',
            'started_at':now()}
    if a.resume:
        old=read(out/'config.json')
        assert all(old[k]==v for k,v in config.items() if k!='started_at')
        config=old
    else:dump(out/'config.json',config)
    dump(out/'environment.json',{'python':sys.version,'torch':torch.__version__,
        'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'pid':os.getpid(),
        'initial_model_sha256':initial,'parameters':config['inference_parameters']})
    history=[];start_epoch=0
    class_audit=np.zeros((18,4),np.float64)
    if a.resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==config['protocol_sha256']
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        generator.set_state(saved['loader_generator']);search_rng.set_state(saved['search_rng'])
        class_audit=np.asarray(saved['class_audit'],dtype=np.float64)
        start_epoch=saved['epoch'];del saved
        with (out/'training.csv').open(encoding='utf-8') as f:history=list(csv.DictReader(f))
        assert [int(r['epoch']) for r in history]==list(range(1,start_epoch+1)), 'Checkpoint/CSV mismatch'
        weights=torch.tensor(read(out/'relation_weights.json')['weights'],device='cuda',dtype=torch.float32)
    else:
        dump(out/'progress.json',{'status':'source_relation_statistics','source':a.source,
                                 'epoch':0,'epochs':50,'updated_at':now()})
        weights=fit_relation_weights(a.source,out/'relation_weights.json',a.smoke).cuda()
        config['relation_weights_sha256']=file_sha(out/'relation_weights.json')
        dump(out/'config.json',config)
    assert config['relation_weights_sha256']==file_sha(out/'relation_weights.json')
    start=time.monotonic()
    # Smoke deliberately exercises adversarial search AND one normal warmup epoch.
    epochs=range(2) if a.smoke else range(start_epoch,50)
    for epoch in epochs:
        adversarial=(epoch>=CONFIG['warmup_epochs']) or (a.smoke and epoch==1)
        model.train();begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
        lr=.1*(.1**sum(epoch>=m for m in [30,40]))
        for g in opt.param_groups:g['lr']=lr
        totals={k:0. for k in ['loss','ce_original','ce_expanded','supcon','teacher_kl',
                              'teacher_correct_fraction','relation_rms_mean','backtracked_fraction',
                              'shape_log_abs_mean','global_scale_mean','yaw_abs_degrees']}
        correct=0;seen=0;search_seconds=0.;max_error=0.
        for step,(original,y) in enumerate(loader,1):
            original=original.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            with torch.no_grad():
                recovered=recovery.apply_W_all(original,'ABL_linearextrap_beta','cropresize0.7')
                base=recovery.merge_raw_and_aug(original,recovered,.75)
            if not torch.isfinite(base).all():raise FloatingPointError('Non-finite RAR input')
            stat=dict(relation_rms_mean=0.,relation_rms_max=0.,backtracked_fraction=0.,
                      shape_log_abs_mean=0.,global_scale_mean=1.,yaw_abs_degrees=0.)
            if adversarial:
                stamp=time.monotonic()
                expanded,stat,params=adversarial_expand(model,base,y,weights,search_rng)
                search_seconds+=time.monotonic()-stamp
                max_error=max(max_error,stat['relation_rms_max'])
                if step==1:
                    np.savez_compressed(out/f'epoch_{epoch+1:03d}_search_sample.npz',
                        labels=y.cpu().numpy(),params=params.cpu().numpy())
            else:expanded=base
            with torch.no_grad():tlogits=teacher(original)
            opt.zero_grad(set_to_none=True)
            logits,features=model(torch.cat([original,expanded]),return_feat=True)
            z0,z1=logits.chunk(2);f0,f1=features.chunk(2)
            ce0=F.cross_entropy(z0,y);ce1=F.cross_entropy(z1,y)
            con=supervised_contrastive(torch.cat([f0,f1]),y.repeat(2),CONFIG['supcon_temperature'])
            kl,tcorr=teacher_preservation(z0,tlogits,y)
            loss=.5*(ce0+ce1)+CONFIG['supcon_weight']*con+CONFIG['teacher_weight']*kl
            if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss {epoch+1}/{step}')
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Non-finite training gradients')
            opt.step()
            n=len(y);seen+=n;correct+=int(z0.detach().argmax(1).eq(y).sum())
            metrics={'loss':float(loss.detach()),'ce_original':float(ce0.detach()),
                     'ce_expanded':float(ce1.detach()),'supcon':float(con.detach()),
                     'teacher_kl':float(kl.detach()),'teacher_correct_fraction':float(tcorr),**stat}
            for k in totals:totals[k]+=metrics[k]*n
            for k in range(18):
                take=y==k;nk=int(take.sum())
                class_audit[k]+=[nk,stat['relation_rms_mean']*nk,
                                 stat['backtracked_fraction']*nk,float(tlogits.argmax(1)[take].eq(y[take]).sum())]
            if step==1 or step%25==0 or step==len(loader):
                print('STEP',json.dumps({'epoch':epoch+1,'step':step,'steps':len(loader),
                                         'loss':metrics['loss'],'adversarial':adversarial,
                                         'relation_rms':stat['relation_rms_mean']}),flush=True)
                dump(out/'progress.json',{'status':'smoke' if a.smoke else 'training','method':METHOD,
                    'source':a.source,'epoch':epoch+1,'epochs':50,'step':step,
                    'steps_per_epoch':len(loader),'adversarial_active':adversarial,
                    'updated_at':now(),'loss':metrics['loss']})
            if a.smoke and step>=2:break
        torch.cuda.synchronize()
        row={'epoch':epoch+1,'learning_rate':lr,**{k:v/seen for k,v in totals.items()},
             'training_accuracy_percent':correct/seen*100,'samples_seen':seen,'updates':step,
             'search_active':adversarial,'relation_rms_max':max_error,
             'seconds':time.monotonic()-begin,'search_seconds':search_seconds,
             'peak_gpu_allocated_gb':torch.cuda.max_memory_allocated()/2**30}
        history.append(row)
        with (out/'training.csv').open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if epoch==0:writer.writeheader()
            writer.writerow(row)
        print('EPOCH_END',json.dumps(row),flush=True)
        if not a.smoke:
            saved={'epoch':epoch+1,'model':model.state_dict(),'optimizer':opt.state_dict(),
                   'protocol_sha256':config['protocol_sha256'],'config':config,
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                   'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                   'loader_generator':generator.get_state(),'search_rng':search_rng.get_state(),
                   'class_audit':class_audit.tolist()}
            temp=out/'last.pt.tmp';torch.save(saved,temp);temp.replace(out/'last.pt')
            if (epoch+1)%10==0:torch.save(saved,out/f'epoch_{epoch+1:03d}.pt')
        dump(out/'source_training_audit.json',{'columns':['sample_exposures','batch_mean_relation_rms_sum',
            'batch_backtrack_fraction_sum','teacher_correct_exposures'],'classes':NAMES,
            'data':class_audit.tolist(),'note':'relation/backtracking numbers are batch means, not per-class estimates'})
    assert model_sha(teacher)==teacher_sha,'Frozen teacher changed'
    if a.smoke:
        assert history[-1]['shape_log_abs_mean']>0 and history[-1]['relation_rms_max']<=.08001
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
        dump(out/'smoke_result.json',{'passed':True,'source':a.source,'smoke':True,
             'target_data_opened':False,'teacher_unchanged':True,'epochs_exercised':2,
             'history':history,'max_gpu_gb':max(r['peak_gpu_allocated_gb'] for r in history)})
        print('SMOKE_PASSED',flush=True);return
    digest=model_sha(model)
    dump(out/'progress.json',{'status':'final_evaluation','source':a.source,'epoch':50,'epochs':50,'updated_at':now()})
    domains=['etri','pku'] if a.source=='N' else ['ntu','pku']
    metrics={d:evaluate(model,d,out,64,0) for d in domains}
    assert model_sha(model)==digest
    result={'method':METHOD,'source':a.source,'seed':0,'epochs':50,'smoke':False,
            'checkpoint':'last.pt','checkpoint_model_sha256':digest,'selection':'fixed_final_epoch_50',
            'metrics':metrics,'training_seconds':sum(float(r['seconds']) for r in history),
            'search_seconds':sum(float(r['search_seconds']) for r in history),
            'classifier_updates':sum(int(r['updates']) for r in history),
            'teacher_pretraining_epochs':50,'total_seconds_this_process':time.monotonic()-start,
            'completed_at':now()}
    dump(out/'results.json',result)
    dump(out/'progress.json',{'status':'completed','source':a.source,'epoch':50,'epochs':50,'updated_at':now()})
    print('RUN_COMPLETED',json.dumps(result),flush=True)


if __name__=='__main__':main()

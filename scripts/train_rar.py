"""Author Recover-and-Resample augmentation with the established ERM protocol.

The original Processor recovery methods and source-specific published clusters
are used directly; its target-based model selection and I/O are not invoked.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_erm import (
    ROOT, REPO, DATA, NAMES, Skeletons, dump, evaluate, file_sha,
    model_sha, now, seed_worker, Model_thin,
)

sys.path.insert(0, str(REPO / 'main'))
from train_3domain_ours import Processor

AUTHOR_FILES = [
    'main/train_3domain_ours.py', 'main/w_transforms.py',
    'main/linear_extrapolation.py', 'scripts/train_3domain_ours.sh',
    'configs/dataset_base_priornet.yaml', 'nets/agcn/agcn.py',
    'feeders/rotation_aug.py',
]
CLUSTER_DIR = REPO / 'data/cluster_result'


def recovery_metadata(source):
    domain = 'ntu' if source == 'N' else 'etriA'
    files = [CLUSTER_DIR / f'bkg_cluster_poses_{domain}_10.npy',
             CLUSTER_DIR / f'tr_cluster_{domain}_20_T0.1.npy.npz']
    return {
        'implementation': 'unmodified author Processor methods; no Processor constructor',
        'recover_tag': 'ABL_linearextrap_beta',
        'resample_tag': 'cropresize0.7',
        'aug_ratio': 0.75, 'boundary_clusters': 10, 'transform_clusters': 20,
        'extrapolation_beta': [0.1, 0.1], 'crop_ratio': [0.7, 1.0],
        'transform_sampling': 'uniform, as author f2_linear(..., None)',
        'cluster_source': domain,
        'cluster_provenance': 'author-published source-train clusters at reference commit; not refitted',
        'target_clusters_used': False,
        'cluster_files_sha256': {str(p.relative_to(REPO)): file_sha(p) for p in files},
    }


def build_recoverer(source, output):
    domain = 'ntu' if source == 'N' else 'etriA'
    obj = Processor.__new__(Processor)
    obj.arg = SimpleNamespace(
        prior_data_npy_name=str(DATA/domain/'train_data_joint.npy'),
        n_cluster_bkg=10, n_cluster_w=20, bkg_type='train', tr_type='train',
        train_feeder_name=domain, cluster_dir=str(CLUSTER_DIR),
        base_dir=str(output), work_dir=str(output), print_log=False,
    )
    obj.output_device = torch.device('cuda:0')
    obj.load_clustering()
    assert tuple(obj.bkg_pose_list.shape) == (10, 3, 1, 25, 2)
    assert obj.traj_mat_list.shape == (20, 64, 64)
    assert np.isfinite(obj.traj_mat_list).all()
    assert np.array_equal(obj.traj_mat_list.sum(axis=2), np.ones((20,64)))
    assert torch.isfinite(obj.bkg_pose_list).all()
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['N', 'EA'], default='N')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.epochs == 50 and args.batch_size == 64
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite an existing experiment: {out}')
    out.mkdir(parents=True, exist_ok=True)
    config = {
        'method':'Recover-and-Resample', 'source':args.source, 'seed':args.seed,
        'epochs':args.epochs, 'batch_size':args.batch_size, 'workers':args.workers,
        'model':'nets.agcn.agcn.Model_thin',
        'initialization':'random; no pretrained checkpoint',
        'input_shape':[3,64,25,2], 'classes':NAMES,
        'optimizer':{'name':'SGD','learning_rate':0.1,'momentum':0.9,
                     'nesterov':True,'weight_decay':0.0001},
        'schedule':{'milestones':[30,40],'gamma':0.1,'warmup_epochs':0},
        'augmentation':'original random rotation(theta=30) followed by original Recover-and-Resample',
        'recovery':recovery_metadata(args.source),
        'drop_last':True, 'precision':'float32; AMP and TF32 disabled',
        'selection':'fixed final epoch; no target-based checkpoint selection',
        'target_usage':'final evaluation only; model.eval(); no adaptation',
        'reference_commit':subprocess.check_output(
            ['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        'runner_sha256':file_sha(__file__),
        'baseline_runner_sha256':file_sha(ROOT/'scripts/train_erm.py'),
        'author_files_sha256':{p:file_sha(REPO/p) for p in AUTHOR_FILES},
        'data_audit_sha256':file_sha(ROOT/'reports/data_audit.json'),
        'etria_data_audit_sha256':file_sha(ROOT/'reports/etria_data_audit.json') if args.source=='EA' else None,
        'smoke_steps':args.smoke_steps, 'started_at':now(),
    }
    assert config['reference_commit'] == 'fb4d621c14f088c3bbac4031faef9cc0a6caf1cd'
    dump(out/'config.json',config)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    model = Model_thin(num_class=18,num_point=25,num_person=2,
                       graph='nets.agcn.graph.ntu_rgb_d.Graph',
                       graph_args={'labeling_mode':'spatial'}).cuda()
    initial_model_sha = model_sha(model)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,
                                nesterov=True,weight_decay=0.0001)
    recoverer = build_recoverer(args.source,out)
    env = {
        'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'torch':torch.__version__,'numpy':np.__version__,'cuda_runtime':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0),
        'compute_capability':list(torch.cuda.get_device_capability(0)),
        'parameters':sum(p.numel() for p in model.parameters()),
        'driver':subprocess.check_output(
            ['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        'initial_model_sha256':initial_model_sha,
    }
    dump(out/'environment.json',env)
    print('ENVIRONMENT',json.dumps(env),flush=True)
    source_domain = 'ntu' if args.source == 'N' else 'etriA'
    dataset = Skeletons(source_domain,'train',rotate=True)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,
                        num_workers=args.workers,pin_memory=True,drop_last=True,
                        worker_init_fn=seed_worker,generator=generator,
                        persistent_workers=args.workers>0)
    print('TRAINING_START',args.source,'samples',len(dataset),'batches',len(loader),
          'epochs',args.epochs,'smoke_steps',args.smoke_steps,flush=True)
    start = time.monotonic()
    history = []
    total_epochs = 1 if args.smoke_steps else args.epochs
    for epoch in range(total_epochs):
        model.train()
        epoch_start = time.monotonic()
        lr = 0.1*(0.1**sum(epoch>=milestone for milestone in (30,40)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        losses = torch.zeros((),device='cuda')
        correct = torch.zeros((),device='cuda',dtype=torch.int64)
        seen = 0
        torch.cuda.reset_peak_memory_stats()
        for step,(x,y) in enumerate(loader,1):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with torch.no_grad():
                recovered = recoverer.apply_W_all(x,'ABL_linearextrap_beta','cropresize0.7')
                x = recoverer.merge_raw_and_aug(x,recovered,0.75)
            if not torch.isfinite(x).all():
                raise FloatingPointError(f'Non-finite augmented inputs: {epoch+1}, {step}')
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss: {epoch+1}, {step}')
            loss.backward()
            optimizer.step()
            losses += loss.detach()*len(y)
            correct += (logits.detach().argmax(1)==y).sum()
            seen += len(y)
            if step==1 or step%50==0:
                value = loss.item()
                print(f'EPOCH {epoch+1:02d}/{total_epochs} STEP {step:03d}/{len(loader)} '
                      f'loss={value:.5f} elapsed={time.monotonic()-epoch_start:.1f}s',flush=True)
                dump(out/'progress.json',{'status':'smoke' if args.smoke_steps else 'training',
                     'method':config['method'],'source':args.source,'epoch':epoch+1,
                     'epochs':total_epochs,'step':step,'steps_per_epoch':len(loader),
                     'loss':value,'updated_at':now()})
            if args.smoke_steps and step>=args.smoke_steps:
                break
        torch.cuda.synchronize()
        row = {'epoch':epoch+1,'learning_rate':lr,'loss':losses.item()/seen,
               'training_accuracy_percent':correct.item()/seen*100,
               'seconds':time.monotonic()-epoch_start,
               'peak_gpu_allocated_gb':torch.cuda.max_memory_allocated()/2**30}
        history.append(row)
        with (out/'training.csv').open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if epoch==0:writer.writeheader()
            writer.writerow(row)
        print('EPOCH_END',json.dumps(row),flush=True)
        if args.smoke_steps:
            dump(out/'smoke_result.json',{'passed':True,'steps':args.smoke_steps,
                 'finite_loss':bool(np.isfinite(row['loss'])),
                 'last_epoch':row,'environment':env,'recovery':config['recovery']})
            print('SMOKE_PASSED',flush=True)
            return
        checkpoint = {'epoch':epoch+1,'model':model.state_dict(),
                      'optimizer':optimizer.state_dict(),'config':config,
                      'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                      'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                      'loader_generator':generator.get_state()}
        temp=out/'last.pt.tmp'
        torch.save(checkpoint,temp)
        temp.replace(out/'last.pt')
        if (epoch+1)%10==0:
            torch.save(checkpoint,out/f'epoch_{epoch+1:03d}.pt')
        dump(out/'progress.json',{'status':'training','method':config['method'],
             'source':args.source,'epoch':epoch+1,'epochs':total_epochs,'last_epoch':row,
             'elapsed_seconds':time.monotonic()-start,'updated_at':now()})
    del loader
    digest=model_sha(model)
    dump(out/'progress.json',{'status':'final_evaluation','source':args.source,
                             'epoch':total_epochs,'updated_at':now()})
    domains = ['ntu','etri','pku'] if args.source=='N' else ['ntu','pku']
    metrics={d:evaluate(model,d,out,args.batch_size,args.workers) for d in domains}
    assert model_sha(model)==digest,'Model changed during target evaluation'
    result={'method':config['method'],'source':args.source,'seed':args.seed,'epochs':args.epochs,
            'checkpoint':'last.pt','checkpoint_model_sha256':digest,
            'selection':config['selection'],'single_seed':True,'metrics':metrics,
            'training_seconds':sum(row['seconds'] for row in history),
            'total_seconds':time.monotonic()-start,'completed_at':now(),
            'paper_comparability':'One seed, fixed final checkpoint, matching local ERM; '
                                  'differs from paper target-test best checkpoint / 10-seed mean.'}
    dump(out/'results.json',result)
    dump(out/'progress.json',{'status':'completed','source':args.source,'epochs':args.epochs,
                             'results':str(out/'results.json'),'updated_at':now()})
    print('RUN_COMPLETED',json.dumps(result),flush=True)


if __name__=='__main__':
    main()

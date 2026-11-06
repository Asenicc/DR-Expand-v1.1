"""Source-only Common-18 ERM with the unmodified author's thin AGCN.

Training: 50 epochs, SGD(0.1, momentum=.9, Nesterov, weight_decay=1e-4),
LR drops at epochs 30/40, batch 64, original random rotation, no warmup.
Evaluation: fixed last checkpoint, targets opened only after training finishes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'Recover-and-Resample'
DATA = ROOT / 'data/common18/xsub64'
sys.path.insert(0, str(REPO))
from nets.agcn.agcn import Model_thin
from feeders.rotation_aug import my_random_rot

NAMES = ['eat', 'drink', 'brush_teeth', 'brush_hair', 'wear_clothes',
         'takeoff_clothes', 'glasses_on_off', 'read', 'write', 'phone_call',
         'play_phone', 'clap', 'bow', 'handshake', 'hug', 'handwave',
         'point_finger', 'fall_down']


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


class Skeletons(Dataset):
    def __init__(self, domain, split, rotate=False):
        self.path = DATA / domain / f'{split}_data_joint.npy'
        with (DATA/domain/f'{split}_label.pkl').open('rb') as f:
            self.samples, labels = pickle.load(f)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.array = None
        self.rotate = rotate
        header = np.load(self.path, mmap_mode='r')
        assert header.shape == (len(labels), 3, 64, 25, 2)
        assert header.dtype == np.float32
        assert set(labels) == set(range(18))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode='r')
        x = np.array(self.array[index], copy=True)
        if self.rotate:
            x = my_random_rot(x, 30).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.labels[index])

    def __getstate__(self):
        state = self.__dict__.copy()
        state['array'] = None
        return state


def evaluate(model, domain, output, batch_size, workers):
    dataset = Skeletons(domain, 'test', rotate=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    logits = []
    loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            scores = model(x)
            loss_sum += nn.functional.cross_entropy(scores, y, reduction='sum').item()
            logits.append(scores.cpu().numpy())
    scores = np.concatenate(logits)
    prediction = scores.argmax(1)
    truth = dataset.labels
    matrix = np.bincount(truth*18+prediction, minlength=18*18).reshape(18, 18)
    counts = matrix.sum(1)
    recalls = matrix.diagonal()/counts
    result = {'domain': domain, 'samples': len(truth),
              'accuracy_percent': float((prediction == truth).mean()*100),
              'balanced_accuracy_percent': float(recalls.mean()*100),
              'cross_entropy': loss_sum/len(truth),
              'class_counts': counts.tolist(),
              'per_class_recall_percent': dict(zip(NAMES, (recalls*100).tolist())),
              'confusion_matrix': matrix.tolist()}
    np.savez_compressed(output/f'{domain}_predictions.npz', logits=scores,
                        predictions=prediction, labels=truth,
                        samples=np.asarray(dataset.samples))
    dump(output/f'{domain}_metrics.json', result)
    print('EVALUATION', json.dumps({k:result[k] for k in ('domain', 'samples', 'accuracy_percent', 'balanced_accuracy_percent')}), flush=True)
    return result


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
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite an existing experiment: {out}')
    out.mkdir(parents=True, exist_ok=True)
    config = {'method': 'ERM', 'source': args.source, 'seed': args.seed,
              'epochs': args.epochs, 'batch_size': args.batch_size,
              'workers': args.workers, 'model': 'nets.agcn.agcn.Model_thin',
              'initialization': 'random; no pretrained checkpoint',
              'input_shape': [3,64,25,2], 'classes': NAMES,
              'optimizer': {'name':'SGD','learning_rate':0.1,'momentum':0.9,
                            'nesterov':True,'weight_decay':0.0001},
              'schedule': {'milestones':[30,40],'gamma':0.1,'warmup_epochs':0},
              'augmentation': 'original feeders.rotation_aug.my_random_rot(theta_val=30)',
              'drop_last': True, 'precision': 'float32; AMP and TF32 disabled',
              'selection': 'fixed final epoch; no target-based checkpoint selection',
              'target_usage': 'final evaluation only; model.eval(); no adaptation',
              'reference_commit': subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
              'runner_sha256': file_sha(__file__),
              'data_audit_sha256': file_sha(ROOT/'reports/data_audit.json'),
              'etria_data_audit_sha256': file_sha(ROOT/'reports/etria_data_audit.json') if args.source=='EA' else None,
              'smoke_steps': args.smoke_steps, 'started_at':now()}
    dump(out/'config.json', config)
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                nesterov=True, weight_decay=0.0001)
    env = {'python':sys.version, 'executable':sys.executable,
           'platform':platform.platform(),'torch':torch.__version__,
           'numpy':np.__version__,'cuda_runtime':torch.version.cuda,
           'cudnn':torch.backends.cudnn.version(),
           'gpu':torch.cuda.get_device_name(0),
           'compute_capability':list(torch.cuda.get_device_capability(0)),
           'parameters':sum(p.numel() for p in model.parameters()),
           'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()}
    dump(out/'environment.json', env)
    print('ENVIRONMENT',json.dumps(env),flush=True)
    source_domain = 'ntu' if args.source == 'N' else 'etriA'
    dataset = Skeletons(source_domain,'train',rotate=True)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,
                        num_workers=args.workers,pin_memory=True,drop_last=True,
                        worker_init_fn=seed_worker,generator=generator,
                        persistent_workers=args.workers>0)
    print('TRAINING_START',args.source,'samples',len(dataset),'batches',len(loader),'epochs',args.epochs,'smoke_steps',args.smoke_steps,flush=True)
    start = time.monotonic()
    history = []
    csv_file = out/'training.csv'
    total_epochs = 1 if args.smoke_steps else args.epochs
    for epoch in range(total_epochs):
        model.train()
        epoch_start = time.monotonic()
        lr = 0.1 * (0.1 ** sum(epoch >= milestone for milestone in (30,40)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        losses = torch.zeros((),device='cuda')
        correct = torch.zeros((),device='cuda',dtype=torch.int64)
        seen = 0
        torch.cuda.reset_peak_memory_stats()
        for step,(x,y) in enumerate(loader,1):
            x=x.cuda(non_blocking=True)
            y=y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss=nn.functional.cross_entropy(logits,y)
            loss.backward()
            optimizer.step()
            losses += loss.detach()*len(y)
            correct += (logits.detach().argmax(1)==y).sum()
            seen += len(y)
            if step == 1 or step%50==0:
                value=loss.item()
                if not np.isfinite(value):
                    raise FloatingPointError(f'Non-finite loss at epoch={epoch+1}, step={step}')
                print(f'EPOCH {epoch+1:02d}/{total_epochs} STEP {step:03d}/{len(loader)} loss={value:.5f} elapsed={time.monotonic()-epoch_start:.1f}s',flush=True)
                dump(out/'progress.json',{'status':'smoke' if args.smoke_steps else 'training',
                      'source':args.source,'epoch':epoch+1,'epochs':total_epochs,
                      'step':step,'steps_per_epoch':len(loader),'loss':value,'updated_at':now()})
            if args.smoke_steps and step>=args.smoke_steps:
                break
        torch.cuda.synchronize()
        row={'epoch':epoch+1,'learning_rate':lr,'loss':losses.item()/seen,
             'training_accuracy_percent':correct.item()/seen*100,
             'seconds':time.monotonic()-epoch_start,
             'peak_gpu_allocated_gb':torch.cuda.max_memory_allocated()/2**30}
        history.append(row)
        with csv_file.open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if epoch==0:writer.writeheader()
            writer.writerow(row)
        print('EPOCH_END',json.dumps(row),flush=True)
        if args.smoke_steps:
            dump(out/'smoke_result.json',{'passed':True,'steps':args.smoke_steps,
                 'finite_loss':bool(np.isfinite(row['loss'])),'last_epoch':row,'environment':env})
            print('SMOKE_PASSED',flush=True)
            return
        checkpoint={'epoch':epoch+1,'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'config':config,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                    'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                    'loader_generator':generator.get_state()}
        temp=out/'last.pt.tmp'
        torch.save(checkpoint,temp)
        temp.replace(out/'last.pt')
        if (epoch+1)%10==0:
            torch.save(checkpoint,out/f'epoch_{epoch+1:03d}.pt')
        dump(out/'progress.json',{'status':'training','source':args.source,
             'epoch':epoch+1,'epochs':total_epochs,'last_epoch':row,
             'elapsed_seconds':time.monotonic()-start,'updated_at':now()})
    # Destroy source workers before final evaluation; target loaders start here.
    del loader
    digest=model_sha(model)
    dump(out/'progress.json',{'status':'final_evaluation','source':args.source,
                             'epoch':total_epochs,'updated_at':now()})
    domains = ['ntu','etri','pku'] if args.source=='N' else ['ntu','pku']
    metrics={domain:evaluate(model,domain,out,args.batch_size,args.workers) for domain in domains}
    assert model_sha(model)==digest,'Model changed during target evaluation'
    result={'method':'ERM','source':args.source,'seed':args.seed,'epochs':args.epochs,
            'checkpoint':'last.pt','checkpoint_model_sha256':digest,
            'selection':config['selection'],'single_seed':True,
            'metrics':metrics,'training_seconds':sum(row['seconds'] for row in history),
            'total_seconds':time.monotonic()-start,'completed_at':now(),
            'paper_comparability':'Training model/settings aligned; fixed final checkpoint differs from paper target-test best checkpoint, and this is one seed rather than a 10-seed mean.'}
    dump(out/'results.json',result)
    dump(out/'progress.json',{'status':'completed','source':args.source,'epochs':args.epochs,
                             'results':str(out/'results.json'),'updated_at':now()})
    print('RUN_COMPLETED',json.dumps(result),flush=True)


if __name__=='__main__':
    main()

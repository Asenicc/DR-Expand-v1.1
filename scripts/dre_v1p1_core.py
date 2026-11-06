"""DR-Expand v1.1. Source-fitted relation constraints and kinematic domain search.

The inference model is the unmodified author AGCN. This module is training-only.
No target data or target-derived class weights are used.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_erm import DATA, NAMES, ROOT, Skeletons, dump, file_sha, now

METHOD = 'DR-Expand-v1.1'
MODEL_ARGS = dict(num_class=18, num_point=25, num_person=2,
                  graph='nets.agcn.graph.ntu_rgb_d.Graph',
                  graph_args={'labeling_mode': 'spatial'})
# Root is the centered spine-mid joint, not an unobserved zero-valued joint.
PARENTS = {0:1,20:1,2:20,3:2,4:20,5:4,6:5,7:6,22:7,21:22,
           8:20,9:8,10:9,11:10,24:11,23:24,12:0,13:12,14:13,15:14,
           16:0,17:16,18:17,19:18}
ORDER = [1,0,20,2,3,4,5,6,7,22,21,8,9,10,11,24,23,12,13,14,15,16,17,18,19]
# Parameters 0/1 are global log-scale/yaw. 2..9 are log bone-group scales.
GROUP = {0:2,20:2,2:3,3:3,4:2,5:4,6:5,7:6,22:6,21:6,
         8:2,9:4,10:5,11:6,24:6,23:6,12:2,13:7,14:8,15:9,
         16:2,17:7,18:8,19:9}
ANCHORS = [0,2,3,6,7,10,11,15,19,20]
PAIRS = list(itertools.combinations(ANCHORS, 2))
CROSS = [(7,7),(7,11),(11,7),(11,11),(0,0),(3,3)]
RELATION_NAMES = [f'intra_{i}_{j}' for i,j in PAIRS] + [f'inter_{i}_{j}' for i,j in CROSS]
BOUNDS = [float(np.log(1.15)), float(np.deg2rad(15))] + [float(np.log(1.10))]*8
LOWER = [float(np.log(.85)), -BOUNDS[1]] + [float(np.log(.90))]*8
CONFIG = dict(warmup_epochs=5, search_steps=2, search_step_normalized=.5,
              relation_penalty=2., relation_max_relative_rms=.08,
              backtracking_steps=8, supcon_weight=.1, supcon_temperature=.1,
              teacher_weight=.25, teacher_temperature=2.,
              teacher_mask='correct on the original rotated source view; no confidence threshold',
              bounds_lower=LOWER, bounds_upper=BOUNDS)


def kinematic_transform(x, p):
    """Constant-per-sequence bone-group scaling, common to both actors.

    Bone directions and per-bone temporal length ratios are preserved before
    common rigid rotation/scale. Raw input tracking noise is not repaired.
    """
    joints = {1:x[:,:,:,1,:]}
    for j in ORDER[1:]:
        parent = PARENTS[j]
        factor = p[:,GROUP[j]].exp()[:,None,None,None]
        joints[j] = joints[parent] + factor*(x[:,:,:,j,:]-x[:,:,:,parent,:])
    z = torch.stack([joints[j] for j in range(25)], dim=3)
    c, s = p[:,1].cos()[:,None,None,None], p[:,1].sin()[:,None,None,None]
    z = torch.stack([c*z[:,0]-s*z[:,1],s*z[:,0]+c*z[:,1],z[:,2]],dim=1)
    z = z * p[:,0].exp()[:,None,None,None,None]
    active = (x.abs().sum((1,3))>0)  # B,T,M; never invent an actor/frame
    return z * active[:,None,:,None,:]


def relations(x):
    """51 scale/rigid-rotation invariant distance trajectories and validity."""
    active = (x.abs().sum((1,3))>0).to(x.dtype)
    torso = (x[:,:,:,20,:]-x[:,:,:,0,:]).norm(dim=1)
    valid_torso = active*(torso>1e-5)
    scale = (torso*valid_torso).sum((1,2))/valid_torso.sum((1,2)).clamp_min(1)
    scale = scale.clamp_min(1e-4)
    a,b = zip(*PAIRS)
    d = (x[:,:,:,list(a),:]-x[:,:,:,list(b),:]).norm(dim=1)
    d = (d*active[:,:,None,:]).sum(-1)/active.sum(-1).clamp_min(1)[:,:,None]
    intra = d.permute(0,2,1)/scale[:,None,None]
    mask_intra = (active.sum(-1)>0)[:,None,:].expand_as(intra)
    a,b = zip(*CROSS)
    cross = (x[:,:,:,list(a),0]-x[:,:,:,list(b),1]).norm(dim=1).permute(0,2,1)
    cross = cross/scale[:,None,None]
    mask_cross = (active[:,:,0]*active[:,:,1]>0)[:,None,:].expand_as(cross)
    return torch.cat([intra,cross],1),torch.cat([mask_intra,mask_cross],1)


def relation_error_squared(candidate, reference, valid, weights):
    """Squared relative trajectory RMS, differentiable at zero residual.

    Evaluating sqrt(q).square() has the same real value but an undefined sqrt
    derivative at q=0 in autograd. The optimizer must use q directly.
    """
    mask = valid.to(candidate.dtype)
    delta = candidate-reference
    dd = delta[:,:,1:]-delta[:,:,:-1]
    vv = mask[:,:,1:]*mask[:,:,:-1]
    energy = ((reference.square()+.01)*mask).sum(-1)
    err = (delta.square()*mask).sum(-1) + .25*(dd.square()*vv).sum(-1)
    channel = err/energy.clamp_min(1e-5)
    active = (mask.sum(-1)>0).to(candidate.dtype)
    return ((channel*weights*active).sum(-1)/(weights*active).sum(-1).clamp_min(1)).clamp_min(0)


def relation_error(candidate, reference, valid, weights):
    """RMS for reporting/feasibility checks; not squared again in optimization."""
    return relation_error_squared(candidate, reference, valid, weights).sqrt()


def fit_relation_weights(source, output, smoke=False):
    """Class-vs-rest separation / within-class variation, source only.

    This is a statistical weighting heuristic, not identified causal factors or
    a learned content/style disentanglement network.
    """
    domain = 'ntu' if source=='N' else 'etriA'
    ds = Skeletons(domain,'train',rotate=False)
    idx = np.arange(len(ds))
    if smoke:
        idx = np.concatenate([np.flatnonzero(ds.labels==k)[:3] for k in range(18)])
    sums = np.zeros((18,51,3)); sq = sums.copy(); counts = np.zeros((18,51))
    arr = np.load(ds.path,mmap_mode='r')
    with torch.inference_mode():
        for start in range(0,len(idx),128):
            rows = idx[start:start+128]
            x = torch.from_numpy(np.array(arr[rows],copy=True))
            r,m = relations(x)
            mf = m.float(); n = mf.sum(-1).clamp_min(1)
            mean = (r*mf).sum(-1)/n
            std = (((r-mean[:,:,None]).square()*mf).sum(-1)/n).sqrt()
            velmask=mf[:,:,1:]*mf[:,:,:-1]
            speed=((r[:,:,1:]-r[:,:,:-1]).abs()*velmask).sum(-1)/velmask.sum(-1).clamp_min(1)
            features=torch.stack([mean,std,speed],-1).numpy()
            ok=(m.sum(-1)>0).numpy(); labels=ds.labels[rows]
            for k in range(18):
                take=labels==k
                sums[k]+=(features[take]*ok[take,:,None]).sum(0)
                sq[k]+=(features[take]**2*ok[take,:,None]).sum(0)
                counts[k]+=ok[take].sum(0)
    means=sums/np.maximum(counts[:,:,None],1)
    variance=np.maximum(sq/np.maximum(counts[:,:,None],1)-means**2,0)
    scores=np.zeros((18,51))
    for k in range(18):
        rest_n=counts.sum(0)-counts[k]
        rest_mean=(sums.sum(0)-sums[k])/np.maximum(rest_n[:,None],1)
        rest_var=np.maximum((sq.sum(0)-sq[k])/np.maximum(rest_n[:,None],1)-rest_mean**2,0)
        contrast=(means[k]-rest_mean)**2/(variance[k]+rest_var+1e-3)
        scores[k]=np.minimum(contrast,5).mean(-1)*(counts[k]>=2)*(rest_n>=2)
    weights=(.25+scores)
    weights/=weights.mean(1,keepdims=True)
    payload={'source':source,'smoke':smoke,'fitted_rows':len(idx),'classes':NAMES,
             'weights':weights.tolist(),'relation_names':RELATION_NAMES,
             'valid_sequence_counts':counts.astype(int).tolist(),
             'source_labels_sha256':file_sha(DATA/domain/'train_label.pkl'),
             'source_data':str(ds.path),'target_data_used':False,'created_at':now(),
             'rule':'mean/std/absolute-velocity class-vs-rest Fisher heuristic; floor .25; score cap 5'}
    dump(output,payload)
    return torch.tensor(weights,dtype=torch.float32)


def adversarial_expand(model,x,y,class_weights,generator,steps=2):
    b=len(x); device=x.device
    lo=x.new_tensor(LOWER); hi=x.new_tensor(BOUNDS)
    p=(torch.rand((b,10),device=device,generator=generator)*2-1)*.10
    # Normalized parameters [-1,1], with an identity origin and asymmetric bounds.
    def physical(u):
        return torch.where(u>=0,u*hi,-u*lo)
    with torch.no_grad():
        reference,valid=relations(x)
    w=class_weights[y]
    previous=model.training
    model.eval()  # Inner search must not contaminate training BN statistics.
    flags=[v.requires_grad for v in model.parameters()]
    for v in model.parameters():v.requires_grad_(False)
    try:
        for _ in range(steps):
            p=p.detach().requires_grad_(True)
            xp=kinematic_transform(x,physical(p))
            r,_=relations(xp)
            err_squared=relation_error_squared(r,reference,valid,w)
            objective=F.cross_entropy(model(xp),y,reduction='none')-CONFIG['relation_penalty']*err_squared
            grad,=torch.autograd.grad(objective.sum(),p)
            if not torch.isfinite(grad).all():raise FloatingPointError('Non-finite kinematic search gradient')
            p=(p+CONFIG['search_step_normalized']*grad.sign()).clamp(-1,1)
        with torch.no_grad():
            before=None
            for attempt in range(CONFIG['backtracking_steps']+1):
                xp=kinematic_transform(x,physical(p))
                r,_=relations(xp)
                err=relation_error(r,reference,valid,w)
                if before is None:before=err.clone()
                bad=err>CONFIG['relation_max_relative_rms']
                if not bad.any():break
                # Preserve the adversarial rigid/global factors; backtrack shape only.
                p[:,2:]=torch.where(bad[:,None],p[:,2:]*.5,p[:,2:])
            # Numerical safety fallback for the rare still-infeasible shape.
            if bad.any():
                p[bad,2:]=0
                xp=kinematic_transform(x,physical(p))
                r,_=relations(xp);err=relation_error(r,reference,valid,w)
            if not torch.isfinite(xp).all() or (err>CONFIG['relation_max_relative_rms']+1e-5).any():
                raise FloatingPointError('Relation-feasibility check failed')
            params=physical(p)
            stats={'relation_rms_mean':float(err.mean()),'relation_rms_max':float(err.max()),
                   'backtracked_fraction':float((before>CONFIG['relation_max_relative_rms']).float().mean()),
                   'shape_log_abs_mean':float(params[:,2:].abs().mean()),
                   'global_scale_mean':float(params[:,0].exp().mean()),
                   'yaw_abs_degrees':float(params[:,1].abs().mean()*180/np.pi)}
            return xp.detach(),stats,params.detach()
    finally:
        for v,flag in zip(model.parameters(),flags):v.requires_grad_(flag)
        model.train(previous)


def supervised_contrastive(features,labels,temperature=.1):
    f=F.normalize(features,dim=1)
    logits=f@f.T/temperature
    diagonal=torch.eye(len(f),dtype=torch.bool,device=f.device)
    positive=labels[:,None].eq(labels[None,:])&~diagonal
    logits=logits-logits.max(1,keepdim=True).values.detach()
    logprob=logits-torch.logsumexp(logits.masked_fill(diagonal,-torch.inf),1,keepdim=True)
    return -((logprob.masked_fill(~positive,0)).sum(1)/positive.sum(1).clamp_min(1)).mean()


def teacher_preservation(student_logits,teacher_logits,labels):
    mask=teacher_logits.argmax(1).eq(labels)
    t=CONFIG['teacher_temperature']
    loss=F.kl_div(F.log_softmax(student_logits/t,1),F.softmax(teacher_logits/t,1),reduction='none').sum(1)*t*t
    return (loss*mask).sum()/mask.sum().clamp_min(1),mask.float().mean()

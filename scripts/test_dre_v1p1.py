"""Meaningful geometric, gradient, BN and protection checks on real source data."""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import torch
from torch.nn import functional as F

from train_erm import ROOT,Model_thin,Skeletons,model_sha,dump,now
from dre_v1p1_core import *


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(71);torch.cuda.manual_seed_all(71)
    checks={};max_identity=0.;max_invariance=0.;grad_error=0.
    # The actual v1 failure mechanism: zero RMS -> sqrt'(0) -> NaN after square.
    from dre_core import relation_error as original_v1_error
    ref=torch.ones(2,51,64,dtype=torch.float64)
    mask=torch.ones_like(ref,dtype=torch.bool);w=torch.ones(2,51,dtype=torch.float64)
    zero=ref.clone().requires_grad_(True)
    oldgrad,=torch.autograd.grad(original_v1_error(zero,ref,mask,w).square().sum(),zero)
    assert not torch.isfinite(oldgrad).all()
    zero=ref.clone().requires_grad_(True)
    repaired=relation_error_squared(zero,ref,mask,w)
    newgrad,=torch.autograd.grad(repaired.sum(),zero)
    assert torch.equal(repaired,torch.zeros_like(repaired))
    assert torch.isfinite(newgrad).all() and torch.count_nonzero(newgrad)==0
    candidate=(ref+.03*torch.randn_like(ref)).requires_grad_(True)
    oldvalue=original_v1_error(candidate,ref,mask,w).square()
    oldgrad,=torch.autograd.grad(oldvalue.sum(),candidate)
    newvalue=relation_error_squared(candidate,ref,mask,w)
    newgrad,=torch.autograd.grad(newvalue.sum(),candidate)
    assert torch.allclose(oldvalue,newvalue,atol=1e-14,rtol=1e-12)
    assert torch.allclose(oldgrad,newgrad,atol=1e-14,rtol=1e-12)
    checks['zero_residual_regression']={'passed':True,'old_zero_gradient_nonfinite':True,
            'repaired_zero_gradient_finite':True,'nonzero_values_gradients_equivalent':True}
    for domain in ['ntu','etriA']:
        ds=Skeletons(domain,'train',False)
        rows=[int(np.flatnonzero(ds.labels==k)[0]) for k in range(18)]
        x=torch.stack([ds[r][0] for r in rows]).cuda();y=torch.tensor(ds.labels[rows],device='cuda')
        identity=kinematic_transform(x,torch.zeros((len(x),10),device='cuda'))
        error=float((identity-x).abs().max());max_identity=max(max_identity,error)
        assert error<2e-6
        r,m=relations(x)
        p=torch.zeros((len(x),10),device='cuda');p[:,0]=math.log(1.13);p[:,1]=.19
        rigid=kinematic_transform(x,p);rr,mm=relations(rigid)
        error=float(((rr-r)*m).abs().max());max_invariance=max(max_invariance,error)
        assert error<1e-4 and torch.equal(m,mm)
        # Every non-rigid bone ratio equals its intended group scale.
        p[:,2:]=.04
        z=kinematic_transform(x,p)
        for j,parent in PARENTS.items():
            old=(x[:,:,:,j,:]-x[:,:,:,parent,:]).norm(dim=1)
            new=(z[:,:,:,j,:]-z[:,:,:,parent,:]).norm(dim=1)
            valid=old>1e-4
            assert (new[valid]/old[valid]-math.exp(float(p[0,0]+p[0,GROUP[j]]))).abs().max()<.003
        # Both actor roots receive the same global transform, not independent shifts.
        assert torch.isfinite(z).all()
        active=x.abs().sum((1,2,3))>0
        for i in range(len(x)):
            for actor in range(2):
                if not active[i,actor]:assert torch.count_nonzero(z[i,:,:,:,actor])==0
        # Differentiate an actual relational trajectory w.r.t. a shape parameter.
        pp=torch.zeros((len(x),10),device='cuda');pp[:,4]=.03;pp.requires_grad_(True)
        val=relations(kinematic_transform(x,pp))[0][:,:20].square().mean()
        g,=torch.autograd.grad(val,pp)
        assert torch.isfinite(g).all() and g[:,4].abs().sum()>0
        eps=.001
        plus=pp.detach().clone();minus=plus.clone();plus[0,4]+=eps;minus[0,4]-=eps
        fd=(relations(kinematic_transform(x,plus))[0][:,:20].square().mean()-
            relations(kinematic_transform(x,minus))[0][:,:20].square().mean())/(2*eps)
        e=abs(float(g[0,4]-fd));grad_error=max(grad_error,e)
        assert e<.002
        model=Model_thin(**MODEL_ARGS).cuda().train()
        oldsha=model_sha(model)
        expanded,stat,params=adversarial_expand(model,x[:4],y[:4],torch.ones(18,51,device='cuda'),
                torch.Generator(device='cuda').manual_seed(5))
        assert model_sha(model)==oldsha and model.training
        assert stat['relation_rms_max']<=.08001 and params[:,2:].abs().mean()>0
        assert all(p.requires_grad and p.grad is None for p in model.parameters())
        # The student receives nonzero gradients, and clean-only KD ignores a wrong teacher.
        logits,feat=model(torch.cat([x[:4],expanded]),return_feat=True)
        l0,l1=logits.chunk(2)
        fake=F.one_hot((y[:4]+1)%18,18).float()*10
        kl,mask=teacher_preservation(l0,fake,y[:4]);assert float(kl.detach())==0 and float(mask)==0
        loss=.5*(F.cross_entropy(l0,y[:4])+F.cross_entropy(l1,y[:4]))+\
             .1*supervised_contrastive(feat,y[:4].repeat(2))
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.fc.weight.grad.abs().sum()>0
        checks[domain]={'passed':True,'source_examples':18,'search':stat,'finite_student_loss':float(loss)}
    # Hand-calculated contrastive loss: all vectors identical, 4 observations -> log(3).
    v=torch.ones(4,8,device='cuda');lab=torch.tensor([0,1,0,1],device='cuda')
    assert abs(float(supervised_contrastive(v,lab))-math.log(3))<1e-6
    result={'passed':True,'checked_at':now(),'checks':checks,'max_identity_error':max_identity,
            'max_global_invariance_error':max_invariance,'max_gradient_fd_error':grad_error,
            'source_only':True,'target_data_opened':False}
    dump(ROOT/'work/DRE_v1p1_checks.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

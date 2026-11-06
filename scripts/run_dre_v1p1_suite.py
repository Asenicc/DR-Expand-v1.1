"""One non-overwriting DR-Expand v1.1 suite; N then EA, fixed seed0."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from train_erm import ROOT,dump,now
from train_dre_v1p1 import METHOD,preflight,read


def main():
    preflight()
    state_path=ROOT/'runs/suite_dre_v1p1_seed0.json'
    if state_path.exists():raise FileExistsError(state_path)
    for source in ['N','EA']:
        out=ROOT/'runs'/f'DRE_v1p1_{source}_seed0'
        if out.exists():
            # Only the separately verified epoch-5 warmup inheritance is allowed.
            if source!='N' or not (out/'warmup_inheritance.json').exists():raise FileExistsError(out)
            inheritance=read(out/'warmup_inheritance.json')
            assert inheritance['passed'] and inheritance['inherited_epoch']==5
            assert not (out/'results.json').exists()
        smoke=read(ROOT/'work'/f'DRE_v1p1_smoke_{source}_seed0'/'smoke_result.json')
        assert smoke['passed'] and smoke['source']==source and not smoke['target_data_opened']
    state={'method':METHOD,'pid':os.getpid(),'status':'starting','smoke':False,'seed':0,
           'sources':['N','EA'],'epochs_per_source':50,'batch_size':64,
           'completed_sources':[],'started_at':now(),'updated_at':now(),
           'protocol':str(ROOT/'outputs/DRE_v1p1_seed0_protocol.md'),
           'active_run':None,'active_source':None,'child_pid':None,'log':None}
    dump(state_path,state)
    try:
        for source in ['N','EA']:
            out=ROOT/'runs'/f'DRE_v1p1_{source}_seed0'
            log=ROOT/'work'/f'DRE_v1p1_{source}_seed0.log'
            command=[sys.executable,'-u','-B',str(ROOT/'scripts/train_dre_v1p1.py'),
                     '--source',source,'--output',str(out)]
            if source=='N':command.append('--resume')
            with log.open('x',encoding='utf-8') as stream:
                child=subprocess.Popen(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT)
                state.update(status='training',active_source=source,active_run=str(out),
                             child_pid=child.pid,log=str(log),command=command,updated_at=now())
                dump(state_path,state);print('CHILD_STARTED',source,child.pid,flush=True)
                code=child.wait()
            if code:raise RuntimeError(f'{source} exited {code}, see {log}')
            result=read(out/'results.json')
            assert result['epochs']==50 and not result['smoke']
            state['completed_sources'].append(source)
            state.update(active_run=None,active_source=None,child_pid=None,updated_at=now())
            dump(state_path,state)
        log=ROOT/'work/DRE_v1p1_seed0_finalize.log'
        command=[sys.executable,'-u','-B',str(ROOT/'scripts/finalize_dre_v1p1_results.py')]
        with log.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT)
            state.update(status='validating',child_pid=child.pid,log=str(log),command=command,updated_at=now())
            dump(state_path,state)
            code=child.wait()
        if code:raise RuntimeError(f'Export validation exited {code}, see {log}')
        verification=read(ROOT/'outputs/DRE_v1p1_seed0_verification.json')
        assert verification['passed'] and verification['final']
        state.update(status='completed',child_pid=None,completed_at=now(),updated_at=now(),
                     verification=str(ROOT/'outputs/DRE_v1p1_seed0_verification.json'))
        dump(state_path,state);print('SUITE_COMPLETED',json.dumps(state),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),updated_at=now())
        dump(state_path,state);raise


if __name__=='__main__':main()

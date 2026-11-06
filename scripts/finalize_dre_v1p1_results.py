"""Validate saved final predictions/checkpoints; never train or infer."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from train_erm import ROOT,NAMES,Model_thin,dump,file_sha,model_sha,now
from finalize_results import CLASSES,SETTINGS
from dre_v1p1_core import METHOD,MODEL_ARGS,CONFIG
from train_dre_v1p1 import preflight,read


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    protocol=preflight()
    suite=read(ROOT/'runs/suite_dre_v1p1_seed0.json')
    assert suite['status'] in ['validating','completed'] and not suite['smoke']
    assert suite['completed_sources']==['N','EA']
    check={};history={};metrics={};costs={}
    model=Model_thin(**MODEL_ARGS)
    for source in ['N','EA']:
        run=ROOT/'runs'/f'DRE_v1p1_{source}_seed0'
        cfg=read(run/'config.json');result=read(run/'results.json');progress=read(run/'progress.json')
        assert cfg['method']==result['method']==METHOD and not cfg['smoke'] and not result['smoke']
        assert cfg['source']==result['source']==source
        assert cfg['epochs']==result['epochs']==50 and cfg['seed']==result['seed']==0
        assert cfg['selection']==result['selection']=='fixed_final_epoch_50'
        assert cfg['method_config']==CONFIG and cfg['protocol_sha256']==file_sha(ROOT/'outputs/DRE_v1p1_seed0_protocol.json')
        assert progress['status']=='completed'
        if source=='N':
            inherited=read(run/'warmup_inheritance.json')
            assert inherited['passed'] and inherited['inherited_epoch']==5
            assert all(inherited['exact_state_checks'].values())
            assert inherited['original_checkpoint_sha256']==file_sha(ROOT/'runs/DRE_v1_N_seed0/last.pt')
        assert cfg['teacher_checkpoint_sha256']==file_sha(cfg['teacher_checkpoint'])
        for rel,sha in cfg['recovery']['cluster_files_sha256'].items():
            assert file_sha(ROOT/'Recover-and-Resample'/rel)==sha
        with (run/'training.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        assert [int(r['epoch']) for r in rows]==list(range(1,51))
        assert all(np.isfinite(float(r['loss'])) and float(r['relation_rms_max'])<=.08001 for r in rows)
        assert all(float(r['shape_log_abs_mean'])>0 for r in rows[5:])
        assert all(int(r['samples_seen'])==int(r['updates'])*64 for r in rows)
        rel=read(run/'relation_weights.json')
        assert cfg['relation_weights_sha256']==file_sha(run/'relation_weights.json')
        assert not rel['smoke'] and not rel['target_data_used'] and rel['source']==source
        assert rel['fitted_rows']==cfg['source_samples']
        expected=protocol['source_label_sha256'][source]
        assert rel['source_labels_sha256']==expected
        saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        assert saved['epoch']==50 and saved['protocol_sha256']==cfg['protocol_sha256']
        model.load_state_dict(saved['model'])
        assert model_sha(model)==result['checkpoint_model_sha256']
        history[source]=rows
        costs[source]={k:result[k] for k in ['training_seconds','search_seconds','classifier_updates','teacher_pretraining_epochs']}
        del saved
    for source,domain,setting,count in SETTINGS:
        run=ROOT/'runs'/f'DRE_v1p1_{source}_seed0';result=read(run/'results.json')
        old=ROOT/'runs'/f'ERM_{source}_seed0'
        with np.load(run/f'{domain}_predictions.npz',allow_pickle=False) as pred,\
             np.load(old/f'{domain}_predictions.npz',allow_pickle=False) as erm:
            labels=pred['labels'];p=pred['predictions'];z=pred['logits']
            assert len(labels)==count and z.shape==(count,18) and np.isfinite(z).all()
            assert np.array_equal(labels,erm['labels']) and np.array_equal(pred['samples'],erm['samples'])
            assert np.array_equal(p,z.argmax(1))
            cm=np.bincount(labels*18+p,minlength=324).reshape(18,18)
        totals=cm.sum(1);correct=cm.diagonal();recall=100*correct/totals
        declared=result['metrics'][domain]
        assert (totals>0).all() and np.array_equal(cm,np.asarray(declared['confusion_matrix']))
        assert abs(recall.mean()-declared['balanced_accuracy_percent'])<1e-9
        for i,name in enumerate(NAMES):assert abs(recall[i]-declared['per_class_recall_percent'][name])<1e-9
        metrics[setting]={'recall':recall,'counts':totals,'correct':correct,'BA':float(recall.mean())}
        check[setting]={'passed':True,'samples':count,'balanced_accuracy_percent':float(recall.mean()),
                        'predictions_sha256':file_sha(run/f'{domain}_predictions.npz'),
                        'last_checkpoint_sha256':file_sha(run/'last.pt')}
    settings=[v[2] for v in SETTINGS]
    wide=[];counts=[]
    for i,(name,zh) in enumerate(CLASSES):
        row={'class_id':i,'action_en':name,'action_zh':zh}
        for s in settings:
            row[s+'_recall_percent']=float(metrics[s]['recall'][i])
            counts.append({'setting':s,'class_id':i,'action_en':name,'action_zh':zh,
                           'correct':int(metrics[s]['correct'][i]),'test_samples':int(metrics[s]['counts'][i]),
                           'recall_percent':float(metrics[s]['recall'][i])})
        wide.append(row)
    prefix=ROOT/'outputs/DRE_v1p1_seed0'
    write_csv(Path(str(prefix)+'_per_class.csv'),wide)
    write_csv(Path(str(prefix)+'_per_class_counts.csv'),counts)
    comparisons=[]
    for key,label in [('ERM','ERM'),('RAR','Recover-and-Resample'),('SEED','SEED skeleton'),
                      ('LEAware','LEAware skeleton'),('AdvST','AdvST skeleton')]:
        vals=[read(ROOT/'runs'/f'{key}_{src}_seed0'/'results.json')['metrics'][dom]['balanced_accuracy_percent']
              for src,dom,_,_ in SETTINGS]
        comparisons.append((label,vals))
    own=[metrics[s]['BA'] for s in settings];comparisons.append((METHOD,own))
    lines=['# DR-Expand v1.1 exploratory results, seed0','',
           '|Method|'+'|'.join(settings)+'|Mean|','|---|'+'---:|'*5]
    for label,vals in comparisons:lines.append('|'+label+'|'+'|'.join(f'{v:.2f}' for v in vals+[float(np.mean(vals))])+'|')
    for base in ['ERM','Recover-and-Resample']:
        vals=dict(comparisons)[base];delta=[a-b for a,b in zip(own,vals)]
        lines.append('|Delta vs '+base+' (pp)|'+'|'.join(f'{v:+.2f}' for v in delta+[float(np.mean(delta))])+'|')
    lines+=['','N=NTU; E=ETRI; EA=ETRI adults; P=PKU-MMD-I. BA = mean of 18 class recalls.',
            '','## Full class recall (%)','',
            '|Action|'+'|'.join(settings)+'|','|---|'+'---:|'*4]
    for i,(_,zh) in enumerate(CLASSES):lines.append('|'+zh+'|'+'|'.join(f'{metrics[s]["recall"][i]:.2f}' for s in settings)+'|')
    lines+=['','## Fixed configuration and limitations','',
      '- Unmodified author AGCN, random initialization, 50 source epochs, seed0, batch64, SGD .1/.9/Nesterov/wd1e-4, milestones30/40; fixed last checkpoint.',
      '- Author Recover temporal augmentation; source-only class-statistical relation weights; two-step adversarial bone-group retargeting with relative relation RMS <=.08.',
      '- Loss .5*(original CE + expanded CE) + .1 supervised contrastive + .25 correct-source-teacher KL(T=2). Five warmup epochs before shape search.',
      '- Teacher is the previously trained source-only Recover model (50 additional pretraining epochs per source). Two-view BN and compute differ from the old Recover baseline.',
      '- No identified causal/content-style disentanglement; no proof of per-sample semantic preservation. A relation threshold is a proxy, not a semantic certificate.',
      '- Exploration after earlier target inspection. These are single-seed results, not confirmed novelty, significance or a blind target evaluation.',
      '- Negative results and all 18 classes retained. Additional teacher, two-view forward/backward and inner search costs must be reported.',
      '- v1.1 repairs sqrt(q).square() -> q in the inner objective, preserving the mathematical loss. N inherits exactly the unaffected five warmup epochs from the failed v1; epoch6 is replayed. Old failed artifacts are retained. Training seconds include inherited warmup and exclude the aborted partial epoch6.',
      '','## Compute','', '|Source|Classifier updates|Training seconds (incl. search)|Search seconds|Teacher epochs|',
      '|---|---:|---:|---:|---:|']
    for src,c in costs.items():lines.append(f'|{src}|{c["classifier_updates"]}|{c["training_seconds"]:.1f}|{c["search_seconds"]:.1f}|50|')
    report=Path(str(prefix)+'_report.md');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for source,rows in history.items():
        epoch=[int(r['epoch']) for r in rows]
        axes[0].plot(epoch,[float(r['loss']) for r in rows],label=source)
        axes[1].plot(epoch,[float(r['training_accuracy_percent']) for r in rows],label=source)
    axes[0].set(title='Source training loss',xlabel='Epoch');axes[1].set(title='Original-view training accuracy (%)',xlabel='Epoch')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(Path(str(prefix)+'_training.png'),dpi=160);plt.close(fig)
    verification={'method':METHOD,'passed':True,'final':True,'checked_at':now(),'checks':check,
                  'completed_settings':4,'protocol_sha256':file_sha(ROOT/'outputs/DRE_v1p1_seed0_protocol.json'),
                  'mean_balanced_accuracy_percent':float(np.mean(own)),'costs':costs,
                  'recomputed_from_saved_predictions':True,'training_or_inference_performed':False}
    dump(Path(str(prefix)+'_verification.json'),verification)
    # Default is an in-workspace delivery folder. External copying is explicit configuration only.
    delivery=Path(protocol['delivery_directory']);delivery.mkdir(parents=True,exist_ok=True)
    for suffix in ['_report.md','_per_class.csv','_per_class_counts.csv','_training.png','_verification.json','_protocol.md','_protocol.json']:
        path=Path(str(prefix)+suffix)
        if path.resolve()!= (delivery/path.name).resolve():shutil.copy2(path,delivery/path.name)
    print(json.dumps(verification),flush=True)


if __name__=='__main__':main()

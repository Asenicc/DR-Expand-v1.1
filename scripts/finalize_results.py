"""Verify saved ERM predictions and export the requested 18-class results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [
    ('eat','吃东西'),('drink','喝水'),('brush_teeth','刷牙'),('brush_hair','梳头'),
    ('wear_clothes','穿衣服'),('takeoff_clothes','脱衣服'),('glasses_on_off','戴／摘眼镜'),
    ('read','阅读'),('write','写字'),('phone_call','打电话'),('play_phone','玩手机'),
    ('clap','拍手'),('bow','鞠躬'),('handshake','握手'),('hug','拥抱'),
    ('handwave','挥手'),('point_finger','手指指向'),('fall_down','跌倒')]
SETTINGS = [('N','etri','N→E',13350),('N','pku','N→P',968),
            ('EA','ntu','EA→N',5212),('EA','pku','EA→P',968)]
MARKER = '\n<!-- PER_CLASS_RESULTS -->\n'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):value.update(data)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    seed=args.seed
    suite=json.loads((ROOT/'runs'/f'suite_seed{seed}.json').read_text(encoding='utf-8'))
    if not args.partial and suite['status']!='completed':
        raise RuntimeError('The full suite is not completed; no final result has been produced.')
    out=ROOT/'outputs'
    out.mkdir(exist_ok=True)
    metrics={}
    verification={'seed':seed,'verified_at':now(),'suite_status':suite['status'],
                  'single_seed':True,'selection':'fixed_final_epoch_50','checks':{}}
    for source,domain,setting,count in SETTINGS:
        directory=ROOT/'runs'/f'ERM_{source}_seed{seed}'
        result_path=directory/'results.json'
        if not result_path.exists():
            if args.partial:continue
            raise FileNotFoundError(result_path)
        result=json.loads(result_path.read_text(encoding='utf-8'))
        config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        progress=json.loads((directory/'progress.json').read_text(encoding='utf-8'))
        assert result['epochs']==config['epochs']==50
        assert result['seed']==config['seed']==seed
        assert progress['status']=='completed'
        assert config['selection']=='fixed final epoch; no target-based checkpoint selection'
        assert config['reference_commit']=='fb4d621c14f088c3bbac4031faef9cc0a6caf1cd'
        with (directory/'training.csv').open(encoding='utf-8') as f:history=list(csv.DictReader(f))
        assert [int(row['epoch']) for row in history]==list(range(1,51))
        assert all(np.isfinite(float(row['loss'])) for row in history)
        assert config['runner_sha256']==digest(ROOT/'scripts/train_erm.py')
        assert (directory/'last.pt').stat().st_size>0
        predicted=directory/f'{domain}_predictions.npz'
        with np.load(predicted,allow_pickle=False) as saved:
            truth=saved['labels'];prediction=saved['predictions'];logits=saved['logits']
            assert len(truth)==len(prediction)==len(saved['samples'])==count
            assert logits.shape==(count,18) and np.isfinite(logits).all()
            assert np.array_equal(prediction,logits.argmax(1))
            confusion=np.bincount(truth*18+prediction,minlength=18*18).reshape(18,18)
        n=confusion.sum(1);correct=confusion.diagonal();accuracy=correct/n*100
        assert (n>0).all()
        declared=result['metrics'][domain]
        assert np.array_equal(confusion,np.asarray(declared['confusion_matrix']))
        assert np.array_equal(n,np.asarray(declared['class_counts']))
        assert abs(accuracy.mean()-declared['balanced_accuracy_percent'])<1e-10
        assert abs(correct.sum()/count*100-declared['accuracy_percent'])<1e-10
        for i,(name,_) in enumerate(CLASSES):
            assert abs(accuracy[i]-declared['per_class_recall_percent'][name])<1e-10
        metrics[setting]={'accuracy':accuracy,'counts':n,'correct':correct,
                          'balanced_accuracy_percent':float(accuracy.mean()),
                          'overall_accuracy_percent':float(correct.sum()/count*100)}
        verification['checks'][setting]={'passed':True,'test_samples':count,
             'epochs':50,'predictions_sha256':digest(predicted),
             'checkpoint_sha256':digest(directory/'last.pt'),
             'balanced_accuracy_percent':float(accuracy.mean()),
             'overall_accuracy_percent':float(correct.sum()/count*100)}
    complete=len(metrics)==4
    assert args.partial or complete
    settings=[item[2] for item in SETTINGS]
    wide=[];long=[]
    for i,(name,zh) in enumerate(CLASSES):
        row={'class_id':i,'action_zh':zh,'action_en':name}
        for setting in settings:
            row[setting+'_accuracy_percent']=round(float(metrics[setting]['accuracy'][i]),6) if setting in metrics else ''
            if setting in metrics:
                long.append({'setting':setting,'class_id':i,'action_zh':zh,'action_en':name,
                     'correct':int(metrics[setting]['correct'][i]),
                     'test_samples':int(metrics[setting]['counts'][i]),
                     'class_accuracy_percent':float(metrics[setting]['accuracy'][i])})
        wide.append(row)
    for name,rows in [('per_class',wide),('per_class_counts',long)]:
        if not rows:continue
        with (out/f'ERM_seed{seed}_{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['## 18类动作的准确率（%）','',
           '每类准确率 = 该类预测正确数 ÷ 该类测试样本数，即该类召回率。','',
           '| 动作 | N→E | N→P | EA→N | EA→P |',
           '|---|---:|---:|---:|---:|']
    for i,(_,zh) in enumerate(CLASSES):
        values=[f"{metrics[s]['accuracy'][i]:.2f}" if s in metrics else '待完成' for s in settings]
        lines.append('| '+zh+' | '+' | '.join(values)+' |')
    lines += ['', '各类正确数和测试样本数保存在 `ERM_seed'+str(seed)+'_per_class_counts.csv`。',
              '平衡准确率是上表18类准确率的算术平均；整体准确率按全部测试样本计算，两者分别报告。',
              '', '## 已核对的汇总指标', '',
              '| 方向 | 平衡准确率（%） | 整体准确率（%） |',
              '|---|---:|---:|']
    for setting in settings:
        if setting in metrics:
            m=metrics[setting]
            lines.append(f"| {setting} | {m['balanced_accuracy_percent']:.2f} | {m['overall_accuracy_percent']:.2f} |")
    if complete:
        average=float(np.mean([metrics[s]['balanced_accuracy_percent'] for s in settings]))
        lines.append(f'| 四方向平衡准确率平均 | {average:.2f} | — |')
        verification['average_balanced_accuracy_percent']=average
    report_path=out/f'ERM_seed{seed}_report.md'
    base=report_path.read_text(encoding='utf-8').split(MARKER)[0]
    if not complete:
        base=re.sub(r'状态：\*\*.*?\*\*。更新时间：[^\n]+',
                    f'状态：**NTU源域已完成，ETRI成人源域训练中**。更新时间：{now()}。',base,count=1)
    report_path.write_text(base+MARKER+'\n'.join(lines)+'\n',encoding='utf-8')
    verification['passed']=True
    verification['final']=complete
    (out/f'ERM_seed{seed}_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'final':complete,'completed_settings':list(metrics),
          'report':str(report_path),'per_class_csv':str(out/f'ERM_seed{seed}_per_class.csv'),
          'verification':str(out/f'ERM_seed{seed}_verification.json')},ensure_ascii=False))


if __name__=='__main__':main()

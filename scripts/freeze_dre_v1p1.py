"""Freeze an executable configuration after meaningful checks and source smokes."""
from __future__ import annotations

import json
from pathlib import Path

from train_erm import ROOT,DATA,file_sha,dump,now
from train_dre_v1p1 import FILES,read
from dre_v1p1_core import METHOD,CONFIG


def main():
    manifest=ROOT/'outputs/DRE_v1p1_seed0_protocol.json'
    md=ROOT/'outputs/DRE_v1p1_seed0_protocol.md'
    if manifest.exists() or md.exists():raise FileExistsError('Frozen DRE protocol already exists')
    assert read(ROOT/'work/DRE_v1p1_checks.json')['passed']
    for source in ['N','EA']:
        smoke=read(ROOT/'work'/f'DRE_v1p1_smoke_{source}_seed0'/'smoke_result.json')
        assert smoke['passed'] and smoke['smoke'] and not smoke['target_data_opened']
    text='''# DR-Expand v1.1：本轮实验配置（seed0）

本文件用于固定实际执行配置，不是已经成立的论文贡献或方法论证。

- 任务：Common-18跨数据集骨架单源DG。N→E、N→P、EA→N、EA→P。N源12651条，EA源13568条；所有额外统计仅拟合本源训练集。
- 推理主干：未经修改的作者AGCN Model_thin，随机初始化，与既有Recover同seed主干初始参数完全一致；没有新增推理分支/投影。原作者仓库与所有旧实验不改动。
- 训练：seed0，各源50轮，batch64/drop_last，SGD lr .1、momentum .9、Nesterov、wd1e-4，0-based30/40衰减；FP32，workers0。固定末轮，目标只最终推理，不按目标成绩改配置。
- 基础时间扩展：原作者Recover-and-Resample，ABL_linearextrap_beta、cropresize0.7、增强比例.75，沿用对应完整源训练聚类。既有随机旋转theta_val30实际为±60度角参数。
- 判别关系：10个人体锚点的45条同人距离关系，以及6条双人距离关系，按平均躯干尺度归一化；比较64帧距离轨迹和其一阶差分。双人关系仅在两人都有效时参与。
- 关系权重：仅用当前源训练集，依据每条关系的均值、标准差、绝对变化量计算类对其余类的Fisher式统计权重；分数截断5、加底值.25后类内归一化。它是类统计启发式，没有证明因果性、跨域不变性或内容/风格可识别解耦。
- 运动学扩展：以spine-mid为根，保持骨方向，对躯干、头颈、上臂、前臂、手、股、小腿、足八组骨长施加每序列恒定的.9–1.1倍率；同一参数共享给双侧及两个人。共同整体尺度.85–1.15、规范Z轴yaw±15度。保留原帧序、两人和空人物；不插入新时刻，不独立删除人物或手部。
- 对抗搜索：前5轮只做原Recover和双视图监督。第6轮起，每批在归一化参数空间随机初始化±.1，执行2步符号梯度上升，步长.5、投影[-1,1]；最大化CE−2×关系相对RMS平方。模型eval，BN不更新、模型参数不求梯度。
- 关系限制：以当前Recover时间视图作参照，避免错配时刻。类加权距离/差分相对RMS不超过.08；不满足则仅二分收缩非刚性骨长参数，最多8次，仍失败则形变参数归零。整体旋转/尺度保留。这个数值代理不等于逐样本语义认证。
- 最小化：.5×[CE(原旋转源视图,y)+CE(扩展Recover视图,y)] + .1×两视图监督对比损失(温度.1，256维归一化主干特征，无新头) + .25×源域教师KL(T=2)。两视图合批，因此训练BN统计及计算不同于旧Recover。
- 教师：分别使用已完成的RAR_N_seed0和RAR_EA_seed0固定第50轮模型。仅教师在原旋转源视图上预测正确的样本参加KL；所有18类遵循同一规则。不使用目标低分列表作训练权重。该教师额外需要每源50轮预训练，尽管本轮复用现成权重，也必须披露方法成本。
- 范围：这是判别关系约束的对抗运动学域扩展候选版，没有实现SEED式学习型内容/风格双编码器，也不声称已完成此前宽泛设想中的所有机制。
- 输出：每轮损失分项、原视图训练准确率、关系偏离/回退率/形变幅度、显存与搜索时间；保存随机数/优化器/数据顺序和末轮模型。跨域只在各源50轮结束后推理，保存全量预测。结果导出校验只读取保存文件，不重训、不重推理。
- 比较：对比ERM、Recover、SEED、LEAware、AdvST的四方向BA、平均值、完整18类召回率和正确数/分母，保留负结果。所有已知目标结果参与过前期探索；单种子仅作候选筛选，准确率高也不自动证明新颖性或论文结论。
- 本轮不自动追加参数搜索、组合方法、消融或重训。异常先报告。每小时汇报，完成后交付。
'''
    text+='''\n## 数值修复与恢复来源\n\n原v1在NTU第6轮的对抗搜索发生非有限梯度并退出，原目录、源码和冻结配置全部保留。v1.1只把搜索目标中的sqrt(q).square()改为直接使用平方误差q；非零残差下目标值/梯度等价，零残差处导数定义为0，不改变模型、损失系数、几何上限或训练预算。已添加零残差回归检查。\n\nNTU继承原v1完整第1–5轮的模型、优化器、全部随机数与训练CSV，从第6轮重新执行；第6轮中断的未提交更新丢弃。继承阶段未开启对抗搜索，因此不受本次缺陷影响。ETRI从随机初始化正常训练50轮。所有新输出均使用DRE_v1p1前缀，原失败实验不覆盖。warmup_inheritance.json记录两份检查点模型/优化器/RNG一致性。\n'''
    md.write_text(text,encoding='utf-8')
    teachers={}
    for source in ['N','EA']:
        run=ROOT/'runs'/f'RAR_{source}_seed0'
        teachers[source]={'files_sha256':{str(run/f):file_sha(run/f) for f in ['last.pt','config.json','results.json','environment.json']}}
    files={p:file_sha(ROOT/p) for p in FILES+['scripts/freeze_dre_v1p1.py','scripts/inherit_dre_v1_warmup.py','outputs/DRE_v1p1_seed0_protocol.md']}
    for source,domain in [('N','ntu'),('EA','etriA')]:
        for name in ['train_data_joint.npy','train_label.pkl']:
            rel=str((DATA/domain/name).relative_to(ROOT)).replace('\\','/')
            files[rel]=file_sha(ROOT/rel)
    payload={'method':METHOD,'frozen_at':now(),'seed':0,'epochs':50,'smoke':False,
             'method_config':CONFIG,'files_sha256':files,'teachers':teachers,
             'repair':{'original_protocol_sha256':file_sha(ROOT/'outputs/DRE_v1_seed0_protocol.json'),
                       'original_run':str(ROOT/'runs/DRE_v1_N_seed0'),
                       'inherited_epochs':[1,2,3,4,5],
                       'change':'sqrt(q).square() -> q in inner objective; same objective and hyperparameters'},
             'source_label_sha256':{s:file_sha(DATA/d/'train_label.pkl') for s,d in [('N','ntu'),('EA','etriA')]},
             'delivery_directory':str(ROOT/'outputs/DRE_v1p1_seed0_delivery'),
             'checks_sha256':file_sha(ROOT/'work/DRE_v1p1_checks.json'),
             'smoke_sha256':{s:file_sha(ROOT/'work'/f'DRE_v1p1_smoke_{s}_seed0'/'smoke_result.json') for s in ['N','EA']}}
    dump(manifest,payload)
    print(json.dumps({'frozen':True,'manifest':str(manifest),'code_files':len(files),'teacher_sources':list(teachers)}),flush=True)


if __name__=='__main__':main()

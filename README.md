# DR-Expand v1.1

Common-18 单源跨数据集骨架动作识别的探索实现：Recover-and-Resample + 判别关系约束的对抗运动学扩展 + 双视图监督 + SupCon + 源域教师保护。

这不是 Recover 作者发布的新方法，也不是已通过多种子验证的稳定结论。本仓库保留取得下表结果的原始源码，并增加一个可移植运行入口；不包含后续 SupCon 重加权、跌倒专用分支或其他方法。

## 已完成的四方向结果

以下均为 **seed0、固定第50轮** 的历史结果，BA 是18类召回率的算术平均，不是总体正确率。

| 方法 | N→E | N→P | EA→N | EA→P | 四方向均值 |
|---|---:|---:|---:|---:|---:|
| Recover-and-Resample，本地同口径 | 58.46 | 74.18 | 47.54 | 54.15 | 58.58 |
| DR-Expand v1.1 | 59.33 | 78.95 | 49.09 | 55.44 | 60.70 |

N=NTU，E=ETRI，EA=ETRI成人，P=PKU-MMD-I。精确值见 [results/four_directions.csv](results/four_directions.csv)，完整18类召回和正确数/分母见 [results/per_class.csv](results/per_class.csv)、[results/per_class_counts.csv](results/per_class_counts.csv)。原正式校验与冻结协议见 [reference/verification.json](reference/verification.json)、[reference/protocol.md](reference/protocol.md)。

**选轮口径必须区分：**上述60.70%不是逐目标域逐轮最佳；它是这次既有实验的固定末轮结果。2026-11-06起的新实验约定逐目标域独立取最佳BA，本仓库新入口默认 `--selection target-best`，同时记录所有轮次和最佳轮次。使用新口径产生的结果必须与相同口径基线比较，不能把新结果直接减去此处历史末轮结果。原历史脚本保持原样，不静默改成新协议。

## 代码位置

- [scripts/dre_v1p1_core.py](scripts/dre_v1p1_core.py)：实际使用的变换、51条关系、权重拟合、平方误差、两步搜索、收缩/撤回、SupCon和蒸馏。
- [scripts/train_dre_v1p1.py](scripts/train_dre_v1p1.py)：产生历史结果的原始训练脚本，包含历史冻结/教师/检查记录校验。
- [scripts/train_rar.py](scripts/train_rar.py)、[scripts/train_erm.py](scripts/train_erm.py)：原数据封装、Recover调用、AGCN导入及评估实现。
- [reproduce.py](reproduce.py)：新增可移植入口，直接调用未改的核心函数；历史审计脚本中的本机路径和旧smoke记录不能直接用于新电脑，因此此入口独立处理配置、训练与选轮。它没有在本次发布任务中重新进行GPU训练，不声称已重现一遍60.70%。
- [weights/relation_N.json](weights/relation_N.json)、[weights/relation_EA.json](weights/relation_EA.json)：这次正式源域拟合得到的18×51权重，包含类别/关系顺序和有效样本计数。
- [docs/reproduction_details.md](docs/reproduction_details.md)：实际损失、随机种子、教师选择、权重归一化及原N源恢复说明。

## 在另一台电脑运行

先取得 Recover 作者代码并固定版本；第三方源码通过上游取得，本仓库不重新指定其许可证：

```bash
git clone https://github.com/HanchaoLiu/Recover-and-Resample.git
git -C Recover-and-Resample checkout fb4d621c14f088c3bbac4031faef9cc0a6caf1cd
```

安装兼容GPU的CUDA版PyTorch及本仓库的Python依赖：

```bash
python -m pip install -r requirements.txt
python tools/check_release.py
```

历史环境：Python3.10.21、PyTorch2.11.0+cu128、NumPy2.2.6、RTX5060Ti。不同硬件/算子版本不能保证逐位一致。

数据由用户自行取得及按上游流程预处理，**不上传原始骨架、逐样本预测或模型权重**。`--data-root`应指向已准备好的 `common18/xsub64`，含 `ntu`、`etriA`、`etri`、`pku` 子目录；数据形状为 `[N,3,64,25,2]`、float32，标签为固定0..17。作者源训练聚类位于上游 `data/cluster_result`；此版本沿用它们，没有重新拟合Recover聚类。

新入口会核验源训练数据与历史协议SHA一致，遇到差异就报错，避免在不同数据上仍声称复现同一数字。数据准备说明见上游 [gendata/readme_gendata.md](https://github.com/HanchaoLiu/Recover-and-Resample/blob/fb4d621c14f088c3bbac4031faef9cc0a6caf1cd/gendata/readme_gendata.md)。

以 `D:/data/common18/xsub64` 为例，所有输出目录必须不存在：

```bash
# 先各训练一个源域Recover教师；教师固定选第50轮，不按目标分数选。
python reproduce.py --stage teacher --source N --data-root D:/data/common18/xsub64 --output runs/reproduce_RAR_N_seed0
python reproduce.py --stage teacher --source EA --data-root D:/data/common18/xsub64 --output runs/reproduce_RAR_EA_seed0

# 新实验默认逐目标域逐轮最佳，逐轮推理不会更新模型/BN或消耗后续训练RNG。
python reproduce.py --stage student --source N --data-root D:/data/common18/xsub64 --teacher runs/reproduce_RAR_N_seed0/last.pt --output runs/reproduce_DRE_N_seed0
python reproduce.py --stage student --source EA --data-root D:/data/common18/xsub64 --teacher runs/reproduce_RAR_EA_seed0/last.pt --output runs/reproduce_DRE_EA_seed0
```

若专门核对历史末轮60.70%的设置，在两条学生命令中显式加上 `--selection fixed-final-50`。这是历史结果复核，不把其标为逐轮最佳。

此入口不注册后台任务、不自动续训、不覆盖旧输出。仅运行 `--help` 或 `tools/check_release.py` 不会训练。默认逐轮最佳学生运行保存50个检查点及每轮预测，请预留磁盘空间；目标标签只用于评估和选轮，不参与训练梯度或超参数搜索。该选轮方式应标为 **target-oracle selection**。

## 限制

- 仅seed0；已知目标曾参与研究设计，不能称盲测、多种子稳定结论或已定位因果因素。
- 距离关系是统计代理，不是逐样本动作语义认证；两个人物/空人物按原实现处理。
- 新入口的数学核心复用原函数，但未在发布过程中重跑全实验；静态、导入、CPU核心检查不能替代GPU轨迹复现。
- 旧N源继承v1完整前5轮，再用v1.1从第6轮继续；新增入口从头执行同样5轮warmup。硬件及库差异需单列。
- 后续若改超参数、误差归一化、教师选轮或数据预处理，应单独记录新实验，不修改历史结果标签。

参考：Liu et al., *Recovering Complete Actions for Cross-dataset Skeleton Action Recognition*, NeurIPS2024，[论文](https://arxiv.org/abs/2410.23641)、[作者代码](https://github.com/HanchaoLiu/Recover-and-Resample)。第三方来源及许可说明见 [NOTICE.md](NOTICE.md)。

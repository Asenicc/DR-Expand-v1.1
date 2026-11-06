# DR-Expand v1.1 exploratory results, seed0

|Method|N→E|N→P|EA→N|EA→P|Mean|
|---|---:|---:|---:|---:|---:|
|ERM|53.19|70.68|42.52|51.14|54.38|
|Recover-and-Resample|58.46|74.18|47.54|54.15|58.58|
|SEED skeleton|49.96|65.28|40.18|46.52|50.48|
|LEAware skeleton|55.69|72.91|49.80|53.00|57.85|
|AdvST skeleton|55.27|72.00|45.96|52.22|56.36|
|DR-Expand-v1.1|59.33|78.95|49.09|55.44|60.70|
|Delta vs ERM (pp)|+6.15|+8.27|+6.56|+4.30|+6.32|
|Delta vs Recover-and-Resample (pp)|+0.88|+4.77|+1.54|+1.29|+2.12|

N=NTU; E=ETRI; EA=ETRI adults; P=PKU-MMD-I. BA = mean of 18 class recalls.

## Full class recall (%)

|Action|N→E|N→P|EA→N|EA→P|
|---|---:|---:|---:|---:|
|吃东西|42.13|28.07|18.55|0.00|
|喝水|19.19|61.67|57.66|70.00|
|刷牙|54.35|72.22|25.27|90.74|
|梳头|77.88|83.33|37.36|9.26|
|穿衣服|90.90|92.16|94.18|98.04|
|脱衣服|99.12|94.12|73.91|90.20|
|戴／摘眼镜|65.45|98.04|69.10|94.12|
|阅读|43.74|64.81|25.27|1.85|
|写字|57.39|36.84|3.31|14.04|
|打电话|17.75|27.78|29.09|16.67|
|玩手机|30.66|76.67|86.55|100.00|
|拍手|54.58|91.23|22.34|21.05|
|鞠躬|52.27|96.08|12.32|13.73|
|握手|99.81|100.00|95.29|100.00|
|拥抱|97.09|100.00|91.97|91.67|
|挥手|70.08|98.11|46.35|75.47|
|手指指向|95.27|100.00|77.54|100.00|
|跌倒|0.38|100.00|17.45|11.11|

## Fixed configuration and limitations

- Unmodified author AGCN, random initialization, 50 source epochs, seed0, batch64, SGD .1/.9/Nesterov/wd1e-4, milestones30/40; fixed last checkpoint.
- Author Recover temporal augmentation; source-only class-statistical relation weights; two-step adversarial bone-group retargeting with relative relation RMS <=.08.
- Loss .5*(original CE + expanded CE) + .1 supervised contrastive + .25 correct-source-teacher KL(T=2). Five warmup epochs before shape search.
- Teacher is the previously trained source-only Recover model (50 additional pretraining epochs per source). Two-view BN and compute differ from the old Recover baseline.
- No identified causal/content-style disentanglement; no proof of per-sample semantic preservation. A relation threshold is a proxy, not a semantic certificate.
- Exploration after earlier target inspection. These are single-seed results, not confirmed novelty, significance or a blind target evaluation.
- Negative results and all 18 classes retained. Additional teacher, two-view forward/backward and inner search costs must be reported.
- v1.1 repairs sqrt(q).square() -> q in the inner objective, preserving the mathematical loss. N inherits exactly the unaffected five warmup epochs from the failed v1; epoch6 is replayed. Old failed artifacts are retained. Training seconds include inherited warmup and exclude the aborted partial epoch6.

## Compute

|Source|Classifier updates|Training seconds (incl. search)|Search seconds|Teacher epochs|
|---|---:|---:|---:|---:|
|N|9850|8901.3|2954.1|50|
|EA|10600|9515.9|3169.4|50|

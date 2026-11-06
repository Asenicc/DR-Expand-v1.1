# Sources and permissions

DR-Expand v1.1 is a local exploratory extension, not an official release by the Recover-and-Resample authors. Original experiment records and this new portable runner are identified separately.

The AGCN backbone, original rotation, Recover-and-Resample implementation, published source-training clusters, and data preparation are obtained from:

- https://github.com/HanchaoLiu/Recover-and-Resample
- Fixed reference commit: fb4d621c14f088c3bbac4031faef9cc0a6caf1cd
- Liu et al., Recovering Complete Actions for Cross-dataset Skeleton Action Recognition, NeurIPS 2024: https://arxiv.org/abs/2410.23641

This repository does not vendor or relicense the upstream repository or distribute NTU/ETRI/PKU data. Consult the upstream project and each dataset's terms. No new license grant is asserted for third-party material. A public repository alone does not determine reuse permissions.

The historical local adapter scripts are included to make the experiment inspectable. Do not attribute DR-Expand's SupCon, teacher KL, geometric search, or 51-relation constraint to the original Recover method.

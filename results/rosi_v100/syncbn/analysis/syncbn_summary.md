# Phase 2C — BatchNorm vs SyncBatchNorm

ResNet18, 4 V100 GPUs, global batch 256, local batch 64, FP16 AMP, three runs per condition.

| Normalization | Throughput | CV | Val accuracy | Epoch time | VRAM/rank |
|---|---:|---:|---:|---:|---:|
| BatchNorm | 13163.49 ± 271.37 img/s | 2.06% | 40.47% ± 2.86 pp | 3.526 ± 0.072 s | 0.327 GiB |
| SyncBatchNorm | 7061.68 ± 57.10 img/s | 0.81% | 40.29% ± 3.50 pp | 6.478 ± 0.051 s | 0.327 GiB |

## SyncBatchNorm change relative to BatchNorm

- Throughput: -46.35%
- Mean epoch time: +83.73%
- Validation accuracy: -0.19 percentage points
- Peak VRAM/rank: +0.00%

The validation-accuracy comparison is based on only three five-epoch runs per condition and should not be interpreted as a definitive generalization result.

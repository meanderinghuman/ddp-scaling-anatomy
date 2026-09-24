# Phase 2A — Repeated V100 Strong-Scaling Results

Each workload/GPU-count configuration was run three times. The same model seed and training configuration were retained to measure system/runtime variability rather than seed variability.

The 95% confidence interval uses Student's t distribution with n=3 and df=2. Because only three repetitions are available, the confidence intervals should be interpreted cautiously; the raw runs, standard deviation, and coefficient of variation are also reported.

## ResNet18 / CIFAR-100

| GPUs | Local batch | Mean throughput | Std | CV | 95% CI | Speedup | Efficiency | Mean VRAM/rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 8056.89 | 105.23 | 1.31% | ±261.42 | 1.000× | 100.00% | 0.757 GiB |
| 2 | 128 | 8783.87 | 461.61 | 5.26% | ±1146.70 | 1.090× | 54.51% | 0.474 GiB |
| 4 | 64 | 13163.49 | 271.37 | 2.06% | ±674.13 | 1.634× | 40.85% | 0.327 GiB |

## U-Net / Heat2D

| GPUs | Local batch | Mean throughput | Std | CV | 95% CI | Speedup | Efficiency | Mean VRAM/rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 1046.12 | 27.76 | 2.65% | ±68.96 | 1.000× | 100.00% | 0.482 GiB |
| 2 | 8 | 991.05 | 35.32 | 3.56% | ±87.73 | 0.947× | 47.37% | 0.307 GiB |
| 4 | 4 | 992.26 | 18.12 | 1.83% | ±45.02 | 0.949× | 23.71% | 0.221 GiB |


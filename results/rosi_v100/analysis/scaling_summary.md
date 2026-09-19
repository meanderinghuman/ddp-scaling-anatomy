# ROSI V100 DDP Strong-Scaling Results

Primary performance metrics use the trainer's saved steady-state statistics, which exclude epoch 1.

Speedup: $S_N = Q_N / Q_1$.  Efficiency: $E_N = S_N / N \times 100\%$.

## ResNet18 / CIFAR-100

| GPUs | Global batch | Local batch | Throughput | Speedup | Efficiency | Epoch time | Steady VRAM/rank | Final metric |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 256 | 8111.66 | 1.000x | 100.00% | 10.018s | 0.758 GiB | 40.00% acc |
| 2 | 256 | 128 | 10257.30 | 1.265x | 63.23% | 4.796s | 0.473 GiB | 40.64% acc |
| 4 | 256 | 64 | 13232.80 | 1.631x | 40.78% | 3.503s | 0.327 GiB | 35.52% acc |

## U-Net / Heat2D

| GPUs | Global batch | Local batch | Throughput | Speedup | Efficiency | Epoch time | Steady VRAM/rank | Final metric |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 1017.77 | 1.000x | 100.00% | 4.160s | 0.481 GiB | 0.003700 MSE |
| 2 | 16 | 8 | 1041.74 | 1.024x | 51.18% | 4.000s | 0.308 GiB | 0.003525 MSE |
| 4 | 16 | 4 | 988.47 | 0.971x | 24.28% | 4.208s | 0.221 GiB | 0.003467 MSE |


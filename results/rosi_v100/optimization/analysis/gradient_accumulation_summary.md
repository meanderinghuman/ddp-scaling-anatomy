# Phase 3 — Gradient Accumulation and DDP.no_sync()

U-Net / Heat2D, 4 V100 GPUs, microbatch 4/rank, 4 accumulation steps, effective global batch 64, three runs per synchronization strategy.

| Mode | Throughput | CV | Train wall | Final val MSE | VRAM/rank |
|---|---:|---:|---:|---:|---:|
| Naive accumulation | 1071.95 ± 15.50 fields/s | 1.45% | 3.831 ± 0.053 s | 0.004677 | 0.251 GiB |
| DDP.no_sync() | 1175.56 ± 10.87 fields/s | 0.92% | 3.492 ± 0.032 s | 0.004689 | 0.253 GiB |

## no_sync change relative to naive accumulation

- Throughput: +9.67%
- Speedup: 1.097×
- Mean training wall time: -8.83%
- Final validation MSE change: +0.00001203
- Peak VRAM/rank: +0.47%

Both modes use the same effective global batch and optimizer update frequency. The intended experimental difference is gradient-synchronization frequency.

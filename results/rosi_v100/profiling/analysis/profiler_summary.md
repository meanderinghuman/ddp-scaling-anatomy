# Phase 2B — DDP / NCCL Profiling

NCCL times below represent summed device-kernel residency from the profiler. They must **not** be interpreted as pure unhidden communication overhead because NCCL communication may overlap backward computation.

The benchmark step time is derived independently from the three-run mean throughput as `global_batch_size / throughput`.

| Workload | GPUs | Local batch | All-reduces/step | NCCL ms/step mean | NCCL ms/step max | Rank imbalance | Benchmark step time |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 1 | 256 | 0.00 | 0.000 | 0.000 | n/a | 31.774 ms |
| ResNet18 | 4 | 64 | 3.00 | 11.811 | 19.147 | 8.89× | 19.448 ms |
| U-Net | 1 | 16 | 0.00 | 0.000 | 0.000 | n/a | 15.295 ms |
| U-Net | 4 | 4 | 3.00 | 7.502 | 14.828 | 4.23× | 16.125 ms |

## Interpretation note

`max NCCL residency / benchmark step time` is retained in the CSV as a communication-pressure diagnostic. It is not an overhead percentage and may be large because CUDA streams can overlap in time.

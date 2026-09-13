"""
Distributed PyTorch primitives smoke test.

This script does NOT train a neural network.

Its purpose is to teach and verify the fundamental concepts used by
DistributedDataParallel:

    process
    rank
    local rank
    world size
    process group
    barrier
    all_reduce

We deliberately use the Gloo backend and CPU tensors so that two
distributed processes can be tested on a laptop with only one GPU.

Run with:

    torchrun --standalone --nproc-per-node=2 \
        scripts/07_test_distributed_primitives.py
"""

import os 
import torch
import torch.distributed as dist 


# read variable created by torchrun 

rank = int(
    os.environ["RANK"]
)

local_rank = int(
    os.environ["LOCAL_RANK"]
)

world_size = int(
    os.environ["WORLD_SIZE"]
)

# initialize distributed process group 

# gloo is for cpu distributed smoke test 

# later nccl as backend for gpus

dist.init_process_group(backend="gloo")
print(
    f"[rank {rank}] "
    f"local_rank={local_rank}, "
    f"world_size={world_size}",
    flush=True,
)

# see pytorch agrees with torchrun 

assert dist.get_rank() == rank 

assert (dist.get_world_size() == world_size)

# For two processes:
#
# rank 0:
#     tensor = 1
#
# rank 1:
#     tensor = 2

value = torch.tensor([float(rank + 1)], dtype=torch.float32,)
print(
    f"[rank {rank}] "
    f"before all_reduce: "
    f"{value.item():.1f}",
    flush=True,
)

## collective communication: all_reduce 
# Every rank contributes its local value.
#
# ReduceOp.SUM means:
#
#     result
#       =
#     value_rank0
#       +
#     value_rank1
#       + ...
#
#
# With two processes:
#
#     1 + 2 = 3
#
# IMPORTANT:
#
# all_reduce does not return the result as a new tensor.
# It modifies `value` in-place.
#
# Also, ALL ranks receive the final reduced value.

dist.all_reduce(value, op=dist.ReduceOp.SUM,)

print(
    f"[rank {rank}] "
    f"after all_reduce:  "
    f"{value.item():.1f}",
    flush=True,
)

# verify expected results 
# General formula:
#
#     1 + 2 + ... + world_size
#
#   = world_size * (world_size + 1) / 2

expected_sum = (world_size * (world_size + 1) / 2)

assert torch.isclose(value, torch.tensor(
    [expected_sum],
    dtype=torch.float32,
),
).all()

# barrier() means:
#
#     every process waits here
#
# until:
#
#     every process has reached this point.
#
#
# Rank 0 cannot continue past the barrier while rank 1 is still doing
# earlier work.

print(
    f"[rank {rank}] "
    f"entering barrier",
    flush=True,
)


dist.barrier()


print(
    f"[rank {rank}] "
    f"passed barrier",
    flush=True,
)

# In distributed training we often want only one process to:
#
#     save checkpoints
#     write final JSON
#     print summary tables
#
# Otherwise all ranks could try to write the same file simultaneously.

if rank == 0:

    print()
    print("=" * 72)
    print("DISTRIBUTED PRIMITIVES TEST PASSED")
    print("=" * 72)

    print(
        f"World size     : "
        f"{world_size}"
    )

    print(
        f"Reduced value  : "
        f"{value.item():.1f}"
    )

    print(
        f"Expected value : "
        f"{expected_sum:.1f}"
    )

    print("=" * 72)

# cleanup 
dist.destroy_process_group()
"""
DDP synchronization smoke test.

This script demonstrates three important DDP behaviors:

1. Each rank owns its own model replica.
2. DDP synchronizes the initial model state.
3. During backward(), DDP synchronizes gradients.
4. After optimizer.step(), all replicas remain identical.

We intentionally initialize each rank differently before wrapping
the model in DDP so that we can directly observe synchronization.

Run with:

    torchrun --standalone --nproc-per-node=2 \
        scripts/09_test_ddp_sync.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn.parallel import (
    DistributedDataParallel as DDP,
)

# distributed env 
rank = int(
    os.environ["RANK"]
)

local_rank = int(
    os.environ["LOCAL_RANK"]
)

world_size = int(
    os.environ["WORLD_SIZE"]
)


dist.init_process_group(
    backend="gloo"
)

# helper 
def gather_tensor(tensor,):
    """
    Gather one tensor from every rank, every rank receives the complete list
    """
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered

# Rank 0:
#     seed = 100
#
# Rank 1:
#     seed = 101
#
# Therefore the models should initially have different weights.

torch.manual_seed(100 + rank)

model = nn.Linear(in_features=4, out_features=1, bias=False)

# save weight before ddp wrapping 

weights_before_ddp = (
    model.weight.detach().clone()
)
before_all_ranks = gather_tensor(weights_before_ddp)
if rank == 0:
    print()
    print("=" * 76)
    print("BEFORE DDP")
    print("=" * 76)

    for process_rank, weights in enumerate(before_all_ranks):
        print(
            f"rank {process_rank} weights: "
            f"{weights.flatten().tolist()}"

        )

# wrap model in DistributedDataParallel 
# Because this is a CPU/Gloo test:
#
#     device_ids is NOT supplied.
#
# Later with CUDA:
#
#     DDP(
#         model,
#         device_ids=[local_rank]
#     )

ddp_model = DDP(model)
weights_after_ddp = (
    ddp_model
    .module
    .weight
    .detach()
    .clone()
)


after_all_ranks = gather_tensor(
    weights_after_ddp
)


if rank == 0:

    print()
    print("=" * 76)
    print("AFTER DDP INITIALIZATION")
    print("=" * 76)

    for process_rank, weights in enumerate(
        after_all_ranks
    ):

        print(
            f"Rank {process_rank} weights: "
            f"{weights.flatten().tolist()}"
        )

# all replicas now contain identical parameters
reference_weights = (after_all_ranks[0])

for weights in after_all_ranks:
    assert torch.allclose(weights, reference_weights,)

#different data on different ranks
# We deliberately give each rank a different training example.
#
# Rank 0:
#     x = [1, 0, 0, 0]
#     y = 1
#
# Rank 1:
#     x = [0, 1, 0, 0]
#     y = -1
#
# Therefore the LOCAL gradients before synchronization would be
# different.

if rank == 0:
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    y = torch.tensor([[1.0]])

else:
    x = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    y = torch.tensor([[-1.0]])

optimizer = torch.optim.SGD(
    ddp_model.parameters(),
    lr=0.1,
)


criterion = nn.MSELoss()


optimizer.zero_grad(set_to_none=True)

# forward

predictions = ddp_model(x)
loss = criterion(predictions, y)

print(
    f"[rank {rank}] "
    f"local loss before backward = "
    f"{loss.item():.6f}",
    flush=True,
)

# This line is extremely important.
#
# Each process first computes gradients from its own local example.
#
# DDP then synchronizes gradients between replicas as part of the
# backward pass.
loss.backward()
gradient = (ddp_model.module.weight.grad.detach().clone())

all_gradient = gather_tensor(gradient)

if rank == 0:
    print()
    print("=" * 76)
    print("GRADIENTS AFTER DDP BACKWARD")
    print("=" * 76)

    for process_rank, grad in enumerate(
        all_gradient
    ):

        print(
            f"Rank {process_rank} gradient: "
            f"{grad.flatten().tolist()}"
        )

# sync grad should be same in every rank 
reference_gradient = all_gradient[0]

for grad in all_gradient:
    assert torch.allclose(grad, reference_gradient,)

# optimizer step 

optimizer.step()
updated_weights = (ddp_model.module.weight.detach().clone())

all_updated_weights = gather_tensor(updated_weights)

if rank == 0:

    print()
    print("=" * 76)
    print("PARAMETERS AFTER OPTIMIZER STEP")
    print("=" * 76)

    for process_rank, weights in enumerate(
        all_updated_weights
    ):

        print(
            f"Rank {process_rank} weights: "
            f"{weights.flatten().tolist()}"
        )

# all replica still must agree after update 

reference_updated_weights = all_updated_weights[0]

for weights in all_updated_weights:
    assert torch.allclose(weights, reference_updated_weights)

# final

dist.barrier()

if rank == 0:

    initial_different = (
        not torch.allclose(
            before_all_ranks[0],
            before_all_ranks[1],
        )
    )


    after_ddp_equal = torch.allclose(
        after_all_ranks[0],
        after_all_ranks[1],
    )


    gradients_equal = torch.allclose(
        all_gradient[0],
        all_gradient[1],
    )


    updated_equal = torch.allclose(
        all_updated_weights[0],
        all_updated_weights[1],
    )


    print()
    print("=" * 76)
    print("DDP SYNCHRONIZATION AUDIT")
    print("=" * 76)

    print(
        f"Initial weights different : "
        f"{initial_different}"
    )

    print(
        f"After DDP weights equal   : "
        f"{after_ddp_equal}"
    )

    print(
        f"Gradients synchronized    : "
        f"{gradients_equal}"
    )

    print(
        f"Updated weights equal     : "
        f"{updated_equal}"
    )


    assert initial_different
    assert after_ddp_equal
    assert gradients_equal
    assert updated_equal


    print()
    print(
        "DDP SYNCHRONIZATION TEST PASSED"
    )

    print("=" * 76)

# cleanp 

dist.destroy_process_group()
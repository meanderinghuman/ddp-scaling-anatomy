"""
DistributedSampler smoke test.

Purpose
-------
Verify that different distributed ranks receive different subsets
of the same dataset.

Run with:

    torchrun --standalone --nproc-per-node=2 \
        scripts/08_test_distributed_sampler.py
"""
import os

import torch
import torch.distributed as dist

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from torch.utils.data.distributed import (
    DistributedSampler,
)

# simple datset 
class NumberDataset(Dataset):
    """
    Dataset containing the integers:

        0, 1, 2, ..., 15

    Using simple integers makes it easy to inspect which samples
    each distributed rank receives.
    """

    def __init__(self):
        self.values = torch.arange(16, dtype=torch.long,)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index,):
        return self.values[index]

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

# dataset 

dataset = NumberDataset()

# distributedSamples 
sampler = DistributedSampler(
    dataset,
    # nr of processes
    num_replicas = world_size,
    # which process
    rank=rank,
    shuffle=True,
    seed=42,
    # dataset has 16 sample, world_size=2, no pad or drop
    drop_last=False,
    )


# DistributedSampler uses:
#
#     seed + epoch
#
# to generate its shuffle.
#
# Calling set_epoch(epoch) means:
#
#     epoch 0 → one ordering
#     epoch 1 → another ordering
#     epoch 2 → another ordering
#
# But all ranks still agree on the same global permutation.

sampler.set_epoch(0)

# dataloader 
# Do NOT use:
#
#     shuffle=True
#
# when a sampler is explicitly supplied.
#
# The sampler now controls sample ordering.
loader = DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=False)

# inspect samples received by this rank 

local_samples = []

for batch in loader:
    local_samples.extend(batch.tolist())

print(
    f"[rank {rank}] "
    f"local_rank={local_rank} "
    f"samples={local_samples}",
    flush=True,
)

# gather_object is useful here because local_samples is a Python list
# rather than a tensor.
# gather all rank result onto rank 0

gathered_samples = (
    [None for _ in range(world_size)] if rank == 0 else None
)

dist.gather_object(local_samples, gathered_samples, dst=0,)

if rank == 0:

    print()
    print("=" * 72)
    print("DISTRIBUTED SAMPLER AUDIT")
    print("=" * 72)


    for process_rank, samples in enumerate(
        gathered_samples
    ):

        print(
            f"Rank {process_rank}: "
            f"{samples}"
        )


    # Flatten all rank subsets.
    all_samples = [
        sample
        for rank_samples in gathered_samples
        for sample in rank_samples
    ]


    unique_samples = set(
        all_samples
    )


    expected_samples = set(
        range(len(dataset))
    )

    # coverage -> every dataset must appear somewhere
    coverage_ok = (unique_samples == expected_samples)

    # duplicate check 
    no_duplicates = (len(all_samples) == len(unique_samples))



    print()
    print(
        f"Total dataset samples : "
        f"{len(dataset)}"
    )

    print(
        f"Samples across ranks  : "
        f"{len(all_samples)}"
    )

    print(
        f"Unique samples        : "
        f"{len(unique_samples)}"
    )

    print(
        f"Full coverage         : "
        f"{coverage_ok}"
    )

    print(
        f"No duplicates         : "
        f"{no_duplicates}"
    )


    assert coverage_ok

    assert no_duplicates


    print()
    print(
        "DISTRIBUTED SAMPLER TEST PASSED"
    )

    print("=" * 72)


# synchronize and clean up 

dist.barrier()
dist.destroy_process_group()

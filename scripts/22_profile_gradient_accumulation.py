"""
Profile Phase 3 gradient accumulation.

Compare:
    naive
    DDP.no_sync()

Configuration:
    4 GPUs
    microbatch/rank = 4
    accumulation_steps = 4
    effective global batch = 64

Profiler window:
    wait   = 8 microbatches
    warmup = 4 microbatches
    active = 16 microbatches

The active region therefore contains exactly:

    16 / 4 = 4 optimizer updates

Expected DDP all-reduces:
    naive   ~= 48
    no_sync ~= 12
"""

import argparse
import csv
import json
import os

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from torch.nn.parallel import (
    DistributedDataParallel as DDP,
)

from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
)

from torch.utils.data import DataLoader

from torch.utils.data.distributed import (
    DistributedSampler,
)

from ddp_lab.training.common import (
    build_optimizer,
    build_workload,
    load_config,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "naive",
            "no_sync",
        ],
        required=True,
    )

    parser.add_argument(
        "--tag",
        required=True,
    )

    return parser.parse_args()


def get_float(
    event,
    names,
):

    for name in names:

        if hasattr(event, name):

            value = getattr(
                event,
                name,
            )

            if value is not None:
                return float(value)

    return 0.0


def event_row(
    event,
):

    return {
        "name":
            str(event.key),

        "count":
            int(event.count),

        "self_cpu_time_total_us":
            float(
                event.self_cpu_time_total
            ),

        "self_device_time_total_us":
            get_float(
                event,
                [
                    "self_device_time_total",
                    "self_cuda_time_total",
                ],
            ),
    }


def is_comm_event(
    name,
):

    name = name.lower()

    return any(
        token in name
        for token in [
            "nccl",
            "all_reduce",
            "allreduce",
            "c10d",
        ]
    )


def main():

    args = parse_args()


    rank = int(
        os.environ["RANK"]
    )

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    world_size = int(
        os.environ["WORLD_SIZE"]
    )


    if world_size != 4:

        raise RuntimeError(
            "Expected exactly 4 GPUs."
        )


    torch.cuda.set_device(
        local_rank
    )

    device = torch.device(
        "cuda",
        local_rank,
    )


    dist.init_process_group(
        backend="nccl",
        device_id=device,
    )


    config = load_config(
        args.config
    )


    if config["workload"] != "unet_heat":

        raise RuntimeError(
            "Profiler is restricted to U-Net."
        )


    set_seed(
        int(config["seed"])
    )


    microbatch_size = 4
    accumulation_steps = 4


    (
        train_dataset,
        _,
        _,
        model,
        criterion,
    ) = build_workload(
        workload="unet_heat",
        project_root=ROOT,
        config=config,
    )


    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(config["seed"]),
        drop_last=False,
    )


    loader = DataLoader(
        train_dataset,
        batch_size=microbatch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )


    model = model.to(
        device
    )


    model = DDP(
        model,
        device_ids=[
            local_rank
        ],
        output_device=local_rank,
    )


    optimizer = build_optimizer(
        model,
        config,
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )


    output_dir = (
        ROOT
        / "results"
        / "rosi_v100"
        / "optimization"
        / "profiling"
        / args.tag
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    WAIT = 8
    WARMUP = 4
    ACTIVE = 16

    TOTAL = (
        WAIT
        + WARMUP
        + ACTIVE
    )


    def save_profile(
        prof,
    ):

        events = [
            event_row(event)
            for event in
            prof.key_averages()
        ]


        comm = [
            row
            for row in events
            if is_comm_event(
                row["name"]
            )
        ]


        with (
            output_dir
            / f"communication_rank{rank}.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                comm,
                f,
                indent=2,
            )


        with (
            output_dir
            / f"events_rank{rank}.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    events[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                events
            )


        table = (
            prof
            .key_averages()
            .table(
                sort_by="self_cuda_time_total",
                row_limit=60,
            )
        )


        with (
            output_dir
            / f"top_ops_rank{rank}.txt"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                table
            )


        if rank == 0:

            prof.export_chrome_trace(
                str(
                    output_dir
                    / "trace_rank0.json"
                )
            )


    prof_schedule = schedule(
        wait=WAIT,
        warmup=WARMUP,
        active=ACTIVE,
        repeat=1,
    )


    sampler.set_epoch(
        0
    )


    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )


    dist.barrier()


    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=prof_schedule,
        on_trace_ready=save_profile,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:


        for micro_idx, (
            inputs,
            targets,
        ) in enumerate(
            loader
        ):


            if micro_idx >= TOTAL:
                break


            inputs = inputs.to(
                device,
                non_blocking=True,
            )


            targets = targets.to(
                device,
                non_blocking=True,
            )


            position = (
                micro_idx
                % accumulation_steps
            )


            final_microbatch = (
                position
                ==
                accumulation_steps - 1
            )


            if (
                args.mode == "no_sync"
                and not final_microbatch
            ):

                sync_context = (
                    model.no_sync()
                )

            else:

                sync_context = (
                    nullcontext()
                )


            # Forward MUST be inside no_sync().
            with sync_context:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):

                    prediction = model(
                        inputs
                    )

                    loss = criterion(
                        prediction,
                        targets,
                    )

                    loss = (
                        loss
                        / accumulation_steps
                    )


                scaler.scale(
                    loss
                ).backward()


            if final_microbatch:

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )


            prof.step()


    torch.cuda.synchronize(
        device
    )

    dist.barrier()


    metadata = {
        "mode":
            args.mode,

        "rank":
            rank,

        "world_size":
            world_size,

        "microbatch_size_per_rank":
            microbatch_size,

        "accumulation_steps":
            accumulation_steps,

        "effective_global_batch_size":
            (
                microbatch_size
                * world_size
                * accumulation_steps
            ),

        "profiler_wait_microbatches":
            WAIT,

        "profiler_warmup_microbatches":
            WARMUP,

        "profiler_active_microbatches":
            ACTIVE,

        "active_optimizer_updates":
            ACTIVE
            // accumulation_steps,

        "expected_allreduces_per_optimizer_update":
            (
                12
                if args.mode == "naive"
                else 3
            ),
    }


    with (
        output_dir
        / f"metadata_rank{rank}.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


    if rank == 0:

        print(
            "=" * 80
        )

        print(
            "GRADIENT ACCUMULATION PROFILER COMPLETE"
        )

        print(
            "=" * 80
        )

        print(
            f"Mode                    : {args.mode}"
        )

        print(
            f"Active microbatches     : {ACTIVE}"
        )

        print(
            f"Active optimizer updates: {ACTIVE // accumulation_steps}"
        )

        print(
            f"Output                  : {output_dir}"
        )

        print(
            "=" * 80
        )


    dist.destroy_process_group()


if __name__ == "__main__":
    main()

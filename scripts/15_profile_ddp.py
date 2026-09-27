"""
Profile PyTorch DistributedDataParallel training.

Purpose
-------
This is NOT a throughput benchmark.

It is a short profiling experiment designed to study:

    - forward computation
    - backward computation
    - optimizer work
    - DDP/NCCL gradient communication
    - CUDA kernel activity

The frozen benchmark trainer in scripts/10_train_ddp.py is not modified.

Profiling schedule
------------------
Default:

    wait   = 5 steps
    warmup = 3 steps
    active = 10 steps

The first five steps allow normal model/DDP warm-up.

The profiler then performs its own three-step warm-up and records
ten active iterations.

Only rank 0 exports the full Chrome/Perfetto trace to reduce storage.
Every rank exports aggregated profiler statistics.
"""

import argparse
import csv
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist

from torch.nn.parallel import (
    DistributedDataParallel as DDP,
)

from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
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


# =====================================================================
# Paths
# =====================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# =====================================================================
# Arguments
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Profile DDP compute and NCCL communication."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help=(
            "Unique experiment name, for example "
            "unet_world4 or resnet_world1."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--wait-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--active-steps",
        type=int,
        default=10,
    )

    return parser.parse_args()


# =====================================================================
# torchrun environment
# =====================================================================

def read_distributed_environment():

    required = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
    ]

    missing = [
        key
        for key in required
        if key not in os.environ
    ]

    if missing:
        raise RuntimeError(
            "Missing torchrun environment variables: "
            f"{missing}"
        )

    rank = int(
        os.environ["RANK"]
    )

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    world_size = int(
        os.environ["WORLD_SIZE"]
    )

    return (
        rank,
        local_rank,
        world_size,
    )


# =====================================================================
# Profiler event utilities
# =====================================================================

def get_attr_float(
    obj,
    names,
):

    for name in names:

        if hasattr(obj, name):

            value = getattr(
                obj,
                name,
            )

            if value is not None:
                return float(value)

    return 0.0


def profiler_event_row(event):

    # PyTorch has renamed some CUDA timing properties over time.
    # Check the modern generic "device" names first, then legacy CUDA names.

    self_device_us = get_attr_float(
        event,
        [
            "self_device_time_total",
            "self_cuda_time_total",
        ],
    )

    device_us = get_attr_float(
        event,
        [
            "device_time_total",
            "cuda_time_total",
        ],
    )

    return {
        "name":
            str(event.key),

        "count":
            int(event.count),

        "self_cpu_time_total_us":
            float(
                event.self_cpu_time_total
            ),

        "cpu_time_total_us":
            float(
                event.cpu_time_total
            ),

        "self_device_time_total_us":
            self_device_us,

        "device_time_total_us":
            device_us,
    }


def is_communication_event(
    name,
):

    lower = name.lower()

    communication_tokens = [
        "nccl",
        "all_reduce",
        "allreduce",
        "c10d",
        "reduce_scatter",
        "all_gather",
        "broadcast",
    ]

    return any(
        token in lower
        for token in communication_tokens
    )


# =====================================================================
# Main
# =====================================================================

def main():

    args = parse_args()

    (
        rank,
        local_rank,
        world_size,
    ) = read_distributed_environment()


    # -----------------------------------------------------------------
    # CUDA / NCCL setup
    # -----------------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "Phase 2B profiling requires CUDA."
        )


    if not dist.is_nccl_available():

        raise RuntimeError(
            "NCCL is unavailable."
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


    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    config = load_config(
        args.config
    )

    workload = config[
        "workload"
    ]


    # Keep the same seed policy as the benchmark.
    set_seed(
        int(config["seed"])
    )


    global_batch_size = int(
        config["global_batch_size"]
    )


    if (
        global_batch_size
        % world_size
        != 0
    ):

        raise RuntimeError(
            "Global batch size must be divisible "
            "by world size."
        )


    local_batch_size = (
        global_batch_size
        // world_size
    )


    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "rosi_v100"
        / "profiling"
        / "runs"
        / args.tag
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -----------------------------------------------------------------
    # Workload
    # -----------------------------------------------------------------

    (
        train_dataset,
        _,
        _,
        model,
        criterion,
    ) = build_workload(
        workload=workload,
        project_root=PROJECT_ROOT,
        config=config,
    )


    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(config["seed"]),
        drop_last=False,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(
            args.num_workers > 0
        ),
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


    # -----------------------------------------------------------------
    # FP16 AMP — same V100 precision policy as baseline benchmark
    # -----------------------------------------------------------------

    requested_amp = bool(
        config["amp"]
    )

    amp_dtype_name = (
        config
        .get(
            "amp_dtype",
            "fp16",
        )
        .lower()
    )


    if amp_dtype_name != "fp16":

        raise RuntimeError(
            "The V100 profiler experiment expects "
            "amp_dtype: fp16."
        )


    use_amp = (
        requested_amp
        and device.type == "cuda"
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )


    # -----------------------------------------------------------------
    # Profiler support audit
    # -----------------------------------------------------------------

    supported = (
        torch.profiler
        .supported_activities()
    )


    activities = [
        ProfilerActivity.CPU,
    ]


    if (
        ProfilerActivity.CUDA
        in supported
    ):

        activities.append(
            ProfilerActivity.CUDA
        )

    else:

        if rank == 0:
            print(
                "WARNING: CUDA profiler activity "
                "not reported as supported."
            )


    # -----------------------------------------------------------------
    # Trace handler
    # -----------------------------------------------------------------

    def save_profile(
        prof,
    ):

        events = [
            profiler_event_row(event)
            for event
            in prof.key_averages()
        ]


        events.sort(
            key=lambda row:
                row[
                    "self_device_time_total_us"
                ],
            reverse=True,
        )


        event_json = (
            output_dir
            / f"events_rank{rank}.json"
        )


        with event_json.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                events,
                f,
                indent=2,
            )


        # -------------------------------------------------------------
        # CSV
        # -------------------------------------------------------------

        event_csv = (
            output_dir
            / f"events_rank{rank}.csv"
        )


        if events:

            with event_csv.open(
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


        # -------------------------------------------------------------
        # Communication-only events
        # -------------------------------------------------------------

        communication = [
            row
            for row in events
            if is_communication_event(
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
                communication,
                f,
                indent=2,
            )


        # -------------------------------------------------------------
        # Human-readable top-operator table
        # -------------------------------------------------------------

        table = (
            prof
            .key_averages()
            .table(
                sort_by=(
                    "self_cuda_time_total"
                ),
                row_limit=80,
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


        # -------------------------------------------------------------
        # Full trace only for rank 0.
        #
        # For this single-node symmetric experiment, rank 0 is enough
        # for visual timeline inspection while keeping trace size small.
        # -------------------------------------------------------------

        if rank == 0:

            trace_path = (
                output_dir
                / "trace_rank0.json"
            )

            prof.export_chrome_trace(
                str(trace_path)
            )


    # -----------------------------------------------------------------
    # Profile
    # -----------------------------------------------------------------

    profiler_schedule = schedule(
        wait=args.wait_steps,
        warmup=args.warmup_steps,
        active=args.active_steps,
        repeat=1,
    )


    total_steps = (
        args.wait_steps
        + args.warmup_steps
        + args.active_steps
    )


    if rank == 0:

        print(
            "=" * 88
        )

        print(
            "DDP / NCCL PROFILER"
        )

        print(
            "=" * 88
        )

        print(
            f"Workload           : {workload}"
        )

        print(
            f"World size         : {world_size}"
        )

        print(
            f"Global batch       : {global_batch_size}"
        )

        print(
            f"Local batch        : {local_batch_size}"
        )

        print(
            f"Wait steps         : {args.wait_steps}"
        )

        print(
            f"Profiler warmup    : {args.warmup_steps}"
        )

        print(
            f"Active steps       : {args.active_steps}"
        )

        print(
            f"Total steps        : {total_steps}"
        )

        print(
            f"AMP                : {use_amp}"
        )

        print(
            f"AMP dtype          : fp16"
        )

        print(
            f"Output             : {output_dir}"
        )

        print(
            "=" * 88
        )


    train_sampler.set_epoch(
        0
    )


    model.train()


    dist.barrier()


    with profile(
        activities=activities,
        schedule=profiler_schedule,
        on_trace_ready=save_profile,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    ) as prof:


        completed_steps = 0


        for step, (
            inputs,
            targets,
        ) in enumerate(
            train_loader
        ):


            if step >= total_steps:
                break


            inputs = inputs.to(
                device,
                non_blocking=True,
            )


            targets = targets.to(
                device,
                non_blocking=True,
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            # ---------------------------------------------------------
            # Forward
            # ---------------------------------------------------------

            with record_function(
                "PROFILE_FORWARD"
            ):

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):

                    output = model(
                        inputs
                    )

                    loss = criterion(
                        output,
                        targets,
                    )


            # ---------------------------------------------------------
            # Backward
            #
            # DDP gradient synchronization happens here.
            # ---------------------------------------------------------

            with record_function(
                "PROFILE_BACKWARD"
            ):

                if use_amp:

                    scaler.scale(
                        loss
                    ).backward()

                else:

                    loss.backward()


            # ---------------------------------------------------------
            # Optimizer
            # ---------------------------------------------------------

            with record_function(
                "PROFILE_OPTIMIZER"
            ):

                if use_amp:

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    optimizer.step()


            prof.step()

            completed_steps += 1


    torch.cuda.synchronize(
        device
    )


    dist.barrier()


    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    metadata = {
        "tag":
            args.tag,

        "workload":
            workload,

        "rank":
            rank,

        "local_rank":
            local_rank,

        "world_size":
            world_size,

        "hostname":
            socket.gethostname(),

        "device":
            torch.cuda.get_device_name(
                local_rank
            ),

        "global_batch_size":
            global_batch_size,

        "local_batch_size":
            local_batch_size,

        "wait_steps":
            args.wait_steps,

        "warmup_steps":
            args.warmup_steps,

        "active_steps":
            args.active_steps,

        "completed_steps":
            completed_steps,

        "amp":
            use_amp,

        "amp_dtype":
            "fp16",

        "pytorch":
            torch.__version__,

        "cuda":
            torch.version.cuda,
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

        print()
        print(
            "Profiler run complete."
        )

        print(
            f"Saved to: {output_dir}"
        )


    dist.destroy_process_group()


if __name__ == "__main__":
    main()

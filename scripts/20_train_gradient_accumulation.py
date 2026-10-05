"""
Phase 3 — Gradient accumulation communication experiment.

Workload:
    U-Net / Heat2D

Hardware experiment:
    4 GPUs
    local microbatch = 4
    accumulation steps = 4

Effective global batch:
    microbatch_per_rank
    x world_size
    x accumulation_steps

    4 x 4 x 4 = 64

Compared modes
--------------
naive:
    Every backward pass executes normal DDP gradient synchronization.

no_sync:
    First K-1 forward/backward passes are wrapped in DDP.no_sync().
    Only the final microbatch synchronizes gradients.

Both modes:
    - use the same data
    - use the same initialization
    - use the same optimizer
    - use the same FP16 AMP path
    - divide loss by accumulation_steps
    - perform optimizer.step() once per effective batch
"""

import argparse
import json
import os
import socket
import statistics
import time

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from torch.nn.parallel import (
    DistributedDataParallel as DDP,
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
            "Compare naive DDP gradient accumulation "
            "against DDP.no_sync()."
        )
    )

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
        "--microbatch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--timing-warmup-microbatches",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--result-path",
        type=Path,
        required=True,
    )

    return parser.parse_args()


# =====================================================================
# Distributed helpers
# =====================================================================

def distributed_sum(
    value,
    device,
):

    tensor = torch.tensor(
        float(value),
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM,
    )

    return float(
        tensor.item()
    )


def distributed_max(
    value,
    device,
):

    tensor = torch.tensor(
        float(value),
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.MAX,
    )

    return float(
        tensor.item()
    )


# =====================================================================
# Main
# =====================================================================

def main():

    args = parse_args()


    # -----------------------------------------------------------------
    # torchrun environment
    # -----------------------------------------------------------------

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
            "Phase 3 experiment is defined for exactly 4 GPUs."
        )


    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
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
    # Config
    # -----------------------------------------------------------------

    config = load_config(
        args.config
    )


    workload = config[
        "workload"
    ]


    if workload != "unet_heat":

        raise RuntimeError(
            "Phase 3 is intentionally restricted to U-Net / Heat2D."
        )


    set_seed(
        int(config["seed"])
    )


    # -----------------------------------------------------------------
    # Effective batch
    # -----------------------------------------------------------------

    microbatch_size = int(
        args.microbatch_size
    )

    accumulation_steps = int(
        args.accumulation_steps
    )


    global_microbatch_size = (
        microbatch_size
        * world_size
    )


    effective_global_batch_size = (
        global_microbatch_size
        * accumulation_steps
    )


    # -----------------------------------------------------------------
    # Dataset + model
    # -----------------------------------------------------------------

    (
        train_dataset,
        val_dataset,
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


    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=microbatch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=microbatch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(
            args.num_workers > 0
        ),
    )


    if (
        len(train_loader)
        % accumulation_steps
        != 0
    ):

        raise RuntimeError(
            "Number of training microbatches must be divisible "
            "by accumulation_steps for this controlled experiment."
        )


    optimizer_steps_per_epoch = (
        len(train_loader)
        // accumulation_steps
    )


    model = model.to(
        device
    )


    ddp_model = DDP(
        model,
        device_ids=[
            local_rank
        ],
        output_device=local_rank,
    )


    optimizer = build_optimizer(
        ddp_model,
        config,
    )


    # -----------------------------------------------------------------
    # AMP
    # -----------------------------------------------------------------

    use_amp = bool(
        config.get(
            "amp",
            True,
        )
    )


    amp_dtype = (
        config
        .get(
            "amp_dtype",
            "fp16",
        )
        .lower()
    )


    if amp_dtype != "fp16":

        raise RuntimeError(
            "V100 Phase 3 experiment expects amp_dtype=fp16."
        )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )


    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------

    if rank == 0:

        print(
            "=" * 96
        )

        print(
            "PHASE 3 — DDP GRADIENT ACCUMULATION"
        )

        print(
            "=" * 96
        )

        print(
            f"Mode                     : {args.mode}"
        )

        print(
            f"World size               : {world_size}"
        )

        print(
            f"Microbatch / rank        : {microbatch_size}"
        )

        print(
            f"Global microbatch        : {global_microbatch_size}"
        )

        print(
            f"Accumulation steps       : {accumulation_steps}"
        )

        print(
            f"Effective global batch   : {effective_global_batch_size}"
        )

        print(
            f"Microbatches / epoch     : {len(train_loader)}"
        )

        print(
            f"Optimizer steps / epoch  : {optimizer_steps_per_epoch}"
        )

        print(
            f"Epochs                    : {args.epochs}"
        )

        print(
            f"AMP                       : {use_amp}"
        )

        print(
            "=" * 96
        )


    history = []


    # =================================================================
    # Training
    # =================================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):


        train_sampler.set_epoch(
            epoch - 1
        )


        ddp_model.train()


        torch.cuda.reset_peak_memory_stats(
            device
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        local_train_loss_sum = 0.0
        local_train_examples = 0


        local_measured_examples = 0

        measured_start = None


        dist.barrier()

        torch.cuda.synchronize(
            device
        )


        train_loop_start = (
            time.perf_counter()
        )


        completed_optimizer_steps = 0


        for micro_idx, (
            inputs,
            targets,
        ) in enumerate(
            train_loader
        ):


            inputs = inputs.to(
                device,
                non_blocking=True,
            )


            targets = targets.to(
                device,
                non_blocking=True,
            )


            # ---------------------------------------------------------
            # Start steady timing after complete warm-up microbatches.
            #
            # Warmup=8 is intentionally divisible by K=4 so timing
            # begins at an accumulation-window boundary.
            # ---------------------------------------------------------

            if (
                micro_idx
                == args.timing_warmup_microbatches
            ):

                torch.cuda.synchronize(
                    device
                )

                measured_start = (
                    time.perf_counter()
                )


            if (
                micro_idx
                >= args.timing_warmup_microbatches
            ):

                local_measured_examples += (
                    inputs.shape[0]
                )


            micro_in_window = (
                micro_idx
                % accumulation_steps
            )


            final_microbatch = (
                micro_in_window
                ==
                accumulation_steps - 1
            )


            # ---------------------------------------------------------
            # no_sync must cover BOTH forward and backward.
            # ---------------------------------------------------------

            if (
                args.mode == "no_sync"
                and not final_microbatch
            ):

                sync_context = (
                    ddp_model.no_sync()
                )

            else:

                sync_context = (
                    nullcontext()
                )


            with sync_context:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):

                    predictions = (
                        ddp_model(
                            inputs
                        )
                    )


                    raw_loss = criterion(
                        predictions,
                        targets,
                    )


                    # Average across the complete effective batch.
                    scaled_loss = (
                        raw_loss
                        / accumulation_steps
                    )


                if use_amp:

                    scaler.scale(
                        scaled_loss
                    ).backward()

                else:

                    scaled_loss.backward()


            local_train_loss_sum += (
                float(
                    raw_loss.detach()
                )
                * inputs.shape[0]
            )


            local_train_examples += (
                inputs.shape[0]
            )


            # ---------------------------------------------------------
            # One optimizer update per accumulation window.
            # ---------------------------------------------------------

            if final_microbatch:

                if use_amp:

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    optimizer.step()


                optimizer.zero_grad(
                    set_to_none=True
                )


                completed_optimizer_steps += 1


        # -------------------------------------------------------------
        # Training timing
        # -------------------------------------------------------------

        torch.cuda.synchronize(
            device
        )


        train_loop_end = (
            time.perf_counter()
        )


        if measured_start is None:

            raise RuntimeError(
                "Timing warmup exceeded number of microbatches."
            )


        local_measured_time = (
            train_loop_end
            - measured_start
        )


        local_train_wall = (
            train_loop_end
            - train_loop_start
        )


        global_measured_examples = (
            distributed_sum(
                local_measured_examples,
                device,
            )
        )


        measured_time_max = (
            distributed_max(
                local_measured_time,
                device,
            )
        )


        train_wall_max = (
            distributed_max(
                local_train_wall,
                device,
            )
        )


        throughput = (
            global_measured_examples
            / measured_time_max
        )


        global_train_loss_sum = (
            distributed_sum(
                local_train_loss_sum,
                device,
            )
        )


        global_train_examples = (
            distributed_sum(
                local_train_examples,
                device,
            )
        )


        train_loss = (
            global_train_loss_sum
            / global_train_examples
        )


        # =============================================================
        # Validation
        # =============================================================

        ddp_model.eval()


        local_val_loss_sum = 0.0
        local_val_examples = 0


        with torch.no_grad():

            for (
                inputs,
                targets,
            ) in val_loader:


                inputs = inputs.to(
                    device,
                    non_blocking=True,
                )


                targets = targets.to(
                    device,
                    non_blocking=True,
                )


                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):

                    predictions = (
                        ddp_model(
                            inputs
                        )
                    )


                    val_loss = criterion(
                        predictions,
                        targets,
                    )


                local_val_loss_sum += (
                    float(
                        val_loss
                    )
                    * inputs.shape[0]
                )


                local_val_examples += (
                    inputs.shape[0]
                )


        global_val_loss_sum = (
            distributed_sum(
                local_val_loss_sum,
                device,
            )
        )


        global_val_examples = (
            distributed_sum(
                local_val_examples,
                device,
            )
        )


        val_mse = (
            global_val_loss_sum
            / global_val_examples
        )


        peak_vram_gb = (
            torch.cuda.max_memory_allocated(
                device
            )
            / 1024**3
        )


        peak_vram_max = (
            distributed_max(
                peak_vram_gb,
                device,
            )
        )


        epoch_record = {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "val_mse":
                val_mse,

            "throughput_samples_per_sec":
                throughput,

            "measured_train_time_sec":
                measured_time_max,

            "train_loop_wall_time_sec":
                train_wall_max,

            "global_measured_examples":
                int(
                    global_measured_examples
                ),

            "optimizer_steps":
                completed_optimizer_steps,

            "peak_vram_per_rank_gb":
                peak_vram_max,
        }


        history.append(
            epoch_record
        )


        if rank == 0:

            print(
                f"epoch {epoch:2d} | "
                f"train MSE {train_loss:.6f} | "
                f"val MSE {val_mse:.6f} | "
                f"Q {throughput:.1f} fields/s | "
                f"train wall {train_wall_max:.3f}s | "
                f"opt steps {completed_optimizer_steps} | "
                f"VRAM {peak_vram_max:.3f} GiB"
            )


    # =================================================================
    # Summary
    # =================================================================

    steady = (
        history[1:]
        if len(history) > 1
        else history
    )


    steady_throughputs = [
        row[
            "throughput_samples_per_sec"
        ]
        for row in steady
    ]


    steady_wall_times = [
        row[
            "train_loop_wall_time_sec"
        ]
        for row in steady
    ]


    result = {
        "experiment":
            "gradient_accumulation",

        "mode":
            args.mode,

        "workload":
            workload,

        "world_size":
            world_size,

        "microbatch_size_per_rank":
            microbatch_size,

        "global_microbatch_size":
            global_microbatch_size,

        "accumulation_steps":
            accumulation_steps,

        "effective_global_batch_size":
            effective_global_batch_size,

        "epochs":
            args.epochs,

        "timing_warmup_microbatches":
            args.timing_warmup_microbatches,

        "optimizer_steps_per_epoch":
            optimizer_steps_per_epoch,

        "steady_state_policy":
            "exclude epoch 1",

        "steady_state_throughput_samples_per_sec":
            statistics.mean(
                steady_throughputs
            ),

        "steady_state_throughput_std":
            (
                statistics.stdev(
                    steady_throughputs
                )
                if len(
                    steady_throughputs
                ) > 1
                else 0.0
            ),

        "mean_steady_train_wall_time_sec":
            statistics.mean(
                steady_wall_times
            ),

        "history":
            history,

        "hostname":
            socket.gethostname(),

        "device":
            torch.cuda.get_device_name(
                local_rank
            ),

        "pytorch_version":
            torch.__version__,

        "cuda_version":
            torch.version.cuda,

        "amp":
            use_amp,

        "amp_dtype":
            "fp16",
    }


    # -----------------------------------------------------------------
    # Rank-0 result output
    # -----------------------------------------------------------------

    if rank == 0:

        output_path = (
            args.result_path
        )


        if not output_path.is_absolute():

            output_path = (
                PROJECT_ROOT
                / output_path
            )


        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
            )


        print()
        print(
            "=" * 96
        )

        print(
            "PHASE 3 RUN COMPLETE"
        )

        print(
            "=" * 96
        )

        print(
            f"Mode                    : {args.mode}"
        )

        print(
            f"Effective global batch  : {effective_global_batch_size}"
        )

        print(
            f"Steady throughput       : "
            f"{result['steady_state_throughput_samples_per_sec']:.2f} fields/s"
        )

        print(
            f"Mean steady train wall  : "
            f"{result['mean_steady_train_wall_time_sec']:.3f} s"
        )

        print(
            f"Final validation MSE    : "
            f"{history[-1]['val_mse']:.6f}"
        )

        print(
            f"Result                  : {output_path}"
        )

        print(
            "=" * 96
        )


    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

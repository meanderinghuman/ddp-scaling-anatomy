"""
Distributed training benchmark using PyTorch DistributedDataParallel.

Supported workloads
-------------------
1. ResNet18 / CIFAR-100
2. U-Net / 2D heat-equation regression

Supported execution modes
-------------------------
CPU:
    torchrun + Gloo
    Used only for distributed correctness testing.

GPU:
    torchrun + NCCL
    Used for the real scaling benchmark.

Important benchmark rule
------------------------
We perform STRONG SCALING.

The GLOBAL batch size remains constant as GPU count increases.

Example, ResNet:

    world_size = 1
        global batch = 256
        local batch  = 256

    world_size = 2
        global batch = 256
        local batch  = 128

    world_size = 4
        global batch = 256
        local batch  = 64

Therefore:

    local_batch_size
        =
    global_batch_size / world_size

Each rank:
    - owns one complete model replica
    - receives its own data shard
    - computes its own forward pass
    - computes local gradients

DDP:
    - synchronizes gradients during backward()

Each rank then:
    - executes the same optimizer update

Only rank 0:
    - prints epoch summaries
    - saves checkpoints
    - saves benchmark JSON files
"""

import argparse
import json
import math
import os
import socket
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


# =====================================================================
# Paths
# =====================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# =====================================================================
# Command-line arguments
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "DistributedDataParallel training benchmark."
        )
    )


    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to workload YAML configuration.",
    )


    # -------------------------------------------------------------
    # Device selection
    #
    # auto:
    #     CUDA if available, otherwise CPU.
    #
    # cpu:
    #     force CPU.
    #
    # cuda:
    #     require CUDA.
    # -------------------------------------------------------------

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )


    # -------------------------------------------------------------
    # Communication backend
    #
    # auto:
    #
    #     CUDA -> NCCL
    #     CPU  -> Gloo
    # -------------------------------------------------------------

    parser.add_argument(
        "--backend",
        choices=[
            "auto",
            "gloo",
            "nccl",
        ],
        default="auto",
    )


    # Optional DataLoader override.
    #
    # This is particularly useful for our laptop CPU smoke test,
    # where multiple DDP processes + multiple DataLoader workers
    # could unnecessarily overload the CPU.
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )


    # Smoke-test controls.
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
    )


    return parser.parse_args()


# =====================================================================
# torchrun environment
# =====================================================================

def read_torchrun_environment():
    """
    Read the distributed identifiers supplied by torchrun.

    RANK
        Unique process ID across the entire distributed job.

    LOCAL_RANK
        Process ID relative to the current node.

    WORLD_SIZE
        Total number of participating processes.
    """

    required_variables = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
    ]


    missing = [
        name
        for name in required_variables
        if name not in os.environ
    ]


    if missing:

        raise RuntimeError(
            "Missing torchrun environment variables: "
            f"{missing}. "
            "Launch this script using torchrun."
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
# Device + backend
# =====================================================================

def resolve_device_and_backend(
    args,
    local_rank,
):
    """
    Select the compute device and communication backend.
    """

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    if args.device == "auto":

        device_type = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device_type = args.device


    if (
        device_type == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "--device cuda was requested, "
            "but CUDA is unavailable."
        )


    if device_type == "cuda":

        visible_gpu_count = (
            torch.cuda.device_count()
        )


        if local_rank >= visible_gpu_count:

            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but this process "
                f"can see only {visible_gpu_count} GPU(s). "
                "Use one DDP process per visible GPU."
            )


        # Each process exclusively owns one GPU.
        torch.cuda.set_device(
            local_rank
        )


        device = torch.device(
            "cuda",
            local_rank,
        )

    else:

        device = torch.device(
            "cpu"
        )


    # -------------------------------------------------------------
    # Backend
    # -------------------------------------------------------------

    if args.backend == "auto":

        backend = (
            "nccl"
            if device.type == "cuda"
            else "gloo"
        )

    else:

        backend = args.backend


    if (
        backend == "nccl"
        and device.type != "cuda"
    ):

        raise RuntimeError(
            "NCCL requires CUDA devices."
        )


    if (
        backend == "nccl"
        and not dist.is_nccl_available()
    ):

        raise RuntimeError(
            "NCCL backend is unavailable "
            "in this PyTorch installation."
        )


    if (
        backend == "gloo"
        and not dist.is_gloo_available()
    ):

        raise RuntimeError(
            "Gloo backend is unavailable "
            "in this PyTorch installation."
        )


    return (
        device,
        backend,
    )


# =====================================================================
# Device synchronization
# =====================================================================

def synchronize_device(
    device,
):
    """
    CUDA kernels execute asynchronously relative to Python.

    For accurate GPU timing we must wait until queued GPU work
    has actually completed.

    CPU execution is synchronous, so no action is required there.
    """

    if device.type == "cuda":

        torch.cuda.synchronize(
            device
        )


# =====================================================================
# Distributed reduction helpers
# =====================================================================

def distributed_sum(
    values,
    device,
):
    """
    Sum one or more scalar values across all ranks.

    Example:

        rank 0:
            loss_sum = 10
            examples = 100

        rank 1:
            loss_sum = 12
            examples = 100

    After reduction:

        loss_sum = 22
        examples = 200
    """

    tensor = torch.tensor(
        values,
        dtype=torch.float64,
        device=device,
    )


    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM,
    )


    return tensor.cpu().tolist()


def distributed_max(
    value,
    device,
):
    """
    Return the maximum scalar value across all ranks.

    This is important for timing.

    Distributed training proceeds at the speed of the slowest rank,
    therefore benchmark time should be based on:

        max(rank_0_time,
            rank_1_time,
            ...)
    """

    tensor = torch.tensor(
        [value],
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
    # Read torchrun identifiers
    # -----------------------------------------------------------------

    (
        rank,
        local_rank,
        world_size,
    ) = read_torchrun_environment()


    # -----------------------------------------------------------------
    # Device/backend
    # -----------------------------------------------------------------

    (
        device,
        backend,
    ) = resolve_device_and_backend(
        args=args,
        local_rank=local_rank,
    )


    # -----------------------------------------------------------------
    # Initialize process group
    #
    # torchrun automatically provides:
    #
    #     MASTER_ADDR
    #     MASTER_PORT
    #     RANK
    #     WORLD_SIZE
    #
    # init_process_group() therefore uses env:// discovery.
    # -----------------------------------------------------------------

    if backend == "nccl":

        dist.init_process_group(
            backend=backend,
            device_id=device,
        )

    else:

        dist.init_process_group(
            backend=backend,
        )


    try:

        # =============================================================
        # Configuration
        # =============================================================

        config = load_config(
            args.config
        )


        workload = str(
            config["workload"]
        )


        seed = int(
            config["seed"]
        )


        global_batch_size = int(
            config["global_batch_size"]
        )


        num_epochs = int(
            config["epochs"]
        )


        warmup_batches = int(
            config[
                "timing_warmup_batches"
            ]
        )


        if args.num_workers is None:

            num_workers = int(
                config["num_workers"]
            )

        else:

            num_workers = int(
                args.num_workers
            )


        # =============================================================
        # Strong-scaling batch rule
        # =============================================================

        if (
            global_batch_size
            % world_size
            != 0
        ):

            raise RuntimeError(
                f"Global batch size "
                f"{global_batch_size} "
                f"is not divisible by "
                f"WORLD_SIZE={world_size}."
            )


        local_batch_size = (
            global_batch_size
            // world_size
        )


        if local_batch_size < 1:

            raise RuntimeError(
                "Local batch size became < 1."
            )


        # =============================================================
        # Reproducibility
        # =============================================================

        # Use the SAME seed for model construction on all ranks.
        #
        # DDP will also synchronize model parameters, but identical
        # initial construction makes the experiment easier to reason
        # about and reproduce.
        set_seed(
            seed
        )


        if device.type == "cuda":

            # Fixed tensor shapes make cuDNN algorithm benchmarking
            # useful for this performance-oriented project.
            torch.backends.cudnn.benchmark = True

            torch.cuda.empty_cache()


        # =============================================================
        # Process information
        # =============================================================

        hostname = (
            socket.gethostname()
        )


        if device.type == "cuda":

            device_name = (
                torch.cuda.get_device_name(
                    local_rank
                )
            )

        else:

            device_name = "CPU"


        print(
            f"[rank {rank}] "
            f"host={hostname} "
            f"local_rank={local_rank} "
            f"device={device} "
            f"backend={backend}",
            flush=True,
        )


        # Make sure every rank has completed setup before continuing.
        dist.barrier()


        if rank == 0:

            print()
            print("=" * 92)
            print(
                "DistributedDataParallel Training"
            )
            print("=" * 92)

            print(
                f"Workload               : "
                f"{workload}"
            )

            print(
                f"Backend                : "
                f"{backend}"
            )

            print(
                f"Device type            : "
                f"{device.type}"
            )

            print(
                f"World size             : "
                f"{world_size}"
            )

            print(
                f"Global batch size      : "
                f"{global_batch_size}"
            )

            print(
                f"Local batch size       : "
                f"{local_batch_size}"
            )

            print(
                f"Epochs                 : "
                f"{num_epochs}"
            )

            print(
                f"DataLoader workers/rank: "
                f"{num_workers}"
            )


        # =============================================================
        # Build workload
        # =============================================================

        (
            train_dataset,
            val_dataset,
            test_dataset,
            model,
            criterion,
        ) = build_workload(
            workload=workload,
            project_root=PROJECT_ROOT,
            config=config,
        )


        # =============================================================
        # SyncBatchNorm ablation
        # =============================================================
        #
        # This experiment is intentionally restricted to ResNet18.
        #
        # Baseline:
        #     ordinary BatchNorm
        #
        # Ablation:
        #     SyncBatchNorm
        #
        # SyncBatchNorm must be converted BEFORE wrapping with DDP.
        # Statistics are synchronized across the default process group.
        #
        # Everything else remains identical to the frozen DDP baseline.
        # =============================================================

        if workload != "resnet18":

            raise RuntimeError(
                "SyncBatchNorm ablation supports only resnet18."
            )


        batchnorm_before = sum(
            1
            for module in model.modules()
            if isinstance(
                module,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                ),
            )
        )


        model = (
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                model
            )
        )


        syncbatchnorm_after = sum(
            1
            for module in model.modules()
            if isinstance(
                module,
                torch.nn.SyncBatchNorm,
            )
        )


        if rank == 0:

            print()
            print(
                "SYNCBATCHNORM ABLATION"
            )

            print(
                "-" * 92
            )

            print(
                f"BatchNorm layers before : {batchnorm_before}"
            )

            print(
                f"SyncBatchNorm after     : {syncbatchnorm_after}"
            )

            print(
                "Statistics group        : full DDP world"
            )

            print()


        if batchnorm_before <= 0:

            raise RuntimeError(
                "No BatchNorm layers were found in ResNet18."
            )


        if (
            syncbatchnorm_after
            != batchnorm_before
        ):

            raise RuntimeError(
                "BatchNorm -> SyncBatchNorm conversion count mismatch."
            )


        # Move model before wrapping with DDP.
        model = model.to(
            device
        )


        # =============================================================
        # DDP wrapper
        # =============================================================

        if device.type == "cuda":

            ddp_model = DDP(
                model,

                # This process exclusively controls GPU local_rank.
                device_ids=[
                    local_rank
                ],

                output_device=local_rank,
            )

        else:

            # CPU/Gloo DDP does not use device_ids.
            ddp_model = DDP(
                model
            )


        # Optimizer receives the DDP model parameters.
        optimizer = build_optimizer(
            model=ddp_model,
            config=config,
        )


        parameter_count = sum(
            parameter.numel()
            for parameter
            in ddp_model.parameters()
        )


        trainable_parameter_count = sum(
            parameter.numel()
            for parameter
            in ddp_model.parameters()
            if parameter.requires_grad
        )


        if rank == 0:

            print()
            print("DATASET")
            print("-" * 92)

            print(
                f"Training samples       : "
                f"{len(train_dataset):,}"
            )

            print(
                f"Validation samples     : "
                f"{len(val_dataset):,}"
            )

            print(
                f"Test samples           : "
                f"{len(test_dataset):,}"
            )


            print()
            print("MODEL")
            print("-" * 92)

            print(
                f"Parameters             : "
                f"{parameter_count:,}"
            )

            print(
                f"Trainable parameters   : "
                f"{trainable_parameter_count:,}"
            )


        # =============================================================
        # Distributed samplers
        # =============================================================

        train_sampler = DistributedSampler(
            train_dataset,

            num_replicas=world_size,

            rank=rank,

            shuffle=True,

            seed=seed,

            drop_last=False,
        )


        val_sampler = DistributedSampler(
            val_dataset,

            num_replicas=world_size,

            rank=rank,

            shuffle=False,

            drop_last=False,
        )


        # =============================================================
        # DataLoaders
        # =============================================================

        # Give each rank its own DataLoader RNG stream.
        loader_generator = (
            torch.Generator()
        )

        loader_generator.manual_seed(
            seed + rank
        )


        use_pinned_memory = (
            device.type == "cuda"
        )


        train_loader = DataLoader(
            train_dataset,

            batch_size=local_batch_size,

            sampler=train_sampler,

            # sampler controls ordering.
            shuffle=False,

            num_workers=num_workers,

            pin_memory=use_pinned_memory,

            persistent_workers=(
                num_workers > 0
            ),

            generator=loader_generator,
        )


        val_loader = DataLoader(
            val_dataset,

            batch_size=local_batch_size,

            sampler=val_sampler,

            shuffle=False,

            num_workers=num_workers,

            pin_memory=use_pinned_memory,

            persistent_workers=(
                num_workers > 0
            ),
        )


        # =============================================================
        # Mixed precision
        # =============================================================

        requested_amp = bool(
            config["amp"]
        )


        # Precision is controlled by the config:
        #
        # Laptop RTX 4050:
        #     amp_dtype: bf16
        #
        # ROSI Tesla V100:
        #     amp_dtype: fp16
        #
        # V100 has native FP16 Tensor Core support but does not provide
        # the native BF16 path used by newer NVIDIA architectures.
        amp_dtype_name = str(
            config.get(
                "amp_dtype",
                "bf16",
            )
        ).lower()


        if amp_dtype_name == "bf16":

            amp_dtype = torch.bfloat16

        elif amp_dtype_name == "fp16":

            amp_dtype = torch.float16

        else:

            raise ValueError(
                f"Unsupported amp_dtype: "
                f"{amp_dtype_name}. "
                "Use 'bf16' or 'fp16'."
            )


        use_amp = (
            requested_amp
            and device.type == "cuda"
        )


        # Fail explicitly rather than silently running an unsupported
        # BF16 benchmark.
        if (
            use_amp
            and amp_dtype == torch.bfloat16
            and not torch.cuda.is_bf16_supported()
        ):

            raise RuntimeError(
                "BF16 AMP was requested, but this GPU "
                "does not support BF16. "
                "Use amp_dtype: fp16 for the V100."
            )


        # FP16 has a narrower numerical range than BF16.
        #
        # GradScaler multiplies the loss by a scale factor before
        # backward(), reducing the chance that small gradients
        # underflow to zero.
        #
        # BF16 normally does not require gradient scaling.
        use_grad_scaler = (
            use_amp
            and amp_dtype == torch.float16
        )


        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_grad_scaler,
        )


        def autocast_context():

            if device.type == "cuda":

                return torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=use_amp,
                )


            return nullcontext()


        if rank == 0:

            print(
                f"Requested AMP          : "
                f"{requested_amp}"
            )

            print(
                f"Effective AMP          : "
                f"{use_amp}"
            )

            print(
                f"AMP dtype              : "
                f"{amp_dtype_name if use_amp else 'fp32'}"
            )

            print(
                f"Gradient scaler        : "
                f"{use_grad_scaler}"
            )

            print()
            print("TRAINING")
            print("-" * 92)


        # =============================================================
        # Validation
        # =============================================================

        @torch.no_grad()
        def validate(
            max_batches=None,
        ):

            ddp_model.eval()


            # ---------------------------------------------------------
            # ResNet classification
            # ---------------------------------------------------------

            if workload == "resnet18":

                local_loss_sum = 0.0

                local_correct = 0

                local_examples = 0


                for batch_index, (
                    inputs,
                    targets,
                ) in enumerate(
                    val_loader
                ):

                    if (
                        max_batches
                        is not None
                        and
                        batch_index
                        >= max_batches
                    ):
                        break


                    inputs = inputs.to(
                        device,
                        non_blocking=(
                            device.type
                            == "cuda"
                        ),
                    )


                    targets = targets.to(
                        device,
                        non_blocking=(
                            device.type
                            == "cuda"
                        ),
                    )


                    with autocast_context():

                        logits = (
                            ddp_model(
                                inputs
                            )
                        )


                        loss = criterion(
                            logits,
                            targets,
                        )


                    batch_size = int(
                        inputs.size(0)
                    )


                    local_loss_sum += (
                        float(
                            loss.item()
                        )
                        * batch_size
                    )


                    predictions = (
                        torch.argmax(
                            logits,
                            dim=1,
                        )
                    )


                    local_correct += int(
                        (
                            predictions
                            == targets
                        )
                        .sum()
                        .item()
                    )


                    local_examples += (
                        batch_size
                    )


                (
                    global_loss_sum,
                    global_correct,
                    global_examples,
                ) = distributed_sum(
                    [
                        local_loss_sum,
                        local_correct,
                        local_examples,
                    ],
                    device=device,
                )


                if global_examples <= 0:

                    raise RuntimeError(
                        "No validation "
                        "examples processed."
                    )


                return {
                    "val_loss": (
                        global_loss_sum
                        / global_examples
                    ),

                    "val_accuracy": (
                        global_correct
                        / global_examples
                    ),

                    "val_examples": int(
                        global_examples
                    ),
                }


            # ---------------------------------------------------------
            # U-Net regression
            # ---------------------------------------------------------

            if workload == "unet_heat":

                local_squared_error = 0.0

                local_elements = 0

                local_examples = 0


                for batch_index, (
                    inputs,
                    targets,
                ) in enumerate(
                    val_loader
                ):

                    if (
                        max_batches
                        is not None
                        and
                        batch_index
                        >= max_batches
                    ):
                        break


                    inputs = inputs.to(
                        device,
                        non_blocking=(
                            device.type
                            == "cuda"
                        ),
                    )


                    targets = targets.to(
                        device,
                        non_blocking=(
                            device.type
                            == "cuda"
                        ),
                    )


                    with autocast_context():

                        predictions = (
                            ddp_model(
                                inputs
                            )
                        )


                    # Compute the reported validation error in FP32.
                    squared_error = (
                        (
                            predictions.float()
                            - targets
                        )
                        ** 2
                    )


                    local_squared_error += float(
                        squared_error
                        .sum()
                        .item()
                    )


                    local_elements += int(
                        targets.numel()
                    )


                    local_examples += int(
                        inputs.size(0)
                    )


                (
                    global_squared_error,
                    global_elements,
                    global_examples,
                ) = distributed_sum(
                    [
                        local_squared_error,
                        local_elements,
                        local_examples,
                    ],
                    device=device,
                )


                if global_elements <= 0:

                    raise RuntimeError(
                        "No validation "
                        "elements processed."
                    )


                return {
                    "val_mse": (
                        global_squared_error
                        / global_elements
                    ),

                    "val_examples": int(
                        global_examples
                    ),
                }


            raise RuntimeError(
                f"Unsupported workload: "
                f"{workload}"
            )


        # =============================================================
        # Training loop
        # =============================================================

        history = []


        for epoch_index in range(
            num_epochs
        ):

            epoch = (
                epoch_index + 1
            )


            # ---------------------------------------------------------
            # Critical DistributedSampler step
            #
            # This changes the shuffle deterministically each epoch
            # while keeping all ranks consistent about the global
            # permutation.
            # ---------------------------------------------------------

            train_sampler.set_epoch(
                epoch_index
            )


            ddp_model.train()


            if device.type == "cuda":

                torch.cuda.reset_peak_memory_stats(
                    device
                )


            local_loss_sum = 0.0

            local_examples = 0


            local_measured_examples = 0

            local_measured_time = 0.0


            # ---------------------------------------------------------
            # Align epoch start across ranks.
            # ---------------------------------------------------------

            dist.barrier()


            synchronize_device(
                device
            )


            epoch_wall_start = (
                time.perf_counter()
            )


            # ---------------------------------------------------------
            # Local training loop
            # ---------------------------------------------------------

            for batch_index, (
                inputs,
                targets,
            ) in enumerate(
                train_loader
            ):

                if (
                    args.max_train_batches
                    is not None
                    and
                    batch_index
                    >= args.max_train_batches
                ):
                    break


                # -----------------------------------------------------
                # CPU -> GPU transfer
                # -----------------------------------------------------

                inputs = inputs.to(
                    device,
                    non_blocking=(
                        device.type
                        == "cuda"
                    ),
                )


                targets = targets.to(
                    device,
                    non_blocking=(
                        device.type
                        == "cuda"
                    ),
                )


                # -----------------------------------------------------
                # Timing
                #
                # Transfer happens BEFORE the timer.
                #
                # Therefore primary throughput measures:
                #
                #     forward
                #     loss
                #     backward
                #     DDP gradient communication
                #     optimizer update
                #
                # rather than storage/DataLoader performance.
                # -----------------------------------------------------

                should_measure = (
                    batch_index
                    >= warmup_batches
                )


                if should_measure:

                    synchronize_device(
                        device
                    )


                    batch_start = (
                        time.perf_counter()
                    )


                # -----------------------------------------------------
                # Gradient reset
                # -----------------------------------------------------

                optimizer.zero_grad(
                    set_to_none=True
                )


                # -----------------------------------------------------
                # Forward + loss
                # -----------------------------------------------------

                with autocast_context():

                    outputs = ddp_model(
                        inputs
                    )


                    loss = criterion(
                        outputs,
                        targets,
                    )


                if not torch.isfinite(
                    loss
                ):

                    raise RuntimeError(
                        f"Non-finite loss "
                        f"on rank {rank}."
                    )


                # -----------------------------------------------------
                # Backward + optimizer update
                #
                # FP16:
                #     GradScaler protects small gradients from
                #     numerical underflow.
                #
                # BF16 / FP32:
                #     ordinary backward() + optimizer.step().
                #
                # DDP gradient synchronization still happens during
                # backward() in either case.
                # -----------------------------------------------------

                if use_grad_scaler:

                    scaler.scale(
                        loss
                    ).backward()


                    scaler.step(
                        optimizer
                    )


                    scaler.update()


                else:

                    loss.backward()

                    optimizer.step()


                batch_size = int(
                    inputs.size(0)
                )


                local_loss_sum += (
                    float(
                        loss.item()
                    )
                    * batch_size
                )


                local_examples += (
                    batch_size
                )


                # -----------------------------------------------------
                # Finish batch timing
                # -----------------------------------------------------

                if should_measure:

                    synchronize_device(
                        device
                    )


                    batch_elapsed = (
                        time.perf_counter()
                        - batch_start
                    )


                    local_measured_time += (
                        batch_elapsed
                    )


                    local_measured_examples += (
                        batch_size
                    )


            # =========================================================
            # End of local epoch
            # =========================================================

            synchronize_device(
                device
            )


            local_epoch_wall_time = (
                time.perf_counter()
                - epoch_wall_start
            )


            if local_examples == 0:

                raise RuntimeError(
                    f"Rank {rank} processed "
                    "zero training examples."
                )


            # =========================================================
            # Global training loss
            # =========================================================

            (
                global_loss_sum,
                global_examples,
            ) = distributed_sum(
                [
                    local_loss_sum,
                    local_examples,
                ],
                device=device,
            )


            mean_train_loss = (
                global_loss_sum
                / global_examples
            )


            # =========================================================
            # Global throughput
            # =========================================================

            # Total samples processed across ALL ranks.
            (
                global_measured_examples,
            ) = distributed_sum(
                [
                    local_measured_examples
                ],
                device=device,
            )


            # Distributed execution is limited by the slowest rank.
            #
            # Therefore:
            #
            #     global throughput
            #
            #         =
            #
            #     total global samples
            #     --------------------
            #     slowest rank time
            #
            max_measured_time = (
                distributed_max(
                    local_measured_time,
                    device=device,
                )
            )


            max_epoch_wall_time = (
                distributed_max(
                    local_epoch_wall_time,
                    device=device,
                )
            )


            if max_measured_time > 0:

                throughput = (
                    global_measured_examples
                    / max_measured_time
                )

            else:

                throughput = float(
                    "nan"
                )


            # =========================================================
            # Training GPU memory
            # =========================================================

            if device.type == "cuda":

                local_peak_vram_gb = (
                    torch.cuda
                    .max_memory_allocated(
                        device
                    )
                    / 1024**3
                )


                peak_vram_gb = (
                    distributed_max(
                        local_peak_vram_gb,
                        device=device,
                    )
                )

            else:

                peak_vram_gb = None


            # =========================================================
            # Global validation
            # =========================================================

            validation = validate(
                max_batches=(
                    args.max_val_batches
                )
            )


            # =========================================================
            # Epoch record
            # =========================================================

            result = {
                "epoch": int(
                    epoch
                ),

                "train_loss": float(
                    mean_train_loss
                ),

                "global_train_examples": int(
                    global_examples
                ),

                "global_measured_examples": int(
                    global_measured_examples
                ),

                "measured_train_time_sec": float(
                    max_measured_time
                ),

                "epoch_wall_time_sec": float(
                    max_epoch_wall_time
                ),

                "throughput_samples_per_sec": float(
                    throughput
                ),

                "peak_vram_per_rank_gb": (
                    None
                    if peak_vram_gb is None
                    else float(
                        peak_vram_gb
                    )
                ),

                **validation,
            }


            history.append(
                result
            )


            # =========================================================
            # Rank-zero logging
            # =========================================================

            if rank == 0:

                if workload == "resnet18":

                    print(
                        f"epoch {epoch:2d} | "
                        f"train loss "
                        f"{mean_train_loss:.4f} | "
                        f"val loss "
                        f"{validation['val_loss']:.4f} | "
                        f"val acc "
                        f"{100.0 * validation['val_accuracy']:.2f}% | "
                        f"throughput "
                        f"{throughput:.1f} img/s | "
                        f"measured "
                        f"{max_measured_time:.2f}s | "
                        f"epoch wall "
                        f"{max_epoch_wall_time:.2f}s",
                        end="",
                    )


                    if (
                        peak_vram_gb
                        is not None
                    ):

                        print(
                            f" | peak VRAM/rank "
                            f"{peak_vram_gb:.2f} GiB"
                        )

                    else:

                        print()


                elif workload == "unet_heat":

                    print(
                        f"epoch {epoch:2d} | "
                        f"train MSE "
                        f"{mean_train_loss:.6f} | "
                        f"val MSE "
                        f"{validation['val_mse']:.6f} | "
                        f"throughput "
                        f"{throughput:.1f} fields/s | "
                        f"measured "
                        f"{max_measured_time:.2f}s | "
                        f"epoch wall "
                        f"{max_epoch_wall_time:.2f}s",
                        end="",
                    )


                    if (
                        peak_vram_gb
                        is not None
                    ):

                        print(
                            f" | peak VRAM/rank "
                            f"{peak_vram_gb:.2f} GiB"
                        )

                    else:

                        print()


        # =============================================================
        # Training complete
        # =============================================================

        dist.barrier()


        # =============================================================
        # Steady-state summary
        # =============================================================

        # Epoch 1 often contains:
        #
        #     CUDA initialization
        #     cuDNN algorithm selection
        #     DataLoader startup
        #     cold caches
        #
        # We therefore use epochs 2...N for the primary steady-state
        # summary whenever multiple epochs are available.

        if len(history) > 1:

            steady_state_history = (
                history[1:]
            )

        else:

            steady_state_history = (
                history
            )


        finite_throughputs = [
            entry[
                "throughput_samples_per_sec"
            ]
            for entry
            in steady_state_history
            if math.isfinite(
                entry[
                    "throughput_samples_per_sec"
                ]
            )
        ]


        if finite_throughputs:

            steady_state_throughput = (
                sum(
                    finite_throughputs
                )
                / len(
                    finite_throughputs
                )
            )

        else:

            steady_state_throughput = (
                float("nan")
            )


        steady_state_wall_times = [
            entry[
                "epoch_wall_time_sec"
            ]
            for entry
            in steady_state_history
        ]


        mean_steady_state_wall_time = (
            sum(
                steady_state_wall_times
            )
            / len(
                steady_state_wall_times
            )
        )


        # =============================================================
        # Output naming
        # =============================================================

        is_smoke_test = (
            args.max_train_batches
            is not None
            or
            args.max_val_batches
            is not None
        )


        run_kind = (
            "smoke"
            if is_smoke_test
            else "full"
        )


        output_stem = (
            f"{workload}"
            f"_ddp"
            f"_{device.type}"
            f"_{backend}"
            f"_world{world_size}"
            f"_{run_kind}"
        )


        # =============================================================
        # Rank-0 checkpoint + results
        # =============================================================

        if rank == 0:

            # ---------------------------------------------------------
            # Checkpoint
            # ---------------------------------------------------------

            checkpoint_dir = (
                PROJECT_ROOT
                / "results"
                / "rosi_v100"
                / "syncbn"
                / "checkpoints"
            )


            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            checkpoint_path = (
                checkpoint_dir
                / f"{output_stem}.pt"
            )


            # IMPORTANT:
            #
            # ddp_model is a wrapper.
            #
            # The underlying normal PyTorch model is:
            #
            #     ddp_model.module
            #
            # Saving module.state_dict() means the resulting checkpoint
            # can later be loaded into a normal non-DDP model too.

            torch.save(
                {
                    "workload":
                        workload,

                    "world_size":
                        world_size,

                    "backend":
                        backend,

                    "device_type":
                        device.type,

                    "model_state_dict":
                        ddp_model
                        .module
                        .state_dict(),

                    "optimizer_state_dict":
                        optimizer
                        .state_dict(),

                    "config":
                        config,

                    "history":
                        history,
                },
                checkpoint_path,
            )


            # ---------------------------------------------------------
            # JSON results
            # ---------------------------------------------------------

            result_dir = (
                PROJECT_ROOT
                / "results"
                / "rosi_v100"
                / "syncbn"
                / "raw"
            )


            result_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            result_path = (
                result_dir
                / f"{output_stem}.json"
            )


            summary = {
                "execution":
                    "ddp",

                "run_kind":
                    run_kind,

                "workload":
                    workload,

                "backend":
                    backend,

                "device_type":
                    device.type,

                "world_size":
                    world_size,

                "rank_zero_hostname":
                    hostname,

                "rank_zero_device":
                    device_name,

                "pytorch_version":
                    torch.__version__,

                "cuda_version":
                    torch.version.cuda,

                "global_batch_size":
                    global_batch_size,

                "local_batch_size":
                    local_batch_size,

                "epochs":
                    num_epochs,

                "requested_amp":
                    requested_amp,

                "effective_amp":
                    use_amp,

                "amp_dtype": (
                    amp_dtype_name
                    if use_amp
                    else "fp32"
                ),

                "grad_scaler_enabled":
                    use_grad_scaler,

                "parameter_count":
                    parameter_count,

                "trainable_parameter_count":
                    trainable_parameter_count,

                "num_workers_per_rank":
                    num_workers,

                "timing_warmup_batches":
                    warmup_batches,

                "max_train_batches":
                    args.max_train_batches,

                "max_val_batches":
                    args.max_val_batches,

                "steady_state_epoch_policy":
                    (
                        "epochs_2_to_N"
                        if len(history) > 1
                        else "all_epochs"
                    ),

                "steady_state_throughput_samples_per_sec":
                    float(
                        steady_state_throughput
                    ),

                "mean_steady_state_epoch_wall_time_sec":
                    float(
                        mean_steady_state_wall_time
                    ),

                "history":
                    history,
            }


            with open(
                result_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    summary,
                    file,
                    indent=2,
                )


            print()
            print("=" * 92)
            print(
                "DDP TRAINING COMPLETE"
            )
            print("=" * 92)

            print(
                f"World size             : "
                f"{world_size}"
            )

            print(
                f"Global batch size      : "
                f"{global_batch_size}"
            )

            print(
                f"Local batch size       : "
                f"{local_batch_size}"
            )

            print(
                f"Steady throughput      : "
                f"{steady_state_throughput:.2f} "
                f"samples/s"
            )

            print(
                f"Mean steady epoch time : "
                f"{mean_steady_state_wall_time:.3f} s"
            )

            print(
                f"Checkpoint             : "
                f"{checkpoint_path}"
            )

            print(
                f"Result JSON            : "
                f"{result_path}"
            )

            print("=" * 92)


        # Do not let non-zero ranks destroy the process group while
        # rank 0 is still writing its output.
        dist.barrier()


    finally:

        # =============================================================
        # Cleanup
        # =============================================================

        if dist.is_initialized():

            dist.destroy_process_group()


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":

    main()
"""
Env audit for DDP.

Used on local RTX 4050 and HPC nodes
"""

import json 
import os 
import platform 
import socket 
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (PROJECT_ROOT / "results" / "local")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = (OUTPUT_DIR / "environment.json")

def bytes_to_gb(num_bytes: int) -> float:
    """
    convert bytes to GiB. 
    Dividing by 1024^3 converts bytes -> GiB
    """

    return (num_bytes / (1024 ** 3))

environment = {
    "system": {
        "hostname" : socket.gethostname(),
        "platform" : platform.platform(),
        "python_version" : platform.python_version(),
        "cpu_count" : os.cpu_count(),
    },

    "pytorch" : {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_build_version": torch.version.cuda,
        "cudnn_version": (torch.backends.cudnn.version())
    },

    "distributed" : {
        "distributed_available": dist.is_available(),
        "gloo_available": (
            dist.is_gloo_available()
            if dist.is_available() else False
        ),
        "nccl_available": (
            dist.is_nccl_available()
            if dist.is_available() else False
        ),
        "mpi_available": (
            dist.is_mpi_available()
            if dist.is_available() else False
        ),
    },
    "cuda" : {
        "cuda_available": torch.cuda.is_available(),
        "visible_gpu_count": torch.cuda.device_count(),
        "bf16_supported": (
            torch.cuda.is_bf16_supported()
            if torch.cuda.is_available() else False
        ),
        "devices":[],
    },
    # torchrun create these var automatically.
    "torchrun_environment": {
        "RANK" : os.environ.get("RANK"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
        "MASTER_PORT": os.environ.get("MASTER_PORT"),
    },

    # slurm sets separate collection of env var.
    # This will be populated on Rosi 
    "slurm_environment": {
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_NAME": os.environ.get("SLURM_JOB_NAME"),
        "SLURM_PROCID": os.environ.get("SLURM_PROCID"),
        "SLURM_LOCALID": os.environ.get("SLURM_LOCALID"),
        "SLURM_NTASKS": os.environ.get("SLURM_NTASKS"),
        "SLURM_JOB_NUM_NODES": os.environ.get("SLURM_JOB_NUM_NODES"),
    }
}


# cuda device audit 

if torch.cuda.is_available():
    for device_index in range(torch.cuda.device_count()):
        properties = (torch.cuda.get_device_properties(device_index))
        major, minor = (torch.cuda.get_device_capability(device_index))
        device_info = {
            "index": device_index,
            "name" : properties.name,
            "total_memory_gb": bytes_to_gb(properties.total_memory),
            "compute_capability": (f"{major}.{minor}"),
            "multiprocessor_count": (properties.multi_processor_count),
        }
        environment["cuda"]["devices"].append(device_info)

for device_info in environment["cuda"]["devices"]:
    assert isinstance(
        device_info,
        dict,
    ), (
        "CUDA device information must be "
        f"a dictionary, got {type(device_info)}"
    )
# output 

print("=" * 78)
print("PyTorch DDP Scaling Lab - Environment Audit")
print("=" * 78)


print("\nSYSTEM")
print("-" * 78)
print(
    f"Hostname              : "
    f"{environment['system']['hostname']}"
)

print(
    f"Platform              : "
    f"{environment['system']['platform']}"
)

print(
    f"Python                : "
    f"{environment['system']['python_version']}"
)

print(
    f"CPU logical cores     : "
    f"{environment['system']['cpu_count']}"
)


print("\nPYTORCH")
print("-" * 78)

print(
    f"PyTorch version       : "
    f"{environment['pytorch']['torch_version']}"
)

print(
    f"Torchvision version   : "
    f"{environment['pytorch']['torchvision_version']}"
)

print(
    f"PyTorch CUDA build    : "
    f"{environment['pytorch']['cuda_build_version']}"
)

print(
    f"cuDNN version         : "
    f"{environment['pytorch']['cudnn_version']}"
)


print("\nCUDA")
print("-" * 78)

print(
    f"CUDA available        : "
    f"{environment['cuda']['cuda_available']}"
)

print(
    f"Visible GPU count     : "
    f"{environment['cuda']['visible_gpu_count']}"
)

print(
    f"BF16 supported        : "
    f"{environment['cuda']['bf16_supported']}"
)

for device in environment["cuda"]["devices"]:
    print()
    print(
        f"GPU {device['index']}                 : "
        f"{device['name']}"
    )

    print(
        f"  VRAM                : "
        f"{device['total_memory_gb']:.2f} GiB"
    )

    print(
        f"  Compute capability  : "
        f"{device['compute_capability']}"
    )

    print(
        f"  SM count            : "
        f"{device['multiprocessor_count']}"
    )

    print("\nDISTRIBUTED BACKENDS")
print("-" * 78)

print(
    f"torch.distributed     : "
    f"{environment['distributed']['distributed_available']}"
)

print(
    f"Gloo available        : "
    f"{environment['distributed']['gloo_available']}"
)

print(
    f"NCCL available        : "
    f"{environment['distributed']['nccl_available']}"
)

print(
    f"MPI available         : "
    f"{environment['distributed']['mpi_available']}"
)


print("\nTORCHRUN ENVIRONMENT")
print("-" * 78)

for key, value in environment[
    "torchrun_environment"
].items():

    print(
        f"{key:<21}: {value}"
    )


print("\nSLURM ENVIRONMENT")
print("-" * 78)

for key, value in environment[
    "slurm_environment"
].items():

    print(
        f"{key:<21}: {value}"
    )

# sanity check 

print("\nSANITY CHECKS")
print("-" * 78)


if not dist.is_available():

    raise RuntimeError(
        "torch.distributed is not available "
        "in this PyTorch installation."
    )


print(
    "torch.distributed     : PASSED"
)


if not dist.is_gloo_available():

    raise RuntimeError(
        "Gloo is unavailable. "
        "We need Gloo for the local CPU "
        "distributed smoke test."
    )


print(
    "Gloo backend          : PASSED"
)


if torch.cuda.is_available():

    if not dist.is_nccl_available():

        raise RuntimeError(
            "CUDA is available but NCCL is not. "
            "We need NCCL later for multi-GPU "
            "training."
        )

    print(
        "NCCL backend          : PASSED"
    )

# ML env record 

with open(OUTPUT_PATH, "w", encoding="utf-8",) as file:
    json.dump(environment, file, indent=2,)

print()
print(
    f"Environment record    : "
    f"{OUTPUT_PATH}"
)

print()
print("=" * 78)
print("ENVIRONMENT AUDIT COMPLETE")
print("=" * 78)
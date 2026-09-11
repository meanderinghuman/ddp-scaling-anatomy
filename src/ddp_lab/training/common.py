"""
Shared training utilites.

This is important for scientific benchmarking:

    changing 1 GPU -> 4 GPUs

should change the DISTRIBUTED EXECUTION strategy, not silently change
the model, dataset, optimizer, or loss function.
"""

import random 
from pathlib import Path 
import numpy as np
import torch
import torch.nn as nn
import yaml

from ddp_lab.data.cifar100 import (
    build_cifar100_datasets,
)

from ddp_lab.data.heat_dataset import (
    HeatDataset,
)

from ddp_lab.models.resnet import (
    build_resnet18_cifar,
)

from ddp_lab.models.unet import (
    HeatUNet,
)
# config 
def load_config(config_path: Path,) -> dict:
    """
    load YAML experiment config.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config 

def set_seed(seed: int,) -> None:
    """
    Seed Python, Numpy and Pytorch RNGs
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# workload 

def build_workload(workload: str, project_root: Path, config: dict,):
    """
    construct datasets, model, and loss for one workload 
    Returns:
       train_dataset, val_dataset, test_dataset, model, criterion
    """
    if workload == "resnet18":
        train_dataset, val_dataset, test_dataset = (build_cifar100_datasets(
            data_dir=(
                project_root / "data" / "cifar100"
            ),
            seed = config["seed"],
        ))
        model = build_resnet18_cifar(
            num_classes=config["num_classes"]
        )

        # cross entropy for k class classification 
        # L = -log p(correct class)
        criterion = nn.CrossEntropyLoss()

    elif workload == "unet_heat":
        heat_data_dir = (project_root / "data" / "heat2d")
        train_dataset = HeatDataset(
            data_dir = heat_data_dir, split="train",
        )

        val_dataset = HeatDataset(
            data_dir = heat_data_dir, split="val",
        )

        test_dataset = HeatDataset(
            data_dir = heat_data_dir, split="test",
        )

        model = HeatUNet(
            in_channels=1, out_channels=1, base_channels=config["base_channels"],
        )

        # MSE
        criterion = nn.MSELoss()

    else:
        raise ValueError(f"unknown workload: {workload}")

    return (train_dataset, val_dataset, test_dataset, model, criterion,)

# optimizer 

def build_optimizer(model: nn.Module, config: dict,):
    
    optimizer_name = config["optimizer"].lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

    raise ValueError(f"unsupported optimizer: {optimizer_name}")







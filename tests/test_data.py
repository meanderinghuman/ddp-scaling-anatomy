"""
Basic dataset tests for both project workloads.
"""

from pathlib import Path

import torch

from ddp_lab.data.cifar100 import (
    build_cifar100_datasets,
)

from ddp_lab.data.heat_dataset import (
    HeatDataset,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


def test_heat_dataset():

    dataset = HeatDataset(
        data_dir=(
            PROJECT_ROOT
            / "data"
            / "heat2d"
        ),
        split="train",
    )


    assert len(dataset) == 4096


    x, y = dataset[0]


    assert x.shape == (
        1,
        64,
        64,
    )

    assert y.shape == (
        1,
        64,
        64,
    )


    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()


def test_cifar100_dataset():

    train_dataset, val_dataset, test_dataset = (
        build_cifar100_datasets(
            data_dir=(
                PROJECT_ROOT
                / "data"
                / "cifar100"
            ),
            seed=42,
        )
    )


    assert len(train_dataset) == 45_000
    assert len(val_dataset) == 5_000
    assert len(test_dataset) == 10_000


    image, label = train_dataset[0]


    assert image.shape == (
        3,
        32,
        32,
    )


    assert image.dtype == torch.float32


    assert isinstance(
        label,
        int,
    )


    assert 0 <= label < 100

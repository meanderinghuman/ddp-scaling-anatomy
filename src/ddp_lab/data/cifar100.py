"""
CIFAR-100 dataset utilities.

The official CIFAR-100 training set contains 50,000 images.

We deterministically split it into:

    45,000 training images
     5,000 validation images

The official 10,000-image test set remains separate.

Why deterministic splitting?
----------------------------
Later every DDP process must agree on which samples belong to the
training and validation sets. A fixed seed guarantees reproducibility.
"""

from pathlib import Path

import torch

from torch.utils.data import Subset

from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# CIFAR-100 normalization statistics
#
# These values are commonly used dataset-level channel statistics.
#
# Normalization performs:
#
#     x_normalized = (x - mean) / std
#
# separately for R, G, and B.
# ---------------------------------------------------------------------

CIFAR100_MEAN = (
    0.5071,
    0.4867,
    0.4408,
)

CIFAR100_STD = (
    0.2675,
    0.2565,
    0.2761,
)


def build_cifar100_datasets(
    data_dir: Path,
    seed: int = 42,
):
    """
    Build deterministic train, validation, and test datasets.

    Returns
    -------
    train_dataset
    val_dataset
    test_dataset
    """

    # ---------------------------------------------------------
    # Training augmentation
    #
    # RandomCrop:
    #   slightly shifts the image spatially.
    #
    # RandomHorizontalFlip:
    #   provides another transformed version of the image.
    #
    # These augmentations increase training diversity.
    # ---------------------------------------------------------

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(
                size=32,
                padding=4,
            ),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(
                CIFAR100_MEAN,
                CIFAR100_STD,
            ),
        ]
    )


    # ---------------------------------------------------------
    # Validation/test transforms must be deterministic.
    #
    # We do NOT randomly crop or flip validation examples,
    # otherwise the validation problem changes every epoch.
    # ---------------------------------------------------------

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                CIFAR100_MEAN,
                CIFAR100_STD,
            ),
        ]
    )


    # ---------------------------------------------------------
    # We instantiate the training set twice:
    #
    #   one copy using augmentation
    #   one copy using deterministic evaluation transforms
    #
    # Both copies refer to the same underlying CIFAR-100 images.
    # ---------------------------------------------------------

    full_train_augmented = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )


    full_train_eval = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )


    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )


    total_training_examples = len(
        full_train_augmented
    )

    assert total_training_examples == 50_000


    # ---------------------------------------------------------
    # Deterministic permutation.
    #
    # Every future run with seed=42 gets exactly the same split.
    # ---------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    indices = torch.randperm(
        total_training_examples,
        generator=generator,
    ).tolist()


    train_indices = indices[
        :45_000
    ]

    val_indices = indices[
        45_000:
    ]


    train_dataset = Subset(
        full_train_augmented,
        train_indices,
    )


    val_dataset = Subset(
        full_train_eval,
        val_indices,
    )


    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )

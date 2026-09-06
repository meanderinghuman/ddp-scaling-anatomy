"""
PyTorch Dataset for the generated 2D heat-equation fields.

Each item contains:

    input:
        initial temperature field

    target:
        final temperature field after numerical diffusion

Shape of each tensor:

    [1, 64, 64]
"""

from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset


class HeatDataset(Dataset):
    """
    Memory-mapped heat-equation dataset.

    The .npy arrays remain on disk and are read as samples are
    requested rather than loading every field into RAM.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
    ) -> None:

        super().__init__()


        valid_splits = {
            "train",
            "val",
            "test",
        }


        if split not in valid_splits:

            raise ValueError(
                f"split must be one of "
                f"{sorted(valid_splits)}, "
                f"got {split!r}."
            )


        self.inputs = np.load(
            data_dir
            / f"{split}_inputs.npy",

            mmap_mode="r",
        )


        self.targets = np.load(
            data_dir
            / f"{split}_targets.npy",

            mmap_mode="r",
        )


        if (
            len(self.inputs)
            != len(self.targets)
        ):

            raise RuntimeError(
                "Input and target lengths differ."
            )


    def __len__(self) -> int:

        return len(
            self.inputs
        )


    def __getitem__(
        self,
        index: int,
    ):

        # -----------------------------------------------------
        # Memory-mapped arrays are read-only.
        #
        # np.array(..., copy=True) gives PyTorch a writable,
        # contiguous float32 array.
        # -----------------------------------------------------

        input_array = np.array(
            self.inputs[index],
            dtype=np.float32,
            copy=True,
        )


        target_array = np.array(
            self.targets[index],
            dtype=np.float32,
            copy=True,
        )


        input_tensor = torch.from_numpy(
            input_array
        )


        target_tensor = torch.from_numpy(
            target_array
        )


        return (
            input_tensor,
            target_tensor,
        )

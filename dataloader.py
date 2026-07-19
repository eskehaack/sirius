from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import lightning.pytorch as pl

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class PairedNpyDataset(Dataset):
    """
    Expects:

        root/
            data/
                target_00000.npy
                target_00001.npy
                condition_00000.npy
                condition_00001.npy
    """

    def __init__(
        self,
        root: str | Path = Path("./.data/"),
        image_size: int = 256,
        data_dir: str = "preprocessed",
    ):
        self.root = Path(root)
        self.data_root = self.root / data_dir

        self.target_paths = sorted(self.data_root.glob("target_*.npy"))
        self.condition_paths = sorted(self.data_root.glob("condition_*.npy"))

        if not self.target_paths:
            raise RuntimeError(f"No target images found in {self.data_root}")

        if not self.condition_paths:
            raise RuntimeError(f"No condition images found in {self.data_root}")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.target_paths)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = np.load(path)
        img = np.moveaxis(img, 0, -1)  # Move channels to last dimension
        return self.transform(img)

    def __getitem__(self, idx: int):
        target = self._load_image(self.target_paths[idx])
        condition = self._load_image(self.condition_paths[idx])

        return {
            "target": target,
            "condition": condition,
        }


class PairedNpyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        dataset = PairedNpyDataset(
            root=self.root,
            image_size=self.image_size,
        )

        val_size = int(len(dataset) * self.val_fraction)
        train_size = len(dataset) - val_size

        self.train_ds, self.val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

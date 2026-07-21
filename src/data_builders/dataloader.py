from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning.pytorch as pl

from torch.utils.data import DataLoader

from src.data_builders.smhi_dataloader import TrainingDataset, ClimateDataBuilder, parse_config


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        
        config_path = '/work3/s214643/sirius/src/configs/training_config.toml'
        self.data_builder = ClimateDataBuilder(
            date_config = parse_config(config_path=config_path, config_keyword="dates"),
            preprocessing_config = parse_config(config_path=config_path, config_keyword="preprocessing"),
            predictor_config = parse_config(config_path=config_path, config_keyword="predictors"),
            static_features_config = parse_config(config_path=config_path, config_keyword="static_features"),
            target_config = parse_config(config_path=config_path, config_keyword="targets"),
        )
        self.data_builder.build_training_set()

        self.train_ds = TrainingDataset(
            predictors_path="/work3/s214643/sirius/data/ec_earth_zarr/predictors.zarr",
            targets_path="/work3/s214643/sirius/data/ec_earth_zarr/targets.zarr",
            static_features_path="/work3/s214643/sirius/data/ec_earth_zarr/static.zarr",
            indices=self.data_builder.train_idx,
        )

        self.val_ds = TrainingDataset(
            predictors_path="/work3/s214643/sirius/data/ec_earth_zarr/predictors.zarr",
            targets_path="/work3/s214643/sirius/data/ec_earth_zarr/targets.zarr",
            static_features_path="/work3/s214643/sirius/data/ec_earth_zarr/static.zarr",
            indices=self.data_builder.val_idx,
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

import argparse
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.data_builders.dataloader import DataModule
from src.data_builders.smhi_dataloader import parse_config
from src.model import LitConditionalDDPM


def train_ddpm(run_id: str, config_path: str = "./src/configs/training_config.toml"):

    config: dict = parse_config("training", config_path)

    batch_size = int(config["batch_size"])
    max_epochs = int(config["max_epochs"])
    lr = float(config["lr"])
    base_channels = int(config["base_channels"])
    num_workers = int(config["num_workers"])
    target_channels = int(config["target_channels"])
    condition_channels = int(config["condition_channels"])
    checkpoint_dir = Path(config["checkpoint_dir"]) / str(run_id)
    timesteps = int(config["timesteps"])
    channel_mults = tuple(config["channel_mults"])

    datamodule = DataModule(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = LitConditionalDDPM(
        target_channels=target_channels,
        condition_channels=condition_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        timesteps=timesteps,
        lr=lr,
    )

    model.save_hyperparameters(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="ddpm-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=TensorBoardLogger("logs", name="conditional_ddpm"),
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)

    return model, trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM model.")

    parser.add_argument(
        "--config_path",
        type=str,
        default="./src/configs/training_config.toml",
        help="Path to the training configuration file.",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="ID for the current training run.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_ddpm(run_id=args.run_id, config_path=args.config_path)

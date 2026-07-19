import argparse
import time

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from dataloader import PairedNpyDataModule
from model import LitConditionalDDPM

def train_ddpm(
    data_root: str = "./processed_carra2",
    image_size: int = 256,
    batch_size: int = 8,
    max_epochs: int = 1,
    lr: float = 2e-4,
    base_channels: int = 64,
    timesteps: int = 100,
    num_workers: int = 4,
    checkpoint_dir: str = "./checkpoints",
    target_channels: int = 3,
    condition_channels: int = 3,
):

    datamodule = PairedNpyDataModule(
        root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = LitConditionalDDPM(
        target_channels=target_channels,
        condition_channels=condition_channels,
        image_size=image_size,
        base_channels=base_channels,
        channel_mults=(1, 2, 4, 8),
        timesteps=timesteps,
        lr=lr,
    )

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
    parser = argparse.ArgumentParser(
        description="Train a DDPM model."
    )

    parser.add_argument("--data_root", type=str, default="./.data",
                        help="Root directory containing the training data.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Input image size.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size.")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate.")
    parser.add_argument("--base_channels", type=int, default=128,
                        help="Base number of channels in the U-Net.")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps.")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of DataLoader workers.")
    parser.add_argument("--target_channels", type=int, default=1,
                        help="Number of channels in the target images.")
    parser.add_argument("--condition_channels", type=int, default=1,
                        help="Number of channels in the conditioning images.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints. Defaults to ./checkpoints/<timestamp>.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else f"./checkpoints/{time.strftime('%Y%m%d%H%M%S')}"
    )

    train_ddpm(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        base_channels=args.base_channels,
        timesteps=args.timesteps,
        num_workers=args.num_workers,
        target_channels=args.target_channels,
        condition_channels=args.condition_channels,
        checkpoint_dir=checkpoint_dir,
    )
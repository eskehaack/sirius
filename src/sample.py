import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.model import LitConditionalDDPM
from src.data_builders.dataloader import DataModule


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

DATA_DIR = "./.data/preprocessed"


def main(
    run_id: str, checkpoint: str = "last", sample_dim: int = 0
):
    # ----------------------------------------------------
    # Load model
    # ----------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(f"./checkpoints/{run_id}/{checkpoint}.ckpt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}\nModel training did not complete as expected.")
    
    model = LitConditionalDDPM.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    out_dir = Path(f"./samples/{run_id}")
    out_dir = out_dir / time.strftime("%Y%m%d%H%M%S")
    out_dir.mkdir(exist_ok=False, parents=True)

    # ----------------------------------------------------
    # Load conditioning image
    # ----------------------------------------------------

    datamodule = DataModule(
        batch_size=1,
        num_workers=1,
    )
    datamodule.setup()
    stats = datamodule.data_builder.stats

    x, y, static, idx = next(iter(datamodule.val_dataloader()))

    org_condition_shape = x.shape[-2:]
    
    condition = F.interpolate(
        x,
        size=(model.image_size, model.image_size),
        mode="bilinear",
        align_corners=False,
    )

    static = F.interpolate(
        static,
        size=(model.image_size, model.image_size),
        mode="bilinear",
        align_corners=False,
    )

    condition = torch.cat([condition, static], dim=1)

    target = torch.stack(
        list(y.values()),
        dim=1,
    )

    org_target_shape = target.shape[-2:]

    target = F.interpolate(
        target,
        size=(model.image_size, model.image_size),
        mode="bilinear",
        align_corners=False,
    )

    # ----------------------------------------------------
    # IMPORTANT:
    # Use EXACTLY the same normalization as training
    # ----------------------------------------------------

    # mean = torch.stack([torch.tensor(stat.mean().values) for stat in stats.values()])
    # std = torch.stack([torch.tensor(stat.std().values) for stat in stats.values()]) + 1e-6

    # mean = F.pad(mean, (0, condition.shape[1] - mean.shape[0]), value=0.0).reshape(1, -1, 1, 1)
    # std = F.pad(std, (0, condition.shape[1] - std.shape[0]), value=1.0).reshape(1, -1, 1, 1)

    mean = condition.mean()
    std = condition.std() + 1e-6

    condition_norm = (condition - mean) / std
    condition_tensor = condition_norm.to(device)

    # ----------------------------------------------------
    # Sample
    # ----------------------------------------------------

    with torch.no_grad():
        prediction = model.sample(
            condition_tensor,
            num_steps=250,  # DDIM-like speedup
        )

    prediction = prediction.cpu()

    # Undo normalization
    prediction = prediction * std + mean

    # Scale down to original size
    condition = F.interpolate(
        condition,
        size=org_condition_shape,
        mode="bilinear",
        align_corners=False,
    )

    prediction = F.interpolate(
        prediction,
        size=org_target_shape,
        mode="bilinear",
        align_corners=False,
    )

    # ----------------------------------------------------
    # Save arrays
    # ----------------------------------------------------

    np.save(out_dir / "prediction.npy", prediction)

    # ----------------------------------------------------
    # Plot comparison
    # ----------------------------------------------------

    
    fig = plt.figure(figsize=(15, 5), layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])

    ax = [fig.add_subplot(grid[0, i]) for i in range(3)]
    cax = fig.add_subplot(grid[0, 3])

    condition_img = condition[0, sample_dim]
    prediction_img = prediction[0, sample_dim]
    target_img = target[0, sample_dim]

    vmin = min(img.min().item() for img in [condition_img, prediction_img, target_img])
    vmax = max(img.max().item() for img in [condition_img, prediction_img, target_img])

    for axis, title, image in zip(
        ax,
        ["Condition", "Generated", "Ground Truth"],
        [condition_img, prediction_img, target_img],
    ):
        mappable = axis.imshow(
            image.detach().cpu(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )
        axis.set_title(title)
        axis.axis("off")

    fig.colorbar(mappable, cax=cax, label="Temperature")

    plt.savefig(out_dir / "comparison.png", dpi=300)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a DDPM model.")

    parser.add_argument(
        "--run_id", type=str, help="Run ID defining directory for checkpoint file"
    )

    parser.add_argument(
        "--checkpoint", type=str, default="last", help="Name of checkpoint file"
    )

    parser.add_argument(
        "--sample_dim",
        type=int,
        default=0,
        help="What index of the sample image to show",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.run_id, args.checkpoint, args.sample_dim)

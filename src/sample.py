import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model import LitConditionalDDPM


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

DATA_DIR = "./.data/preprocessed"


def main(
    timestamp: str, checkpoint: str = "last", sample_index: int = 0, sample_dim: int = 0
):
    # ----------------------------------------------------
    # Load model
    # ----------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(f"./checkpoints/{timestamp}/{checkpoint}.ckpt")
    model = LitConditionalDDPM.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    out_dir = Path(f"./samples/{timestamp}")
    if out_dir.exists():
        out_dir = out_dir / time.strftime("%Y%m%d%H%M%S")
    out_dir.mkdir(exist_ok=False, parents=True)

    # ----------------------------------------------------
    # Load conditioning image
    # ----------------------------------------------------

    condition = np.load(Path(DATA_DIR) / f"condition_{sample_index:05d}.npy").astype(
        np.float32
    )

    target = np.load(Path(DATA_DIR) / f"target_{sample_index:05d}.npy").astype(
        np.float32
    )

    # ----------------------------------------------------
    # IMPORTANT:
    # Use EXACTLY the same normalization as training
    # ----------------------------------------------------

    mean = condition.mean()
    std = condition.std() + 1e-6

    condition_norm = (condition - mean) / std

    condition_tensor = torch.from_numpy(condition_norm).unsqueeze(0).to(device)

    # ----------------------------------------------------
    # Sample
    # ----------------------------------------------------

    with torch.no_grad():
        prediction = model.sample(
            condition_tensor,
            num_steps=250,  # DDIM-like speedup
        )

    prediction = prediction.squeeze().cpu().numpy()

    # Undo normalization
    prediction = prediction * std + mean

    # ----------------------------------------------------
    # Save arrays
    # ----------------------------------------------------

    np.save(out_dir / "prediction.npy", prediction)

    # ----------------------------------------------------
    # Plot comparison
    # ----------------------------------------------------

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(condition[sample_dim], cmap="coolwarm")
    ax[0].set_title("Condition")

    ax[1].imshow(prediction[sample_dim], cmap="coolwarm")
    ax[1].set_title("Generated")

    ax[2].imshow(target[sample_dim], cmap="coolwarm")
    ax[2].set_title("Ground Truth")

    for a in ax:
        a.axis("off")

    plt.savefig(out_dir / "comparison.png", dpi=300)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a DDPM model.")

    parser.add_argument(
        "--timestamp", type=str, help="Timestamp defining directory for checkpoint file"
    )

    parser.add_argument(
        "--checkpoint", type=str, default="last", help="Name of checkpoint file"
    )

    parser.add_argument(
        "--sample_index", type=int, default=0, help="Index in dataset to sample from"
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
    main(args.timestamp, args.checkpoint, args.sample_index, args.sample_dim)

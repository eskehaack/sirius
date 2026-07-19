import numpy as np
import torch
import torch.nn.functional as F


def mean_downscale_3d(arr: np.ndarray, factor: int) -> np.ndarray:
    """
    Downscale 3D array by mean aggregation.

    Input:  [D, H, W]
    Output: [D, H/factor, W/factor]
    """
    d, h, w = arr.shape

    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"Shape {arr.shape} is not divisible by factor={factor}")

    return arr.reshape(d, h // factor, factor, w // factor, factor).mean(axis=(2, 4))


def bicubic_upscale_3d(arr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Bicubic upscale 3D array using PyTorch.

    Input:  [D, H/factor, W/factor]
    Output: [D, H, W]
    """
    x = torch.from_numpy(arr).unsqueeze(0).float()  # [1, D, H/factor, W/factor]
    y = F.interpolate(
        x,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )

    return y[0].numpy()
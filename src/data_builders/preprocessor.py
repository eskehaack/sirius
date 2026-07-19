import numpy as np
import torch
import torch.nn.functional as F


def mean_downscale_3d(arr: np.ndarray, factor: int) -> np.ndarray:
    """
    Downscale 3D array by mean aggregation.

    Input:  [T, D, H, W]
    Output: [T, D, H/factor, W/factor]
    """
    t, d, h, w = arr.shape

    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"Shape {arr.shape} is not divisible by factor={factor}")

    return arr.reshape(t, d, h // factor, factor, w // factor, factor).mean(axis=(3, 5))


def bicubic_upscale_3d(arr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Bicubic upscale 3D array using PyTorch.

    Input:  [T, D, H/factor, W/factor]
    Output: [T, D, H, W]
    """
    x = torch.from_numpy(arr).float()  # [T, D, H/factor, W/factor]
    y = F.interpolate(
        x,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )

    return y.numpy()

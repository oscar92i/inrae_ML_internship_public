import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, List, Union, Optional, Tuple


# Time-series augmentations
def ts_augment_batch(x, jitter_std=0.01, min_crop=0.8, max_crop=1.0, prob_mask=0.0):
    """
    x: [B, C, T]
    Returns an augmented batch with:
      - jitter (Gaussian noise)
      - random time crop (pad back to T)
      - optional random temporal masking (set a small span to 0)
    """
    B, C, T = x.shape
    out = []

    for i in range(B):
        xi = x[i]

        # jitter
        if jitter_std > 0:
            xi = xi + jitter_std * torch.randn_like(xi)

        # random crop
        crop_ratio = np.random.uniform(min_crop, max_crop)
        crop_len = max(1, int(T * crop_ratio))
        start = np.random.randint(0, T - crop_len + 1)
        cropped = xi[:, start:start+crop_len]

        # pad back to original length (right-pad with zeros)
        if cropped.shape[1] < T:
            pad_len = T - cropped.shape[1]
            cropped = F.pad(cropped, (0, pad_len))

        # optional temporal masking of a short span
        if prob_mask > 0 and np.random.rand() < prob_mask:
            span = max(1, int(0.05 * T))
            mstart = np.random.randint(0, T - span + 1)
            cropped[:, mstart:mstart+span] = 0.0

        out.append(cropped)

    return torch.stack(out, dim=0)  # [B, C, T]


def jitter(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise to simulate sensor noise."""
    return x + std * torch.randn_like(x) if std > 0 else x


def random_crop(x: torch.Tensor, min_crop: float = 0.8, max_crop: float = 1.0) -> torch.Tensor:
    """Randomly crop temporal dimension and pad back to original size."""
    C, T = x.shape
    crop_ratio = np.random.uniform(min_crop, max_crop)
    crop_len = max(1, int(T * crop_ratio))
    start = np.random.randint(0, T - crop_len + 1)
    cropped = x[:, start:start+crop_len]

    # pad back to original length
    if cropped.shape[1] < T:
        pad_len = T - cropped.shape[1]
        cropped = F.pad(cropped, (0, pad_len))
    return cropped


def temporal_mask(x: torch.Tensor, prob: float = 0.0, span_ratio: float = 0.05) -> torch.Tensor:
    """Randomly zero out a short span of the time series."""
    C, T = x.shape
    if prob > 0 and np.random.rand() < prob:
        span = max(1, int(span_ratio * T))
        mstart = np.random.randint(0, T - span + 1)
        x[:, mstart:mstart+span] = 0.0
    return x


def resampling_augmentation(x: torch.Tensor, upsampling_factor: float = 2.0, subsequence_length_ratio: float = 0.5):
    """
    Resampling augmentation for SSL (two views).

    Args:
        x: Tensor [B, C, T]
        upsampling_factor: Factor to upsample (default 2.0)
        subsequence_length_ratio: Length of subsequence relative to upsampled length (default 0.5)

    Returns:
        (view1, view2): two augmented views of shape [B, C, T]
    """
    B, C, T = x.shape
    if upsampling_factor <= 1.0:
        raise ValueError("upsampling_factor must be > 1.0")
    if not 0.0 < subsequence_length_ratio < 1.0:
        raise ValueError("subsequence_length_ratio must be between 0 and 1")

    # Step 1: Upsample
    up_steps = int(T * upsampling_factor)
    x_up = F.interpolate(x, size=up_steps, mode="linear", align_corners=True)  # [B, C, up_steps]

    # Step 2: Subsequence length
    subseq_len = int(up_steps * subsequence_length_ratio)
    quarters = up_steps // 4
    pts_per_quarter = subseq_len // 4

    all_idx = torch.randperm(up_steps)
    idx1, idx2 = [], []
    for q in range(4):
        q_start, q_end = q * quarters, (q + 1) * quarters
        q_idx = all_idx[(all_idx >= q_start) & (all_idx < q_end)]
        idx1.extend(q_idx[:pts_per_quarter].tolist())
        idx2.extend(q_idx[pts_per_quarter:2*pts_per_quarter].tolist())

    idx1, idx2 = sorted(idx1), sorted(idx2)

    # Step 3: Extract subsequences
    subseq1 = x_up[:, :, idx1]  # [B, C, subseq_len]
    subseq2 = x_up[:, :, idx2]  # [B, C, subseq_len]

    # Step 4: Resample back to original length
    view1 = F.interpolate(subseq1, size=T, mode="linear", align_corners=True)
    view2 = F.interpolate(subseq2, size=T, mode="linear", align_corners=True)

    return view1, view2


class Augmentations:
    def __init__(self, augmentations: Optional[Union[str, List[str]]] = None, **kwargs):
        """
        Args:
            augmentations: single augmentation name or list of names
            kwargs: parameters for the augmentations
        """
        self.available = {
            "jitter": jitter,
            "random_crop": random_crop,
            "temporal_mask": temporal_mask,
            "resampling": resampling_augmentation,
        }
        if isinstance(augmentations, str):
            augmentations = [augmentations]
        self.augmentations = augmentations or []
        self.kwargs = kwargs

    def __call__(self, x: torch.Tensor):
        """
        Always return two augmented views (x1, x2).
        """
        if not self.augmentations:
            raise ValueError("no augmentations specified, pretrain cancelled")

        if "resampling" in self.augmentations:
            # Base: resampling gives two views
            x1, x2 = self.available["resampling"](x, **self.kwargs.get("resampling", {}))
            other_augs = [aug for aug in self.augmentations if aug != "resampling"]
        else:
            # Start with two independent clones
            x1, x2 = x.clone(), x.clone()
            other_augs = self.augmentations

        # Apply remaining augmentations independently to each view
        for aug in other_augs:
            if aug not in self.available:
                raise ValueError(f"Unknown augmentation: {aug}")
            fn = self.available[aug]
            x1 = torch.stack([fn(xi.clone(), **self.kwargs.get(aug, {})) for xi in x1])
            x2 = torch.stack([fn(xi.clone(), **self.kwargs.get(aug, {})) for xi in x2])

        return x1, x2
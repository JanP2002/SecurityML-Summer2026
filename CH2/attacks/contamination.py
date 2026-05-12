"""
attacks/contamination.py — Shared library of image contamination methods.
=========================================================================

All five contamination types from the task specification are implemented
here as both **raw tensor transforms** and **TensorDataset factories**.

Raw transforms
--------------
These functions accept a float32 image tensor of shape (N, C, H, W) with
values in [0.0, 1.0] and return a perturbed tensor of the same shape.
They are composable and can be chained or applied selectively.

Dataset factories
-----------------
``make_*_attack(images, **kwargs)`` calls the corresponding raw transform
and wraps the result in a ``TensorDataset`` with all labels set to 1
(anomaly class).  They are the primary interface used by step1.py and
step2.py.

Contamination methods
---------------------
1. Gaussian noise     — additive i.i.d. N(μ, σ²) pixel perturbation
2. Salt & pepper      — random black (0) and white (1) pixel replacement
3. Geometric distort  — smooth random displacement field via grid_sample
4. Blended attack     — linear mix with a pattern: (1−α)·x + α·p
5. Backdoor trigger   — small bright square inserted at fixed/random position
6. OOD replacement    — samples from a different distribution (no transform)

References
----------
- Gaussian / S&P:   standard additive noise models
- Geometric warp:   Jaderberg et al., "Spatial Transformer Networks" (2015)
- Blended attack:   Chen et al., "Targeted Backdoor Attacks" (2017)
- Backdoor trigger: Gu et al., "BadNets" (2017)
- OOD replacement:  Hendrycks & Gimpel, "A Baseline for OOD Detection" (2017)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _make_attack_dataset(attacked_images: torch.Tensor) -> TensorDataset:
    """Wrap perturbed images into a TensorDataset with all labels = 1 (attack).

    Parameters
    ----------
    attacked_images : torch.Tensor
        Float32 tensor of shape (N, C, H, W), values in [0.0, 1.0].

    Returns
    -------
    TensorDataset
        Pairs of (attacked_image, label=1).
    """
    labels = torch.ones(len(attacked_images), dtype=torch.long)
    return TensorDataset(attacked_images, labels)


# ===========================================================================
# 1. GAUSSIAN NOISE
# ===========================================================================

def gaussian_noise(
    images: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.4,
) -> torch.Tensor:
    """Add independent and identically distributed Gaussian noise to images.

    Each pixel :math:`x_{i,j}` is perturbed as:

    .. math::
        \\tilde{x}_{i,j} = \\text{clip}(x_{i,j} + \\varepsilon_{i,j},\\; 0,\\; 1)
        \\quad \\varepsilon_{i,j} \\sim \\mathcal{N}(\\mu,\\, \\sigma^2)

    Parameters
    ----------
    images : torch.Tensor
        Float32 image tensor of shape (N, C, H, W), values in [0.0, 1.0].
    mean : float
        Mean μ of the Gaussian distribution (default 0.0, zero-mean noise).
    std : float
        Standard deviation σ (default 0.4).  At σ=0.4 roughly 32 % of
        pixels would be clipped for a uniform input distribution.

    Returns
    -------
    torch.Tensor
        Noisy images, same shape as ``images``, clipped to [0.0, 1.0].
    """
    noise = torch.randn_like(images) * std + mean
    return torch.clamp(images + noise, 0.0, 1.0)


def make_gaussian_attack(
    images: torch.Tensor,
    std: float = 0.4,
    mean: float = 0.0,
) -> TensorDataset:
    """Create attack_a dataset: additive Gaussian noise (label = 1).

    Parameters
    ----------
    images : torch.Tensor
        Clean image tensor (N, C, H, W).
    std : float
        Noise standard deviation (default 0.4).
    mean : float
        Noise mean (default 0.0).

    Returns
    -------
    TensorDataset
        (noisy_image, 1) pairs.
    """
    print(f"  [contamination] Gaussian noise  (σ={std}, μ={mean})")
    return _make_attack_dataset(gaussian_noise(images, mean=mean, std=std))


# ===========================================================================
# 2. SALT & PEPPER NOISE
# ===========================================================================

def salt_and_pepper(
    images: torch.Tensor,
    prob: float = 0.10,
) -> torch.Tensor:
    """Apply salt-and-pepper (impulse) noise to images.

    With probability ``prob/2`` a pixel is set to 0 (pepper / black) and
    with probability ``prob/2`` it is set to 1 (salt / white), independently
    for every pixel and every image.

    .. math::
        \\tilde{x}_{i,j} =
        \\begin{cases}
            0   & u_{i,j} < p/2 \\\\
            1   & u_{i,j} > 1 - p/2 \\\\
            x_{i,j} & \\text{otherwise}
        \\end{cases}
        \\quad u_{i,j} \\sim U(0,1)

    Parameters
    ----------
    images : torch.Tensor
        Float32 tensor (N, C, H, W), values in [0.0, 1.0].
    prob : float
        Total corruption probability p ∈ (0, 1) split equally between
        salt and pepper (default 0.10, i.e. 10 % of pixels are corrupted).

    Returns
    -------
    torch.Tensor
        Corrupted images, same shape as ``images``.
    """
    corrupted = images.clone()
    noise_mask = torch.rand_like(images)
    corrupted[noise_mask < prob / 2.0] = 0.0   # pepper: forced black
    corrupted[noise_mask > 1.0 - prob / 2.0] = 1.0  # salt: forced white
    return corrupted


def make_salt_pepper_attack(
    images: torch.Tensor,
    prob: float = 0.10,
) -> TensorDataset:
    """Create attack dataset: salt-and-pepper noise (label = 1).

    Parameters
    ----------
    images : torch.Tensor
        Clean image tensor (N, C, H, W).
    prob : float
        Total pixel corruption probability (default 0.10).

    Returns
    -------
    TensorDataset
        (corrupted_image, 1) pairs.
    """
    print(f"  [contamination] Salt & pepper   (p={prob})")
    return _make_attack_dataset(salt_and_pepper(images, prob=prob))


# ===========================================================================
# 3. GEOMETRIC DISTORTION
# ===========================================================================

def _geo_distort_chunk(images: torch.Tensor, max_displacement: float) -> torch.Tensor:
    """Apply smooth random deformation to a single chunk of images.

    A low-resolution random displacement field is generated and then
    bilinearly upsampled to the full image resolution.  The resulting
    smooth vector field is added to the identity sampling grid and
    applied via ``F.grid_sample`` (reflection padding at borders).

    Parameters
    ----------
    images : torch.Tensor
        Chunk of float32 images, shape (B, C, H, W).
    max_displacement : float
        Maximum pixel displacement magnitude.  Normalised internally to
        the [-1, 1] coordinate system expected by ``grid_sample``.

    Returns
    -------
    torch.Tensor
        Geometrically distorted images, same shape as ``images``.
    """
    N, C, H, W = images.shape

    # Identity grid: shape (N, H, W, 2), values in [-1, 1]
    # theta is the identity affine transform for each sample in the batch.
    theta = (
        torch.eye(2, 3, dtype=torch.float32)
        .unsqueeze(0)
        .expand(N, -1, -1)
    )
    base_grid = F.affine_grid(theta, (N, C, H, W), align_corners=False)
    # base_grid[..., 0] = x-coords (width direction)
    # base_grid[..., 1] = y-coords (height direction)

    # Low-resolution random displacement seeds
    low_h = max(H // 8, 2)
    low_w = max(W // 8, 2)

    # Scale factor: max_displacement pixels → normalized [-1,1] coords
    # In the normalised grid, the full width maps to 2 units, so:
    #   1 pixel = 2 / W  normalised units
    dx_scale = max_displacement * 2.0 / W
    dy_scale = max_displacement * 2.0 / H

    raw_dx = torch.randn(N, 1, low_h, low_w, dtype=torch.float32) * dx_scale
    raw_dy = torch.randn(N, 1, low_h, low_w, dtype=torch.float32) * dy_scale

    # Upsample to full resolution via bilinear interpolation → smooth field
    dx = F.interpolate(raw_dx, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
    dy = F.interpolate(raw_dy, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)

    # Displacement tensor: (N, H, W, 2) in the same [x, y] order as base_grid
    displacement = torch.stack([dx, dy], dim=-1)

    # Clamp the final grid to valid range to avoid out-of-bounds artefacts
    grid = (base_grid + displacement).clamp(-1.0, 1.0)

    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )


def geometric_distortion(
    images: torch.Tensor,
    max_displacement: float = 5.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Apply smooth random geometric distortion to every image.

    Implements a dense deformation field warp:

    .. math::
        \\tilde{x}(p) = x(p + \\delta(p))

    where :math:`\\delta(p)` is a smooth (bilinearly upsampled from a
    low-resolution grid) random displacement field with magnitude bounded
    by ``max_displacement`` pixels.

    Parameters
    ----------
    images : torch.Tensor
        Float32 tensor (N, C, H, W), values in [0.0, 1.0].
    max_displacement : float
        Typical displacement magnitude in pixels (default 5.0).
        For 28×28 images this shifts pixels by ≤5 px, clearly visible
        but not enough to destroy digit identity.
    chunk_size : int
        Number of images processed per call to ``F.grid_sample`` to
        bound peak GPU/RAM usage (default 512).

    Returns
    -------
    torch.Tensor
        Geometrically distorted images, same shape as ``images``.
    """
    # Process in chunks to limit peak memory for large datasets
    chunks = [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]
    result_chunks = [_geo_distort_chunk(chunk, max_displacement) for chunk in chunks]
    return torch.cat(result_chunks, dim=0)


def make_geometric_attack(
    images: torch.Tensor,
    max_displacement: float = 5.0,
) -> TensorDataset:
    """Create attack dataset: smooth random geometric distortion (label = 1).

    Parameters
    ----------
    images : torch.Tensor
        Clean image tensor (N, C, H, W).
    max_displacement : float
        Max pixel displacement (default 5.0).

    Returns
    -------
    TensorDataset
        (distorted_image, 1) pairs.
    """
    print(f"  [contamination] Geometric warp  (max_disp={max_displacement}px)")
    return _make_attack_dataset(geometric_distortion(images, max_displacement=max_displacement))


# ===========================================================================
# 4. BLENDED ATTACK
# ===========================================================================

def blended_attack(
    images: torch.Tensor,
    alpha: float = 0.30,
    pattern: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend each image with a fixed pattern at mixing ratio α.

    .. math::
        \\tilde{x} = (1 - \\alpha)\\,x + \\alpha\\,p

    where :math:`p` is a pattern image (random uniform noise by default)
    and :math:`\\alpha` controls the blend strength.

    Parameters
    ----------
    images : torch.Tensor
        Float32 tensor (N, C, H, W), values in [0.0, 1.0].
    alpha : float
        Blending weight for the pattern.  α = 0.0 → original image,
        α = 1.0 → pure pattern.  Default 0.3 gives a clearly visible
        but recognisable corruption.
    pattern : torch.Tensor or None
        Pattern to blend in.  Can be:
        - ``None``: one random uniform noise pattern sampled once and
          broadcast across the entire batch (consistent contamination).
        - A single pattern of shape (C, H, W) or (1, C, H, W): broadcast.
        - A batch pattern of shape (N, C, H, W): per-image pattern.
        Values should be in [0.0, 1.0].

    Returns
    -------
    torch.Tensor
        Blended images, same shape as ``images``, clipped to [0.0, 1.0].
    """
    if pattern is None:
        # One consistent random noise pattern broadcast to all images.
        # Using a single pattern (rather than per-image noise) distinguishes
        # this attack from Gaussian noise — the trigger pattern is systematic.
        pattern = torch.rand(1, *images.shape[1:], dtype=images.dtype)

    # Ensure pattern broadcasts correctly to (N, C, H, W)
    if pattern.dim() == 3:
        pattern = pattern.unsqueeze(0)

    blended = (1.0 - alpha) * images + alpha * pattern.expand_as(images)
    return torch.clamp(blended, 0.0, 1.0)


def make_blended_attack(
    images: torch.Tensor,
    alpha: float = 0.30,
    pattern: torch.Tensor | None = None,
) -> TensorDataset:
    """Create attack dataset: blended attack (label = 1).

    A single random noise pattern is generated at call time and blended
    into all images at ratio α.  The same pattern is used for training
    and test sets only if the caller passes an explicit ``pattern``
    argument; otherwise separate random patterns are generated.  For
    reproducible experiments, pre-generate the pattern and pass it in.

    Parameters
    ----------
    images : torch.Tensor
        Clean image tensor (N, C, H, W).
    alpha : float
        Blend ratio (default 0.30).
    pattern : torch.Tensor or None
        Optional fixed pattern tensor. If None, a new random pattern is
        sampled at each call.

    Returns
    -------
    TensorDataset
        (blended_image, 1) pairs.
    """
    print(f"  [contamination] Blended attack  (α={alpha})")
    return _make_attack_dataset(blended_attack(images, alpha=alpha, pattern=pattern))


# ===========================================================================
# 5. BACKDOOR TRIGGER
# ===========================================================================

def backdoor_trigger(
    images: torch.Tensor,
    trigger_size: int = 5,
    trigger_value: float = 1.0,
    position: str = "bottom_right",
) -> torch.Tensor:
    """Insert a small bright square (trigger patch) into every image.

    In backdoor attack research the trigger is typically a small, fixed
    pattern placed at a consistent location so that the model learns to
    associate its presence with the attack class.

    Parameters
    ----------
    images : torch.Tensor
        Float32 tensor (N, C, H, W), values in [0.0, 1.0].
    trigger_size : int
        Side length of the square trigger patch in pixels (default 5).
    trigger_value : float
        Pixel intensity of the trigger (default 1.0 = white).
    position : str
        Where to place the trigger.  Options:
        ``'bottom_right'`` (default), ``'top_left'``, ``'top_right'``,
        ``'bottom_left'``, or ``'random'`` (single random location for
        the whole batch, vectorised).

    Returns
    -------
    torch.Tensor
        Images with trigger inserted, same shape as input.

    Notes
    -----
    The trigger is inserted uniformly in all images at the same position
    (one position per call) to keep the operation fully vectorised.
    ``'random'`` picks a single random location for the entire batch.
    """
    triggered = images.clone()
    H, W = images.shape[2], images.shape[3]

    # Determine top-left corner (y0, x0) of the trigger patch
    pad = 1  # keep at least 1 px from the border
    if position == "bottom_right":
        y0 = H - trigger_size - pad
        x0 = W - trigger_size - pad
    elif position == "top_left":
        y0 = pad
        x0 = pad
    elif position == "top_right":
        y0 = pad
        x0 = W - trigger_size - pad
    elif position == "bottom_left":
        y0 = H - trigger_size - pad
        x0 = pad
    else:  # 'random' — single location for entire batch (vectorised)
        y0 = int(torch.randint(pad, H - trigger_size - pad, (1,)).item())
        x0 = int(torch.randint(pad, W - trigger_size - pad, (1,)).item())

    # Vectorised assignment: all images simultaneously
    triggered[:, :, y0 : y0 + trigger_size, x0 : x0 + trigger_size] = trigger_value
    return triggered


def make_backdoor_attack(
    images: torch.Tensor,
    trigger_size: int = 5,
    trigger_value: float = 1.0,
    position: str = "bottom_right",
) -> TensorDataset:
    """Create attack dataset: backdoor trigger insertion (label = 1).

    Parameters
    ----------
    images : torch.Tensor
        Clean image tensor (N, C, H, W).
    trigger_size : int
        Trigger patch side length in pixels (default 5).
    trigger_value : float
        Trigger pixel intensity (default 1.0 = white square).
    position : str
        Trigger location (default ``'bottom_right'``).

    Returns
    -------
    TensorDataset
        (triggered_image, 1) pairs.
    """
    print(
        f"  [contamination] Backdoor trigger (size={trigger_size}px, "
        f"pos={position}, val={trigger_value})"
    )
    return _make_attack_dataset(
        backdoor_trigger(
            images,
            trigger_size=trigger_size,
            trigger_value=trigger_value,
            position=position,
        )
    )


# ===========================================================================
# 6. OOD REPLACEMENT (no image transform — different source distribution)
# ===========================================================================

def make_ood_attack(ood_images: torch.Tensor) -> TensorDataset:
    """Create attack dataset from Out-of-Distribution (OOD) samples.

    No pixel-level transform is applied.  The images are drawn from a
    structurally different distribution (e.g. Fashion-MNIST clothing items
    used as attacks against a MNIST-trained detector).  The entire semantic
    content is different, making this a strong generalisation test.

    Parameters
    ----------
    ood_images : torch.Tensor
        Images from the OOD source, shape (N, C, H, W), values in [0.0, 1.0].
        Must have the same spatial dimensions as the clean training images.

    Returns
    -------
    TensorDataset
        (ood_image, 1) pairs — all labelled as anomaly.
    """
    print(f"  [contamination] OOD replacement (N={len(ood_images)}, no transform)")
    labels = torch.ones(len(ood_images), dtype=torch.long)
    return TensorDataset(ood_images, labels)

"""
metrics/reconstruction.py
─────────────────────────
Evaluation metrics for split learning privacy research.

Three categories, matching the proposal's evaluation framework:

  Reconstruction Quality (attack success)
    - SSIM  : Structural Similarity Index         [0, 1],  higher = better attack
    - PSNR  : Peak Signal-to-Noise Ratio          [dB],    higher = better attack
    - LPIPS : Learned Perceptual Image Similarity [0, 1],  lower  = better attack (optional)

  Privacy Leakage
    - Distance Correlation (dCor) between raw input and smashed data
      dCor = 0 → statistically independent (perfect privacy)
      dCor = 1 → perfectly correlated     (total leakage)

  Utility Preservation
    - Test accuracy tracked separately by trainers; helper provided here for
      computing accuracy from tensors in evaluation scripts.

All functions operate on torch.Tensors and are differentiable where possible
so they can be used inside training loops as loss terms.

Usage:
    from metrics.reconstruction import compute_ssim, compute_psnr, distance_correlation

    ssim  = compute_ssim(reconstructed, originals)   # scalar tensor
    psnr  = compute_psnr(reconstructed, originals)   # scalar tensor
    dcor  = distance_correlation(smashed, inputs)    # scalar tensor (for NoPeekNN loss)
"""

import torch
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_01(x: torch.Tensor) -> torch.Tensor:
    """
    Rescale a batch of images so each image's pixel values lie in [0, 1].

    CIFAR-10 images are normalized with per-channel mean/std, producing values
    outside [0, 1]. Metrics like SSIM and PSNR are defined on [0, 1] images.
    We rescale per-image (min-max) rather than using fixed statistics so that
    the metrics reflect reconstruction fidelity independent of normalization.
    """
    b = x.shape[0]
    x_flat = x.view(b, -1)
    mn = x_flat.min(dim=1).values.view(b, 1, 1, 1)
    mx = x_flat.max(dim=1).values.view(b, 1, 1, 1)
    return (x - mn) / (mx - mn + 1e-8)


def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create a Gaussian kernel for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    return kernel_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    rescale: bool = True,
) -> torch.Tensor:
    """
    Compute mean SSIM over a batch of images.

    Args:
        pred:        Reconstructed images  (B, C, H, W)
        target:      Original images       (B, C, H, W)
        window_size: Size of Gaussian window (default 11, standard SSIM)
        sigma:       Gaussian sigma         (default 1.5)
        data_range:  Value range of images  (1.0 after rescaling)
        rescale:     If True, rescale both tensors to [0, 1] before computing.
                     Set False if images are already in [0, 1].

    Returns:
        Scalar tensor: mean SSIM across the batch. Range [0, 1].
    """
    if rescale:
        pred   = _to_01(pred)
        target = _to_01(target)

    C = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, C).to(pred.device)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    padding = window_size // 2

    mu_pred   = F.conv2d(pred,   kernel, padding=padding, groups=C)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=C)

    mu_pred_sq   = mu_pred   ** 2
    mu_target_sq = mu_target ** 2
    mu_cross     = mu_pred * mu_target

    sigma_pred_sq   = F.conv2d(pred   ** 2, kernel, padding=padding, groups=C) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, kernel, padding=padding, groups=C) - mu_target_sq
    sigma_cross     = F.conv2d(pred * target, kernel, padding=padding, groups=C) - mu_cross

    ssim_map = (
        (2 * mu_cross + C1) * (2 * sigma_cross + C2)
    ) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )

    return ssim_map.mean()


# ─────────────────────────────────────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    rescale: bool = True,
) -> torch.Tensor:
    """
    Compute mean PSNR over a batch of images.

    Args:
        pred:       Reconstructed images (B, C, H, W)
        target:     Original images      (B, C, H, W)
        data_range: Value range          (1.0 after rescaling)
        rescale:    Rescale to [0, 1] before computing

    Returns:
        Scalar tensor: mean PSNR in dB. Higher = better reconstruction.
        Returns inf if MSE = 0 (perfect reconstruction).
    """
    if rescale:
        pred   = _to_01(pred)
        target = _to_01(target)

    mse = F.mse_loss(pred, target, reduction='none').view(pred.shape[0], -1).mean(dim=1)
    psnr = 10 * torch.log10(data_range ** 2 / (mse + 1e-10))
    return psnr.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Distance Correlation (dCor)
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distance matrix for a batch of vectors.

    Args:
        x: Tensor of shape (N, D) — N samples, D features

    Returns:
        Distance matrix of shape (N, N)
    """
    x_flat = x.view(x.shape[0], -1).float()
    sq = (x_flat ** 2).sum(dim=1, keepdim=True)
    dist_sq = sq + sq.T - 2 * x_flat @ x_flat.T
    dist_sq = dist_sq.clamp(min=0)
    return torch.sqrt(dist_sq + 1e-10)


def _double_center(d: torch.Tensor) -> torch.Tensor:
    """
    Apply double centering to a distance matrix.

    Double centering subtracts row means, column means, and adds overall mean.
    This transforms the distance matrix into a covariance-like form needed
    for distance correlation.
    """
    row_mean = d.mean(dim=1, keepdim=True)
    col_mean = d.mean(dim=0, keepdim=True)
    grand_mean = d.mean()
    return d - row_mean - col_mean + grand_mean


def distance_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute distance correlation between two batches of tensors.

    Distance correlation (Székely et al., 2007) is a measure of statistical
    dependence that is zero if and only if the variables are independent.
    This makes it more powerful than linear correlation for detecting
    information leakage in smashed data.

    Used in NoPeekNN as the privacy regularization term:
        loss = task_loss + λ * distance_correlation(smashed_data, raw_input)

    Args:
        x: First batch  (N, ...) — typically raw input images
        y: Second batch (N, ...) — typically smashed data activations
           Both tensors must have the same batch dimension N.

    Returns:
        Scalar tensor in [0, 1]: distance correlation coefficient.
        0 = statistically independent (perfect privacy)
        1 = perfectly correlated (maximum leakage)

    Note:
        Computation is O(N²) in batch size. For large batches (N > 256),
        consider using a random subsample to keep this tractable.
    """
    n = x.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    # Pairwise distance matrices
    a = _pairwise_distances(x)
    b = _pairwise_distances(y)

    # Double center
    A = _double_center(a)
    B = _double_center(b)

    # Distance covariance and variance
    dcov_sq  = (A * B).sum() / (n * n)
    dvar_x   = (A * A).sum() / (n * n)
    dvar_y   = (B * B).sum() / (n * n)

    # Clamp to avoid sqrt of negative (numerical noise)
    dvar_product = (dvar_x * dvar_y).clamp(min=0)

    dcor = torch.sqrt(dcov_sq.clamp(min=0)) / (torch.sqrt(torch.sqrt(dvar_product)) + 1e-8)
    return dcor.clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: classification accuracy from tensors
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute top-1 accuracy from logits and ground-truth labels.

    Args:
        logits: Model output (B, num_classes)
        labels: Ground-truth class indices (B,)

    Returns:
        Float accuracy in [0, 1]
    """
    _, predicted = logits.max(1)
    return predicted.eq(labels).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: evaluate a full reconstruction result set
# ─────────────────────────────────────────────────────────────────────────────

def reconstruction_report(
    originals: torch.Tensor,
    reconstructed: torch.Tensor,
    smashed_data: torch.Tensor = None,
    label: str = "Attack",
) -> dict:
    """
    Compute and print a full reconstruction quality report.

    Args:
        originals:      Original input images (B, C, H, W)
        reconstructed:  Reconstructed images  (B, C, H, W)
        smashed_data:   Intermediate activations for dCor computation (optional)
        label:          Label for the printed report header

    Returns:
        Dict with keys: ssim, psnr, dcor (if smashed_data provided)
    """
    results = {}

    ssim = compute_ssim(reconstructed, originals).item()
    psnr = compute_psnr(reconstructed, originals).item()
    results["ssim"] = ssim
    results["psnr"] = psnr

    lines = [
        f"\n{'─' * 50}",
        f"  {label} — Reconstruction Report",
        f"{'─' * 50}",
        f"  SSIM : {ssim:.4f}  (1.0 = perfect, 0.0 = random)",
        f"  PSNR : {psnr:.2f} dB  (higher = better reconstruction)",
    ]

    if smashed_data is not None:
        with torch.no_grad():
            dcor = distance_correlation(
                originals.to(smashed_data.device),
                smashed_data
            ).item()
        results["dcor"] = dcor
        lines.append(f"  dCor : {dcor:.4f}  (0.0 = no leakage, 1.0 = full leakage)")

    lines.append(f"{'─' * 50}")
    print("\n".join(lines))

    return results
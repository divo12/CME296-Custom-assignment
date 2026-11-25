"""
Provided staff dataset utilities for low-resource (CPU / Apple MPS / single-GPU) experiments.

Do NOT modify this file. Every generator runs in pure PyTorch with zero network downloads
so you can iterate on Assignments 1-5 in seconds on a laptop.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor


def sample_2d_checkerboard(
    n_samples: int,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Sample points in [-2, 2]^2 on a 4x4 alternating checkerboard (Assignment 1 & 3)."""
    ix = torch.randint(0, 4, (n_samples,), generator=generator, device=device)
    iy_parity = torch.randint(0, 2, (n_samples,), generator=generator, device=device)
    iy = 2 * iy_parity + (ix % 2)
    u = torch.rand((n_samples, 2), generator=generator, device=device)
    x1 = (ix.float() + u[:, 0]) - 2.0
    x2 = (iy.float() + u[:, 1]) - 2.0
    return torch.stack([x1, x2], dim=-1)


def sample_2d_swiss_roll(
    n_samples: int,
    noise: float = 0.06,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Sample a 2D Swiss-roll spiral normalized to approximately unit variance."""
    u = torch.rand(n_samples, generator=generator, device=device)
    theta = 1.5 * math.pi * (1.0 + 2.0 * u)
    eps = torch.randn((n_samples, 2), generator=generator, device=device) * noise
    x = torch.stack([theta * torch.cos(theta), theta * torch.sin(theta)], dim=-1) / 7.0
    return x + eps


def sample_2d_two_moons(
    n_samples: int,
    noise: float = 0.06,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Sample two interleaving half-moons in R^2 (Assignment 3 Reflow benchmark)."""
    n_top = n_samples // 2
    n_bot = n_samples - n_top
    theta_top = torch.rand(n_top, generator=generator, device=device) * math.pi
    theta_bot = torch.rand(n_bot, generator=generator, device=device) * math.pi

    top = torch.stack([torch.cos(theta_top), torch.sin(theta_top)], dim=-1)
    bot = torch.stack([1.0 - torch.cos(theta_bot), 0.5 - torch.sin(theta_bot)], dim=-1)
    pts = torch.cat([top, bot], dim=0)
    pts = (pts - torch.tensor([0.5, 0.25], device=device)) * 1.4
    pts = pts + noise * torch.randn(pts.shape, generator=generator, device=device)
    perm = torch.randperm(n_samples, generator=generator, device=device)
    return pts[perm]


def sample_2d_asymmetric_gmm(
    n_samples: int,
    weight_mode1: float = 0.8,
    sigma: float = 0.25,
    separation: float = 5.0,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """
    Asymmetric 2-mode Gaussian mixture:
        p(x) = w * N([-d/2, -d/2], sigma^2 I) + (1 - w) * N([+d/2, +d/2], sigma^2 I)
    Used in Assignment 2 to test low-density score blindness vs. Annealed Langevin / SDEs.
    """
    u = torch.rand(n_samples, generator=generator, device=device)
    labels = (u >= weight_mode1).long()
    half = separation / 2.0
    centers = torch.tensor([[-half, -half], [half, half]], device=device)
    means = centers[labels]
    samples = means + sigma * torch.randn((n_samples, 2), generator=generator, device=device)
    return samples, labels


def sample_2d_eight_gaussians(
    n_samples: int,
    radius: float = 2.0,
    sigma: float = 0.15,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """8-mode symmetric ring of Gaussians with class labels 0..7."""
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=9, device=device)[:-1]
    centers = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    labels = torch.randint(0, 8, (n_samples,), generator=generator, device=device)
    samples = centers[labels] + sigma * torch.randn((n_samples, 2), generator=generator, device=device)
    return samples, labels


def sample_procedural_shapes(
    n_samples: int,
    image_size: int = 16,
    noise_std: float = 0.03,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """
    Zero-download 4-class grayscale image dataset in [-1, 1] of shape (N, 1, H, W)
    for fast laptop testing of VAEs, CFG, U-Net, DiT, and MM-DiT (Assignments 4 & 5).

    Classes:
      0: Circle / ring
      1: Axis-aligned square
      2: Plus / cross
      3: Diagonal X
    """
    labels = torch.randint(0, 4, (n_samples,), generator=generator, device=device)
    coords = torch.linspace(-1.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    yy = yy.unsqueeze(0)  # (1, H, W)
    xx = xx.unsqueeze(0)

    # Slight per-sample jitter in center and scale so each class forms a smooth manifold
    cx = (torch.rand((n_samples, 1, 1), generator=generator, device=device) - 0.5) * 0.35
    cy = (torch.rand((n_samples, 1, 1), generator=generator, device=device) - 0.5) * 0.35
    scale = 0.42 + 0.18 * torch.rand((n_samples, 1, 1), generator=generator, device=device)

    dx = xx - cx
    dy = yy - cy

    circle = (torch.sqrt(dx * dx + dy * dy) <= scale).float()
    square = ((dx.abs() <= scale * 0.85) & (dy.abs() <= scale * 0.85)).float()
    cross = (
        ((dx.abs() <= scale * 0.28) & (dy.abs() <= scale))
        | ((dy.abs() <= scale * 0.28) & (dx.abs() <= scale))
    ).float()
    diag_x = (
        (((dx - dy).abs() <= scale * 0.30) | ((dx + dy).abs() <= scale * 0.30))
        & (dx.abs() <= scale)
        & (dy.abs() <= scale)
    ).float()

    bank = torch.stack([circle, square, cross, diag_x], dim=1)  # (N, 4, H, W)
    idx = labels.view(n_samples, 1, 1, 1).expand(n_samples, 1, image_size, image_size)
    imgs = bank.gather(1, idx) * 2.0 - 1.0  # scale {0, 1} -> {-1, +1}
    if noise_std > 0:
        imgs = (imgs + noise_std * torch.randn(imgs.shape, generator=generator, device=device)).clamp(-1.0, 1.0)
    return imgs, labels

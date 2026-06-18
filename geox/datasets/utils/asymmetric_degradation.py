"""
Asymmetric Modality Degradation Augmentation for Omni-Rescue.

Core idea: During training, randomly degrade ONE modality while keeping the other intact.
This forces the model to learn "when one modality collapses, rely entirely on the other."

Two degradation scenarios (applied per-sample in the batch):
  1. RGB degradation  → simulates night / extreme low-light (LiDAR still works)
  2. Depth degradation → simulates glass / reflective surfaces (RGB is fine)
  3. Clean            → both modalities intact (normal training)
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def apply_asymmetric_degradation_batch(
    images: torch.Tensor,   # [B, S, 3, H, W]  float32, range [0, 1]
    depth: torch.Tensor,    # [B, S, H, W, 1]  float32
    mask: torch.Tensor,     # [B, S, H, W]     float32, {0, 1}
    rgb_deg_prob: float = 0.30,    # probability of RGB degradation per sample
    depth_deg_prob: float = 0.30,  # probability of depth degradation per sample
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies asymmetric modality degradation independently for each sample in the batch.

    Called inside the aggregator's forward() BEFORE patch embedding and ImageNet
    normalization, so the visual backbone sees the degraded image directly.

    Degradation is applied symmetrically across all S views of a sample —
    i.e., if sample b gets RGB degradation, ALL S views of that sample are degraded.
    This mimics real scenarios (night-time affects all frames; glass affects all depth).

    Args:
        images:        [B, S, 3, H, W] float32, value range [0, 1]
        depth:         [B, S, H, W, 1] float32
        mask:          [B, S, H, W]    float32
        rgb_deg_prob:  probability a sample undergoes RGB degradation
        depth_deg_prob: probability a sample undergoes depth degradation

    Returns:
        images, depth, mask — same shapes, with degradation applied in-place (cloned).
    """
    B, S, C, H, W = images.shape
    images = images.clone()
    depth = depth.clone()
    mask = mask.clone()

    # Per-sample scenario sampling on CPU for speed
    rand_vals = torch.rand(B)

    for b in range(B):
        r = rand_vals[b].item()
        if r < rgb_deg_prob:
            # ── Scenario A: extreme RGB degradation, depth is untouched ──────────
            images[b] = _degrade_rgb_views(images[b])
        elif r < rgb_deg_prob + depth_deg_prob:
            # ── Scenario B: depth degradation (holes), RGB is untouched ──────────
            depth[b], mask[b] = _degrade_depth_views(depth[b], mask[b])
        # else: clean sample — nothing to do

    return images, depth, mask


# ─────────────────────────── RGB degradation ─────────────────────────────────

def _degrade_rgb_views(views: torch.Tensor) -> torch.Tensor:
    """
    Simulate extreme low-light / sensor noise for all S views of one sample.

    views: [S, 3, H, W], range [0, 1]
    """
    # 1. Random severe darkening (retain only 2–15% of brightness)
    brightness = torch.empty(1).uniform_(0.02, 0.15).item()
    views = views * brightness

    # 2. Gaussian noise — simulate photon shot noise + read noise
    noise_sigma = torch.empty(1).uniform_(0.04, 0.18).item()
    noise = torch.randn_like(views) * noise_sigma

    # 3. Optionally add a colour cast (e.g. orange street-light tint)
    if torch.rand(1).item() < 0.4:
        cast = torch.zeros(3, device=views.device, dtype=views.dtype)
        cast[0] = torch.empty(1).uniform_(0.02, 0.08).item()   # R channel boost
        cast[2] = -torch.empty(1).uniform_(0.01, 0.05).item()  # B channel reduce
        views = views + cast.view(1, 3, 1, 1)

    views = (views + noise).clamp_(0.0, 1.0)
    return views


# ─────────────────────────── Depth degradation ───────────────────────────────

def _degrade_depth_views(
    depth: torch.Tensor,   # [S, H, W, 1]
    mask: torch.Tensor,    # [S, H, W]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate specular / reflective surfaces for all S views of one sample.

    Punches random rectangular holes into the depth map — the model must then
    rely on RGB alone for the missing regions.

    depth: [S, H, W, 1], mask: [S, H, W]
    """
    S, H, W, _ = depth.shape

    # Number and size of holes are sampled once and applied to all views
    num_holes = int(torch.randint(3, 9, (1,)).item())
    new_mask = mask.clone()

    for _ in range(num_holes):
        hole_h = int(torch.empty(1).uniform_(0.08, 0.40).item() * H)
        hole_w = int(torch.empty(1).uniform_(0.08, 0.40).item() * W)
        y0 = int(torch.randint(0, max(1, H - hole_h), (1,)).item())
        x0 = int(torch.randint(0, max(1, W - hole_w), (1,)).item())
        new_mask[:, y0:y0 + hole_h, x0:x0 + hole_w] = 0.0

    # Zero depth where mask is invalid
    new_depth = depth * new_mask.unsqueeze(-1)
    return new_depth, new_mask

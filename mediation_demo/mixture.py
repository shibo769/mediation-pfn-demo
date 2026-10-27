"""Inference-only components of the frozen error-distribution mixture.

The definitions match the training code's v10-continuous-expert-mixture format.
Checkpoints are loaded with PyTorch's restricted weights-only unpickler.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ErrorDensity(nn.Module):
    def __init__(self, dim: int, width: int = 128, components: int = 5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.LayerNorm(width),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3 * components))

    def forward(self, x):
        weights, means, scales = self.net(x).chunk(3, -1)
        return weights.softmax(-1), means, .05 + F.softplus(scales)


class Router(nn.Module):
    def __init__(self, dim: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, width), nn.GELU(),
                                 nn.LayerNorm(width), nn.Linear(width, 3))

    def forward(self, x):
        concentrations = .5 + F.softplus(self.net(x))
        return concentrations / concentrations.sum(-1, keepdim=True)


class Scaled(nn.Module):
    def __init__(self, base, scale):
        super().__init__()
        self.base, self.scale = base, float(scale)

    def forward(self, x):
        weights, means, scales = self.base(x)
        return weights, means * self.scale, scales * self.scale


def load_mixture(path, device="cpu", router_name="sft"):
    """Load a matching, tensor/primitive-only v10 mixture export."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != "v10-continuous-expert-mixture":
        raise ValueError("Expected a v10-continuous-expert-mixture checkpoint")
    if len(checkpoint["head_states"]) != 3 or len(checkpoint["scales"]) != 3:
        raise ValueError("Expected exactly three error experts")
    if router_name not in checkpoint["router_states"]:
        raise ValueError(f"Router {router_name!r} is absent from this checkpoint")
    heads = []
    for state, scale in zip(checkpoint["head_states"], checkpoint["scales"]):
        base = ErrorDensity(checkpoint["dim"], checkpoint["head_width"]).to(device)
        base.load_state_dict(state)
        heads.append(Scaled(base, scale).to(device).eval())
    router = Router(checkpoint["dim"], checkpoint["router_width"]).to(device)
    router.load_state_dict(checkpoint["router_states"][router_name])
    mean, std = checkpoint["mean"], checkpoint["std"]
    if isinstance(mean, torch.Tensor): mean = mean.detach().cpu().numpy()
    if isinstance(std, torch.Tensor): std = std.detach().cpu().numpy()
    return heads, router.eval(), mean, std


def mix_experts(parts, weights, top2=True):
    if top2:
        indices = weights.topk(2, dim=-1).indices
        mask = torch.zeros_like(weights).scatter_(-1, indices, 1.)
        weights = weights * mask
        weights = weights / weights.sum(-1, keepdim=True)
    return (torch.cat([weights[:, i, None] * p[0] for i, p in enumerate(parts)], -1),
            torch.cat([p[1] for p in parts], -1), torch.cat([p[2] for p in parts], -1))


def quantile(parts, probability, iterations=48):
    if not 0 < probability < 1:
        raise ValueError("Quantile probability must be between 0 and 1")
    weights, means, scales = parts
    if not all(torch.isfinite(x).all() for x in parts):
        raise ValueError("Nonfinite error distribution")

    def cdf(y):
        return (weights * torch.special.ndtr((y[..., None] - means) / scales)).sum(-1)

    lo = (means - 14 * scales).min(-1).values
    hi = (means + 14 * scales).max(-1).values
    if not bool(((cdf(lo) <= probability) & (cdf(hi) >= probability)).all()):
        raise ValueError("Could not bracket mixture quantile")
    for _ in range(iterations):
        mid = (lo + hi) / 2
        below = cdf(mid) < probability
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return (lo + hi) / 2


@torch.inference_mode()
def interval_from_features(features, estimate, standard_error, model, alpha=.05, device="cpu"):
    """Invert the routed standardized-error CDF for a single new table."""
    import numpy as np

    heads, router, mean, std = model
    features = np.asarray(features, dtype=np.float32)
    mean, std = np.asarray(mean), np.asarray(std)
    if features.shape != mean.shape or mean.shape != std.shape:
        raise ValueError("Checkpoint and feature dimensions differ; use a matching model bundle")
    if not np.isfinite(features).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Nonfinite features or invalid normalization")
    if not np.isfinite(estimate) or not np.isfinite(standard_error) or standard_error <= 0:
        raise ValueError("Invalid estimate or standard error")
    x = torch.as_tensor(((features - mean) / std)[None], dtype=torch.float32, device=device)
    weights = router(x)
    parts = mix_experts([head(x) for head in heads], weights, top2=True)
    indices = weights.topk(2, dim=-1).indices
    mask = torch.zeros_like(weights).scatter_(-1, indices, 1.)
    selected_weights = weights * mask
    selected_weights = selected_weights / selected_weights.sum(-1, keepdim=True)
    lower_error = float(quantile(parts, alpha / 2).item())
    upper_error = float(quantile(parts, 1 - alpha / 2).item())
    mean_error = float((parts[0] * parts[1]).sum(-1).item())
    return ((estimate - standard_error * upper_error,
             estimate - standard_error * lower_error),
            selected_weights[0].cpu().tolist(), mean_error)

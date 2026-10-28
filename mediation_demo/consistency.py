"""Frozen-PFN nuisance adapter and mediation-identity post-training inference.

The direct eta prediction is used only to regularize training. At inference,
only adjusted q0 and outcome regressions enter the existing mediation EIF.
"""
from pathlib import Path
import hashlib
import math

import numpy as np
import torch

from .data import pad_x
from .inference import canonical_nie, mediation_features
from .train_consistency import Adapter


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_adapter(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = Adapter(state["input_dim"], state["width"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    center = np.asarray(state["center"], dtype=np.float32)
    scale = np.asarray(state["scale"], dtype=np.float32)
    if center.shape != (state["input_dim"],) or scale.shape != center.shape:
        raise ValueError("Invalid adapter normalization shape")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid adapter normalization")
    return model, center, scale


def adapt_view(x, nuisance, adapter):
    """Apply the adapter to one held-out view without consuming A, M, or Y."""
    model, center, scale = adapter
    x, out = np.asarray(x, dtype=np.float32), np.asarray(nuisance, dtype=float).copy()
    if len(center) == 60:
        padded, mask = pad_x(x)
        x = np.concatenate((padded, np.broadcast_to(mask, padded.shape)), axis=1)
    if x.ndim != 2 or x.shape[1] != len(center) or out.shape != (len(x), 5):
        raise ValueError("Adapter/input dimension mismatch")
    base = out[:, [1, 3, 4]]
    with torch.inference_mode():
        pred, _ = model(torch.as_tensor((x - center) / scale),
                        torch.as_tensor(base, dtype=torch.float32))
    out[:, [1, 3, 4]] = pred.numpy()[:, :3]
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite adapted nuisances")
    return out


def adapted_score(rows, base_result, adapter, alpha=.05):
    """Recompute scores and features from the three adapted held-out views."""
    rows = np.asarray(rows, dtype=float)
    raw_views = np.asarray(base_result["views"], dtype=float)
    if raw_views.ndim != 3 or raw_views.shape[0] != 3:
        raise ValueError("Exactly three precomputed support views are required")
    views = np.stack([adapt_view(rows[:, :-3], view, adapter) for view in raw_views])
    consensus = views.mean(0)
    summaries = [canonical_nie(rows, view, alpha=alpha) for view in views]
    summaries.append(canonical_nie(rows, consensus, alpha=alpha))
    generic = np.asarray(base_result["x_single"], dtype=np.float32).copy()
    generic[2], generic[3] = summaries[0]["estimate"], math.log(summaries[0]["se"])
    features = mediation_features(rows, consensus, summaries, base_result["metadata"], generic)
    return summaries[-1], features

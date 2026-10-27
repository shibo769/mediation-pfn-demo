"""Small inference API for binary-mediator natural indirect effects."""
from __future__ import annotations

import numpy as np

from .data import make_folds, observed_rows
from .inference import canonical_nie, crossfit_views
from .mixture import interval_from_features, load_mixture
from .model import load_checkpoint


def estimate(table, backbone_path, mixture_path, *, alpha=.05, fold_seed=2026,
             view_seed=2027, device="cpu"):
    """Estimate the treated NIE using trusted, matching v9/v10 checkpoints.

    ``table`` is a mapping with numeric ``x`` (n by p), ``a``, ``m``, and ``y``.
    Treatment and mediator must be binary. The method uses two folds and three
    support views, then inverts the top-2 mixture CDF for the interval. The
    pretrained backbone and post-trained mixture are separate local artifacts.
    This function never consumes a ground-truth effect.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    rows = observed_rows(table, dtype=np.float32)
    folds = make_folds(len(rows), fold_seed)
    model = load_checkpoint(backbone_path, device=device)
    prediction = crossfit_views(model, rows, folds, seed=view_seed, device=device)
    score = canonical_nie(rows, prediction["nuisance"], alpha=alpha)
    mixture = load_mixture(mixture_path, device=device)
    interval, router_weights, mean_error = interval_from_features(
        prediction["x_multi"], score["estimate"], score["se"], mixture,
        alpha=alpha, device=device,
    )
    return {"estimate": score["estimate"] - score["se"] * mean_error,
            "score_estimate": score["estimate"], "standard_error": score["se"],
            "interval": list(interval), "nominal_coverage": 1-alpha,
            "rows": score["used_rows"], "folds": 2,
            "support_views": 3, "clip_rate": score["clipping_rate"],
            "router_weights": router_weights}


def oracle_demo(table, mechanism, *, alpha=.05):
    """Evaluate a Wald score with true nuisances, available only in simulation.

    This is an oracle sanity check, not the foundation model, a cross-fitted
    prediction, or evidence of post-training performance.
    """
    nuisance = mechanism.nuisance(table["x"])
    values = np.column_stack([nuisance[k] for k in ("g", "q0", "q1", "u0", "u1")])
    score = canonical_nie(observed_rows(table, dtype=np.float64), values, alpha=alpha)
    return {"estimate": score["estimate"], "standard_error": score["se"],
            "interval": [score["lo"], score["hi"]], "truth": mechanism.theta_NIE,
            "nominal_coverage": 1-alpha, "rows": score["used_rows"],
            "mode": "oracle_nuisance_wald_simulation_only"}

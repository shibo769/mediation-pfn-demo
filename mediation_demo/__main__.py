"""Run an oracle simulation or infer from local research checkpoint exports."""
from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

from .data import FAMILIES, sample_mechanism
from .pipeline import estimate, oracle_demo


def read_csv(path):
    """Read numeric columns A, M, Y and 1–30 covariate columns."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {"A", "M", "Y"}.issubset(fields):
            raise ValueError("CSV must have unique column names including A, M, and Y")
        covariates = [c for c in fields if c not in {"A", "M", "Y"}]
        if not 1 <= len(covariates) <= 30:
            raise ValueError("CSV must have 1 to 30 numeric covariate columns")
        data = list(reader)
    if not data:
        raise ValueError("CSV is empty")
    try:
        return {"x": np.asarray([[float(row[c]) for c in covariates] for row in data]),
                "a": np.asarray([float(row["A"]) for row in data]),
                "m": np.asarray([float(row["M"]) for row in data]),
                "y": np.asarray([float(row["Y"]) for row in data])}
    except (ValueError, TypeError) as error:
        raise ValueError("Every CSV value must be numeric and nonmissing") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("oracle", "pfn"), default="oracle")
    parser.add_argument("--family", choices=FAMILIES, default="smooth")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--checkpoint", default="weights/backbone_13m.pt",
                        help="Trusted 13M backbone checkpoint")
    parser.add_argument("--mixture", default="weights/mixture_sft.pt",
                        help="Trusted matching post-trained mixture checkpoint")
    parser.add_argument("--csv", help="Numeric input table; requires --mode pfn")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.mode == "oracle" and args.csv:
        parser.error("Oracle mode uses synthetic truth and cannot analyze a real CSV")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    if args.csv:
        table, mechanism = read_csv(args.csv), None
    else:
        mechanism = sample_mechanism(args.seed, args.p, args.family)
        table = mechanism.draw(args.n, args.seed + 1)
    if args.mode == "oracle":
        result = oracle_demo(table, mechanism, alpha=args.alpha)
    else:
        result = estimate(table, args.checkpoint, args.mixture,
                          alpha=args.alpha, device=args.device)
        result["mode"] = "pretrained_pfn_with_routed_error_mixture"
        if mechanism is not None:
            result["synthetic_truth_for_evaluation_only"] = mechanism.theta_NIE
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

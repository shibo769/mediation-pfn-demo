"""Numerical and end-to-end tests; synthetic weights are not trained models."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mediation_demo.data import make_folds, observed_rows, sample_mechanism
from mediation_demo.inference import canonical_nie, crossfit_views
from mediation_demo.mixture import ErrorDensity, Router, mix_experts, quantile
from mediation_demo.model import MediationPFN, ModelConfig, save_checkpoint
from mediation_demo.pipeline import estimate, oracle_demo


class DemoTests(unittest.TestCase):
    def test_oracle_truth_and_score(self):
        mechanism = sample_mechanism(4, 3, "smooth", population_size=512)
        table = mechanism.draw(300, 5)
        expected = mechanism.nuisance(mechanism.population)
        truth = np.mean((expected["q1"] - expected["q0"]) * (expected["mu11"] - expected["mu10"]))
        self.assertAlmostEqual(mechanism.theta_NIE, truth, places=12)
        result = oracle_demo(table, mechanism)
        self.assertTrue(math.isfinite(result["estimate"]))
        self.assertGreater(result["standard_error"], 0)
        self.assertLess(result["interval"][0], result["interval"][1])
        self.assertEqual(result["mode"], "oracle_nuisance_wald_simulation_only")

    def test_mixture_inversion_and_top_two(self):
        # A mixture of identical N(0,1) experts must retain its known quantiles.
        normal = (torch.ones((1, 1)), torch.zeros((1, 1)), torch.ones((1, 1)))
        mixed = mix_experts([normal, normal, normal], torch.tensor([[.2, .3, .5]]))
        self.assertAlmostEqual(float(mixed[0].sum()), 1., places=6)
        self.assertAlmostEqual(float(quantile(mixed, .975)), 1.959964, places=4)
        self.assertAlmostEqual(float(quantile(mixed, .025)), -1.959964, places=4)

    def test_full_pipeline_with_fixture_weights(self):
        # Tiny random fixtures exercise interfaces, not statistical performance.
        torch.manual_seed(2)
        mechanism = sample_mechanism(8, 3, "smooth", population_size=256)
        table = mechanism.draw(24, 9)
        model = MediationPFN(ModelConfig(width=16, heads=4, inducing_tokens=4,
                                        depth=1, ff_multiplier=2, size="fixture"))
        prediction = crossfit_views(model.eval(), observed_rows(table), make_folds(24, 11),
                                    seed=12, context_size=8, query_batch=16)
        dim = prediction["x_multi"].shape[0]
        heads = [ErrorDensity(dim, 12) for _ in range(3)]
        router = Router(dim, 8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backbone_path, mixture_path = root / "backbone.pt", root / "mixture.pt"
            save_checkpoint(backbone_path, model)
            torch.save({"format": "v10-continuous-expert-mixture", "dim": dim,
                        "head_width": 12, "router_width": 8,
                        "head_states": [head.state_dict() for head in heads],
                        "scales": [1., 1.25, .8], "router_states": {"sft": router.state_dict()},
                        "mean": torch.zeros(dim), "std": torch.ones(dim)}, mixture_path)
            result = estimate(table, backbone_path, mixture_path, fold_seed=11, view_seed=12)
        self.assertEqual(result["rows"], 24)
        self.assertEqual(result["folds"], 2)
        self.assertEqual(result["support_views"], 3)
        self.assertTrue(all(math.isfinite(x) for x in result["interval"]))
        self.assertLess(result["interval"][0], result["interval"][1])
        self.assertEqual(sum(w > 0 for w in result["router_weights"]), 2)

    def test_reject_invalid_treatment(self):
        rows = np.array([[0., 2., 0., 1.], [1., 0., 0., 2.],
                         [2., 1., 1., 3.], [3., 1., 0., 4.]])
        nuisance = np.tile([.5, .5, .5, 0., 1.], (4, 1))
        with self.assertRaises(ValueError):
            canonical_nie(rows, nuisance)


if __name__ == "__main__":
    unittest.main()

import json

import numpy as np
import pytest
import torch

from mediation_demo.consistency import adapt_view, load_adapter, sha256
from mediation_demo.consistency_ci import load_ci
from mediation_demo.train_consistency import Adapter, load


def test_auxiliary_eta_does_not_enter_nuisance_inference(tmp_path):
    net = Adapter(2, 8)
    adapter_path = tmp_path / "adapter.pt"
    base = {"width": 8, "input_dim": 2, "center": torch.zeros(2),
            "scale": torch.ones(2), "lambda_consistency": 1.}
    x = np.array([[.1, .3], [.3, .5]])
    nuisance = np.array([[.5, .3, .4, 1., 2.], [.5, .6, .2, 0., 1.]])
    torch.save({**base, "state_dict": net.state_dict()}, adapter_path)
    before = adapt_view(x, nuisance, load_adapter(adapter_path))
    with torch.no_grad():
        net.net[-1].bias[3] = 100.
    torch.save({**base, "state_dict": net.state_dict()}, adapter_path)
    after = adapt_view(x, nuisance, load_adapter(adapter_path))
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(after[:, [0, 2]], nuisance[:, [0, 2]])


def test_training_refuses_test_or_overlapping_mechanisms(tmp_path):
    path = tmp_path / "features.npz"
    data = dict(x=np.ones((2, 2)), base=np.array([[.3, 0, 1], [.4, 1, 2]]),
                target=np.ones((2, 4)) * .5, mechanism_id=np.array(["same", "same"]),
                split=np.array(["train", "dev"]))
    np.savez(path, **data)
    with pytest.raises(ValueError, match="overlap"):
        load(path)
    data["split"] = np.array(["train", "test"])
    np.savez(path, **data)
    with pytest.raises(ValueError, match="train/dev only"):
        load(path)


def test_ci_head_rejects_nonmatching_adapter(tmp_path):
    weights = tmp_path / "ci_head.pt"
    torch.save({"unused": 1}, weights)
    (tmp_path / "ci_freeze.json").write_text(json.dumps({
        "status": "ci_head_frozen_before_test", "head_sha256": sha256(weights),
        "adapter_sha256": "another-adapter"}), encoding="utf-8")
    with pytest.raises(ValueError, match="freeze mismatch"):
        load_ci(tmp_path, "current-adapter")

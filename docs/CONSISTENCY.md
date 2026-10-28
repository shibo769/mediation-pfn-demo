# Mediation-identity consistency post-training

The consistency module is an additional post-training path for a **frozen**
mediation PFN. It is based on a given mediation structure; it does not learn a
causal graph or impose an acyclicity penalty on a learned adjacency matrix.

For binary treatment A and mediator M, the adapter adjusts cross-fitted
predictions of q0(X) = P(M=1 | A=0, X), mu10(X) = E[Y | A=1, M=0, X], and
mu11(X) = E[Y | A=1, M=1, X]. It also has a separately parameterized direct
prediction of eta10(X). The synthetic training tasks provide oracle row-level
targets for these four quantities. We minimize the sum of their mean squared
errors plus

```text
lambda * mean((eta10_direct - ((1 - q0) * mu10 + q0 * mu11)) ** 2)
```

The direct eta prediction is a **training-only auxiliary output**. At inference,
the adjusted q0, mu10, and mu11 enter the existing mediation EIF; eta10_direct
does not enter that score. Its indirect effect on those inputs depends on the
learned adapter and the consistency loss.

The original error mixture was trained for unadapted EIF estimates. The
experimental pipeline instead trains a matching single error-density CI head
on separate synthetic mechanisms, then selects its epoch using a disjoint DEV
bank. `estimate_consistency` requires this frozen head and checks its adapter
SHA. As with all causal mediation estimators, interpreting NIE requires the
relevant identification assumptions.

## Experiment reported so far

In the 1M model experiment, four adapter candidates used the same training
budget, with lambda 0, 0.1, 0.5, and 1. The DEV supervised loss selected
lambda 1 before the independent test. At the 95% nominal level, among 200 ID
tables, coverage was 86.5% for lambda 0 and 87.5% for lambda 1; among 160 OOD
tables, both were 83.125%. On 200 stress tables representing five scenarios
paired on 40 base mechanisms, both were 88.0%. All methods had zero failures.
The paired ID coverage difference was 1 percentage point, with a 95% bootstrap
interval of [0, 2.5] percentage points using mechanism-level resampling.
Original 1M SFT coverage was 91.0%, 85.625%, and 93.0%, respectively, using
its existing routed mixture rather than the separately retrained single head.
All four candidate checkpoints and logs were retained in the research run.
This experiment does not establish an OOD coverage improvement.

## Training and inference inputs

The `v0.1.0` weights contain a 13M backbone and original SFT mixture. The
[v0.2.0 release](https://github.com/shibo769/mediation-pfn-demo/releases/tag/v0.2.0)
adds `consistency_1m_bundle.zip` with the matched 1M backbone, DEV-selected
lambda=1 adapter, and separately retrained CI head. The ZIP SHA-256 is
`d5d7fa87345965f70f527377531fe5cd9e0344cbd88cb08daf2e0c68176f3532`.
The research-format feature cache, CI shard generator, and full evaluation
scheduler are not included in this demo.

```bash
gh release download v0.2.0 --repo shibo769/mediation-pfn-demo \
  --pattern consistency_1m_bundle.zip --dir weights
python -m zipfile -e weights/consistency_1m_bundle.zip weights/consistency_1m
```

Given an already prepared synthetic-oracle train/DEV cache with `x`, `base`
(`q0, mu10, mu11`), `target` (`q0, mu10, mu11, eta10`), `mechanism_id`, and
`split`, train candidates and select on DEV:

```bash
python -m mediation_demo.train_consistency --cache train_dev.npz \
  --output candidate0 --lambda-consistency 0
python -m mediation_demo.train_consistency --cache train_dev.npz \
  --output candidate1 --lambda-consistency 1
python -m mediation_demo.freeze_consistency --cache train_dev.npz \
  --candidates candidate0 candidate1 --output frozen
```

The trainer rejects test rows and overlapping train/DEV mechanism IDs. CI-head
training is provided by `python -m mediation_demo.train_consistency_ci --help`;
it requires separate research-format, provenance-checked synthetic CI shards.
Generating those shards requires the research pipeline. Output directories must
not already exist.

Run the released matched artifacts on a synthetic table:

```bash
python -m mediation_demo --mode pfn \
  --checkpoint weights/consistency_1m/backbone_1m.pt \
  --consistency-adapter weights/consistency_1m/consistency_adapter_1m.pt \
  --consistency-ci-dir weights/consistency_1m/consistency_ci_1m
```

The CI loader verifies its adapter and head hashes. The caller must also use
the backbone for which that adapter was trained. For a numeric CSV, add
`--csv path/to/data.csv`; its columns follow the main README. Selection and
reporting should use independent DEV and test mechanisms.

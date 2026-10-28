# Amortized Inference and Post-Training for Causal Mediation

A small, self-contained demonstration of natural indirect effect (NIE)
estimation with a mediation score and cross-fitted nuisance predictions. The
research pipeline combines a tabular foundation model with learned
standardized-error distributions to construct confidence intervals. Its
post-training pipeline learns uncertainty heads and experiments with SFT,
DPO, and reinforcement learning with verifiable rewards (RLVR) for routing
between them.

## Post-training at a glance

After nuisance prediction with the 13.5M-parameter PFN, we compute a
cross-fitted mediation-score estimate and standard error. The PFN is frozen
while the post-training modules learn to represent the standardized NIE
estimation error. For synthetic training tasks, the target NIE is known and
provides supervision; it is not used at inference time.

| Component | What it learns |
| --- | --- |
| **Error heads** | Global, undercoverage-oriented, and overcoverage-oriented distributions of standardized estimation error. |
| **SFT** | Router weights for combining the error distributions, trained on synthetic error labels. |
| **DPO** | Preferences between candidate router weights, constructed from simulated group-level interval rewards. |
| **RLVR** | A router policy updated using verifiable simulated rewards for interval quality. |
| **Mediation consistency** | A frozen-PFN nuisance adapter learns the mediation identity with an auxiliary outcome head. |

The implemented comparisons include SFT → DPO → RLVR, SFT → RLVR, and a
matched additional-SFT control. At inference, the released pipeline combines
the two highest-weight heads and uses their mixture quantiles to form the
interval. The released `mixture_sft.pt` contains the three trained error heads
and the SFT router; it does not contain the DPO or RLVR router checkpoints.
In the development selection for this release, all three router candidates
retained their initial parameters. This does not establish a post-training
improvement or an OOD result.

The newer component-specific, calibration, and distribution-shift
post-training experiments remain under evaluation and are not included in
this demo. The separate mediation-consistency path has a matching 1M backbone,
adapter, and CI head in the [v0.2.0 release](https://github.com/shibo769/mediation-pfn-demo/releases/tag/v0.2.0).
See [consistency notes](docs/CONSISTENCY.md) for its inputs and evaluation status.

This repository provides two paths:

1. **Run the simulation demonstration.** Generate a synthetic mediation
   dataset and compute the score using the simulator's nuisance functions.
   Because these functions are known, this is an oracle example for checking
   the score and the target; it is not an evaluation of a trained model.
2. **Use compatible research checkpoints.** Supply a trained nuisance model
   and a matching uncertainty-mixture checkpoint to run the cross-fitted PFN
   pipeline on a synthetic table or your own numeric CSV.

## Quick start

Python 3.10 or newer is required. Install the package in an environment with
PyTorch, NumPy, and SciPy:

```bash
git clone https://github.com/shibo769/mediation-pfn-demo.git
cd mediation-pfn-demo
python -m pip install -e .
python -m mediation_demo --mode oracle
```

The oracle command prints a simulated NIE estimate, standard error, Wald
interval, and the known finite-population NIE. The known target is available
only because the dataset is simulated.

Download the matching checkpoint bundle from the
[v0.1.0 release](https://github.com/shibo769/mediation-pfn-demo/releases/tag/v0.1.0),
either in a browser or with the GitHub CLI:

```bash
gh release download v0.1.0 --repo shibo769/mediation-pfn-demo --dir weights
```

Then run the pretrained model on a synthetic table:

```bash
python -m mediation_demo --mode pfn \
  --checkpoint weights/backbone_13m.pt --mixture weights/mixture_sft.pt
```

For the separate 1M mediation-consistency path, download and unpack its matched
backbone, adapter, and retrained CI head:

```bash
gh release download v0.2.0 --repo shibo769/mediation-pfn-demo \
  --pattern consistency_1m_bundle.zip --dir weights
python -m zipfile -e weights/consistency_1m_bundle.zip weights/consistency_1m
python -m mediation_demo --mode pfn \
  --checkpoint weights/consistency_1m/backbone_1m.pt \
  --consistency-adapter weights/consistency_1m/consistency_adapter_1m.pt \
  --consistency-ci-dir weights/consistency_1m/consistency_ci_1m
```

The bundle SHA-256 is
`d5d7fa87345965f70f527377531fe5cd9e0344cbd88cb08daf2e0c68176f3532`.
It contains the DEV-selected lambda=1 adapter and its separately retrained
CI head, trained with synthetic-oracle supervision; see the
[experiment details](docs/CONSISTENCY.md#experiment-reported-so-far).

The backbone and mixture must come from the same fitted pipeline. If these
files are unavailable, the simulation demonstration still runs; its oracle
example should not be presented as a PFN benchmark.
Both checkpoint loaders use PyTorch's restricted `weights_only=True` mode;
use checkpoint files from sources you trust.

The released backbone has 13,496,449 parameters. The factual SFT selection
retained the initial backbone checkpoint. Three uncertainty heads were
trained; development selection retained the initial router parameters for
the SFT, DPO, and reward-based candidates. Independent ID and distribution
shift experiments are ongoing; their final results are not included here.

| Release asset | SHA-256 |
| --- | --- |
| `backbone_13m.pt` | `074c588144955ded3c2b9eafb6571182f47b9b78d399ec574cbed58cb99fe223` |
| `mixture_sft.pt` | `bb98c8b3e9ca263598f2f6c221c98599663012963c74f4e5f2d9e7c64754d9e9` |

To analyze a numeric CSV, pass `--csv data.csv` in PFN mode. It must contain
columns named `A`, `M`, and `Y` plus 1–30 covariate columns. All values must
be numeric and nonmissing, and `A` and `M` must be coded as 0 or 1. For
example:

```text
X1,X2,A,M,Y
0.2,-0.1,1,0,1.4
0.4,0.3,0,1,0.8
-0.3,0.2,1,1,1.1
0.1,-0.5,0,0,-0.2
```

```bash
python -m mediation_demo --mode pfn --csv data.csv \
  --checkpoint weights/backbone_13m.pt --mixture weights/mixture_sft.pt
```

The CLI accepts `--alpha` (default `0.05`), `--n`, `--p`, `--family`,
`--seed`, and `--device`; run `python -m mediation_demo --help` for details.
Specifying an `alpha` selects the nominal interval level; it does not by
itself establish coverage for the chosen level or dataset.

In PFN mode, `score_estimate` is the uncorrected mediation-score estimate;
`estimate` applies the mean correction from the fitted error mixture. The
interval is obtained by inverting mixture quantiles around `score_estimate`.

Run the checks with:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Training scope

The demo includes the factual nuisance-backbone training code and the
error-head/router post-training code:

```bash
python -m mediation_demo.train_backbone --help
python -m mediation_demo.train_mixture --help
python -m mediation_demo.posttrain --help
```

`train_mixture` fits the global and specialist error heads and candidate
SFT, DPO, and reward-based routers. It requires a prepared research NPZ
cache with nuisance-derived features, score estimates and standard errors,
synthetic targets, and mechanism/split/role labels. Training and development
mechanisms and training-role banks must be separated; sealed test labels are
not accepted. The demo does not include the full cache-generation,
experiment-scheduling, or evaluation pipeline, so these entry points alone
do not reproduce the research experiments or the newer post-training studies.
No real-world dataset is included.

Given a compatible train/development cache, a CPU training run can be started
as follows. The output directory must not already exist:

```bash
python -m mediation_demo.train_mixture \
  --cache path/to/train_dev_cache.npz \
  --output path/to/new_output_dir --seed 42 --device cpu --top2
```

The `train_mixture` entry point trains the continuous mixture used by the
released inference pipeline. `posttrain` contains a separate categorical-router
experiment; its checkpoints are not interchangeable with the released mixture.
The public repository does not provide the research cache needed by these
commands, so preparing a compatible cache requires additional research code.

Training a new backbone does not train a matching uncertainty mixture; do not
pair newly trained weights with the released mixture without additional
training and validation.

## What the method estimates

For covariates $X$, binary treatment $A$, binary mediator $M$, and
outcome $Y$, the demonstration targets the treated natural indirect effect

$$
\theta_{\mathrm{NIE}}=\mathbb{E}\{Y(1,M(1))-Y(1,M(0))\}.
$$

The score combines estimates of the treatment propensity, mediator
distribution, and outcome regression. In PFN mode, rows in each held-out
fold receive nuisance predictions using the opposite fold as context; the
score and its standard error are computed from the held-out rows. The
uncertainty module combines learned error distributions and inverts their
mixture to form an interval.

The synthetic generator samples treatment, mediator, and outcome in that
order. Its known NIE is computed over a fixed finite covariate population.
Real-world identification requires the corresponding consistency,
positivity, and exchangeability assumptions; this demo cannot verify them
from observed data.

See [method notes](docs/METHOD.md) for the score, cross-fitting, and interval
construction.

## Scope and experimental status

- The current example supports binary treatment and one binary mediator.
- Oracle mode uses oracle nuisance functions for a reproducible
  demonstration. It does not require or evaluate model weights.
- The 13.5-million-parameter backbone and its post-training/OOD experiments
  are research artifacts. No independent coverage, interval-length, or speed
  result is claimed by this demo.
- A learned interval is an empirical uncertainty procedure, not an automatic
  coverage guarantee under arbitrary distribution shift.

## Repository layout

```text
mediation_demo/  Nuisance model, score, interval heads, and post-training code
docs/            Method notes
tests/           Small, reproducible checks
pyproject.toml   Package metadata and dependencies
```

## Research background

The score-based estimator draws on mediation analysis and double/debiased
machine learning; the implementation in this repository is a research demo,
not a replacement for checking the assumptions or validating a particular
application. For background, see [Causal Mediation Analysis with Double
Machine Learning](https://arxiv.org/abs/2002.12710).

# Method notes

The demonstration targets the treated natural indirect effect

$$
\theta_{\mathrm{NIE}}
=\mathbb{E}\{Y(1,M(1))-Y(1,M(0))\}.
$$

The implementation assumes binary $A$ and $M$ and a numeric outcome. Let
$g(x)=P(A=1\mid X=x)$, $q_a(m\mid x)=P(M=m\mid A=a,X=x)$, and
$\mu_1(x,m)=E(Y\mid A=1,M=m,X=x)$. Define
$\eta_a(x)=\sum_{m=0}^1\mu_1(x,m)q_a(m\mid x)$.

For each observation, the two uncentered score contributions used by the
demo are

$$
s_{11}=\eta_1(X)+\frac{A}{g(X)}\{Y-\eta_1(X)\},
$$

$$
s_{10}=\eta_0(X)
+\frac{A}{g(X)}\frac{q_0(M\mid X)}{q_1(M\mid X)}\{Y-\mu_1(X,M)\}
+\frac{1-A}{1-g(X)}\{\mu_1(X,M)-\eta_0(X)\}.
$$

The uncorrected score estimate is the mean of $s_{11}-s_{10}$; the score
standard deviation divided by $\sqrt n$ supplies its standard error. The code clips
the estimated treatment and mediator probabilities to $[0.01,0.99]$ before
computing the score and reports the clipping rate.

In PFN mode, the dataset is split into two folds. Each held-out fold receives
nuisance predictions from a context selected using the other fold. The code
uses three covariate-only context-selection views and averages their nuisance
predictions before computing the score. The router selects two of three
trained error-distribution heads; their CDFs are mixed, and the mixture
quantiles are inverted around the uncorrected score estimate to form the
interval. In PFN mode, the reported `estimate` additionally subtracts the
mixture's mean standardized error times the score standard error;
`score_estimate` is the uncorrected score estimate. The corrected estimate
need not equal the interval midpoint.

The simulator can compute the treated NIE exactly for its fixed covariate
population. Oracle mode uses the generating nuisance functions and a Wald
interval; it does not load a PFN or its error heads. The target effect in an
observational dataset is not available to the estimator.

Identification of this natural indirect effect requires consistency,
positivity, and suitable treatment and mediator exchangeability assumptions.
The nominal interval level alone does not guarantee coverage, particularly
under distribution shift or weak overlap. This repository is a research
demonstration; evaluate it on the intended design before drawing conclusions.

The mediation score is related to [Farbmacher et al., *Causal mediation
analysis with double machine learning*](https://arxiv.org/abs/2002.12710).

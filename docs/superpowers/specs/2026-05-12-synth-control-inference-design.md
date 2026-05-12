# Synthetic Control Inference — Notebook Design

**Date:** 2026-05-12
**Deliverable:** A single self-contained teaching notebook (e.g. `40 - Synth Control Inference.ipynb`) in the same numbered style as existing notebooks in this repo.
**Goal:** Develop Abadie-style placebo p-values and valid confidence intervals for synthetic control, and demonstrate their validity via a Monte Carlo coverage simulation on a synthetic factor-model panel.

## Scope

Three phases, layered:

- **Phase I — Abadie p-values.** In-space placebo distribution + post/pre RMSPE-ratio test statistic + exact rank p-value. Faithful to Abadie–Diamond–Hainmueller (2014/15).
- **Phase II — Confidence intervals.** Two methods side by side: (a) test-inversion CI (Firpo–Possebom-style); (b) conformal CI (Chernozhukov–Wüthrich–Zhu 2021).
- **Phase III — Coverage simulation.** Monte Carlo over the synthetic DGP: empirical size of the p-value, empirical coverage of each CI, across a grid of true treatment effects. This is what makes "valid" CIs verifiable rather than asserted.

Phase I is shippable on its own; Phases II and III layer on without modifying earlier code.

## Environment

Dedicated uv venv with a Jupyter kernel registered at user level, per repo convention (see `CLAUDE.md`):

```bash
uv venv .venv-synthcontrol
uv pip install numpy pandas matplotlib scipy joblib ipykernel
uv run python -m ipykernel install --user \
    --name synthcontrol --display-name "Python (synthcontrol)"
```

No `pysyncon` (see "Estimator" below). The notebook's first markdown cell tells the user to select the `Python (synthcontrol)` kernel.

## Notebook architecture

Linear top-to-bottom flow:

1. **Setup** — imports, RNG seed, kernel selection note.
2. **Synthetic DGP** — `simulate_panel(...)`.
3. **SC estimator** — `fit_sc(...)`.
4. **Phase I — Abadie p-values** — placebo loop, RMSPE-ratio, rank p-value, one worked panel + diagnostic plot.
5. **Phase II — CIs** — test-inversion CI and conformal CI on the same worked panel.
6. **Phase III — Coverage simulation** — Monte Carlo over (rep, τ); size, power, coverage tables and a coverage plot.
7. **Discussion** — what the simulation taught us about each method's behavior.

## Synthetic DGP

```python
def simulate_panel(J=30, T=40, T0=30, r=2, sigma=0.5, tau=0.0,
                   inside_hull=True, seed=0):
    """
    Generate a factor-model panel.

    y_{it} = mu_i + lambda_i^T F_t + eps_{it},  eps ~ N(0, sigma^2)
    Treated unit (id=0) has lambda_0 inside convex hull of donor lambdas
    when inside_hull=True. Treatment effect tau added to y_{0t} for t > T0.

    Returns:
      df: long-format DataFrame [unit, time, y]
      truth: dict with tau, mu, lambda, F, treated_id, T0
    """
```

Defaults: J=30 donor units, T=40 time periods, T0=30 (30 pre-, 10 post-), r=2 latent factors, σ=0.5, treated unit's loading vector constructed as a random convex combination of donor loadings (literally inside the convex hull). `unit=0` is treated; `unit=1..J` are donors. Long format suitable for direct use by the estimator.

The function is deterministic given `seed` so the coverage Monte Carlo is reproducible: each rep calls `simulate_panel(..., seed=rep_idx, tau=tau)`.

## SC estimator

```python
def fit_sc(df, treated_id, T0):
    """
    Fit Abadie SC matching on lagged outcomes only (no extra predictors).
    Solve simplex-constrained QP for donor weights W directly.

    Returns:
      gap:  pd.Series indexed by time, length T,  y_treated - y_synth
      W:    pd.Series indexed by donor_id, weights on simplex
      yhat: pd.Series indexed by time, the synthetic counterfactual
    """
```

Internals:

- Pivot the long DataFrame to wide: rows = time, columns = unit; split into `y_treated` and `Y_donors`.
- Pre-period blocks: `y_treated[:T0]` and `Y_donors[:T0]`.
- Solve

  $$\min_W \; \|y_1^{\text{pre}} - Y_0^{\text{pre}} W\|_2^2 \quad \text{s.t.} \quad W \geq 0, \; \mathbf{1}^\top W = 1$$

  with `cvxpy` (OSQP solver). ~10 lines. Robust to the simplex constraint and well-behaved across the ~60k fits Phase III will throw at it; SLSQP from `scipy.optimize.minimize` occasionally fails near the simplex boundary, which would corrupt placebo distributions silently.
- Reconstruct `yhat = Y_donors @ W` over full T; `gap = y_treated - yhat`.

**Why no V matrix.** With predictors = pre-period outcomes and no covariates, V's nested-optimization rationale collapses (V tries to minimize the same loss it weights). pysyncon would still run the optimization but it adds runtime, not signal. From-scratch QP gives a transparent estimator and a ~10–50× faster placebo loop, which matters for the R × τ-grid Phase III sim. (Caveat: under non-stationary factors, time-weighting V to downweight distant pre-periods can still help. Not relevant here — our DGP is stationary.)

**Identifiability note.** With J=30 donors and r=2 latent factors, W is massively under-identified in the noiseless limit (any face of the simplex containing $\lambda_0$ is admissible). This doesn't break inference because all our test statistics are functions of the gap series, which is invariant to the choice of W within the admissible set — but the discussion cell must not claim "we recover the donor weights."

## Phase I — Abadie p-values

Three small pure functions:

**`placebo_distribution(df, T0) -> pd.DataFrame`**
For each unit `j` in `0..J`, refit `fit_sc(df, treated_id=j, T0)` and store the resulting gap series. Returns shape `(T, J+1)` — one column per "treated" identity. The real treated unit is just one column among placebos.

**`rmspe_ratio(gap, T0) -> float`**
Abadie 2014/15 test statistic:

$$r_j = \frac{\sqrt{\frac{1}{T-T_0}\sum_{t>T_0} \text{gap}_{jt}^2}}{\sqrt{\frac{1}{T_0}\sum_{t \leq T_0} \text{gap}_{jt}^2}}$$

Pre-RMSPE in the denominator stops badly-fit donors from masquerading as large effects.

**`abadie_pvalue(ratios, treated_idx) -> float`**
Exact rank p-value:

$$p = \frac{\#\{j \in \{0, 1, \dots, J\} : r_j \geq r_{\text{treated}}\}}{J+1}$$

The numerator ranges over **all J+1 units including the treated one**, so $p \geq 1/(J+1)$ always (the floor is the smallest p-value the test can return). Ties are handled conservatively: a placebo with $r_j = r_{\text{treated}}$ counts toward the numerator. (This matches the standard Abadie convention.)

One-sided ratio-based (Abadie's classic — the default reported in the paper). A two-sided variant on signed mean post-period gap is also implemented for the Phase III coverage analysis, but the default in the worked-example cell is the ratio-based one.

**Worked-output cell.** One panel at τ=1, fit, placebo-loop, plot:

- Left subplot: gap series for treated in bold + all J placebo gaps in grey; vertical line at T0.
- Right subplot: histogram of `r_j` across placebos with the treated unit's `r` marked.
- Printed summary: $\hat\tau$ (mean post-period gap), $r_\text{treated}$, $p$.

Phase I is a complete, shippable deliverable here.

## Phase II — Confidence intervals

Two methods, both implemented and applied to the same worked panel.

### Test-inversion CI

For a grid of candidate effects $\tau_0$:

1. Subtract $\tau_0$ from treated unit's post-period outcomes → adjusted panel.
2. Recompute placebo distribution and Abadie p-value on the adjusted panel.
3. Keep $\tau_0$ in the CI iff the adjusted p-value > α.

CI = set of non-rejected $\tau_0$. Coverage is exact **conditional on the assumption that placebo exchangeability still holds after subtracting $\tau_0$** — i.e. under the sharp null that the treatment effect is constant across the post-period and equal to $\tau_0$, with donors exchangeable with treated. This is the Firpo–Possebom (2018) assumption; it's strong but standard, and the spec is honest about it.

Default grid: ~50 points spanning $[\hat\tau - 4\hat\sigma_{\text{placebo}}, \hat\tau + 4\hat\sigma_{\text{placebo}}]$, where $\hat\sigma_{\text{placebo}}$ is the SD of placebo mean-post-gaps. Coarse grid is fast; readers can refine.

### Conformal CI (Chernozhukov–Wüthrich–Zhu 2021)

Different exchangeability assumption: residuals of the treated unit's SC fit are *block-exchangeable* across time. For each candidate $\tau_0$:

1. Subtract $\tau_0$ from treated post-period.
2. Fit SC on the full adjusted treated series (full T, not just pre-period).
3. Compute residuals $\hat u_t$ over all T periods.
4. Define the test statistic on a length-$T_1 = T - T_0$ window of residuals:

   $$S_q(\text{window}) = \left(\frac{1}{T_1}\sum_{t \in \text{window}} |\hat u_t|^q\right)^{1/q}$$

   Default $q=1$ (mean absolute residual).
5. Compute $S_q$ on the actual post-period window. Then compute $S_q$ on **moving-block** (or cyclic) permutations: slide a length-$T_1$ window across all valid start positions in the residual series, recomputing $S_q$ at each. This handles serial dependence in residuals from the factor-model fit (which iid permutation would not).
6. Block size = $T_1 = 10$ (the post-period length, per CWZ default). The number of admissible windows is $T - T_1 + 1 = 31$.
7. P-value = rank of the actual post-period $S_q$ among all $T - T_1 + 1$ window statistics, including the actual one.
8. CI is the set of $\tau_0$ with p-value > α (same grid as test-inversion).

Returns a CI under a different (and not strictly nested) assumption from the test-inversion CI: time-exchangeable residual blocks vs. unit-exchangeable placebos. The worked-panel cell includes one sentence noting that the two CIs may legitimately disagree because they rest on different exchangeability assumptions, not because one is "wrong."

## Phase III — Coverage simulation

```python
def coverage_sim(R=2000, taus=(0.0, 0.5, 1.0, 2.0), alpha=0.05, n_jobs=-1):
    """
    For each (rep, tau): simulate panel, compute Abadie p-value,
    compute test-inversion CI, compute conformal CI.

    Returns long DataFrame:
      [rep, tau, method, pvalue, ci_lo, ci_hi, covered]
    """
```

**R = 2000.** At target coverage 0.95 with R=500 the binomial SE is ≈0.0097, so the 2σ band is ±2pp — too coarse to distinguish 95% from 93%. R=2000 cuts the SE in half (≈±1pp), which is what we need to make claims like "covers nominally" defensible. Cost: ~4× more SC fits, parallelized.

**Seeding for the τ-grid.** Within a rep, the panel is simulated with `seed=rep_idx` and the same factor draws are reused across all τ values (only the additive shift to the treated post-period changes). This is intentional — it isolates the *method's response to τ* from sampling noise in the panel. Consequence: rows of the coverage table at different τ are **paired**, not independent. Reported standard errors on coverage are computed within-τ (binomial across reps) so they're valid; cross-τ comparisons are reported as paired differences when they appear in the discussion.

Outputs:

- **Size table** — empirical rejection rate at τ=0 (target: ≤ α).
- **Power table** — rejection rate at τ ∈ {0.5, 1, 2}.
- **Coverage table** — empirical coverage of each CI method at each τ (target: ≥ 1−α), with within-τ binomial SEs.
- **Coverage plot** — τ on x, empirical coverage on y, horizontal line at 1−α, one curve per method, error bands.

Parallelized with `joblib`.

**Discussion cell.** Brief written analysis: did the Abadie p-value hit nominal size? Which CI was tighter at low τ vs. high τ? Where did each method degrade?

## Out of scope

- Real datasets (e.g. German reunification, California Prop 99) — pure synthetic DGP only, by design.
- The V predictor-weighting matrix — not needed when matching on lagged outcomes only.
- Covariates / extra predictors — same reason.
- Bayesian / structural sensitivity analysis (Zeitler et al. 2023 style) — separate project.
- `pysyncon` or other SC packages.
- A reusable Python module — single notebook only.

## Open follow-ups (not in scope for v1)

- A "fast mode" that fixes V to identity and uses Doudchenko-Imbens (already what we do — kept as a note in case we later add a strict-Abadie variant for comparison).
- Treated unit *outside* the convex hull as a misspecification scenario.
- Heavy-tailed noise as a robustness scenario.
- Larger J with fewer pre-periods (the regime where placebo p-values get coarse).

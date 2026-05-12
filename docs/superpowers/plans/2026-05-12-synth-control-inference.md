# Synth Control Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single self-contained Jupyter notebook (`40 - Synth Control Inference.ipynb`) that develops Abadie-style placebo p-values and valid confidence intervals for synthetic control, demonstrating their validity via a Monte Carlo coverage simulation on a synthetic factor-model panel.

**Architecture:** Notebook source is authored as `40 - Synth Control Inference.py` in **jupytext percent format** for clean editing; jupytext syncs to the `.ipynb` deliverable. Cells are organized into seven sections (Setup → DGP → Estimator → Phase I → Phase II → Phase III → Discussion). TDD ergonomics: each function cell is preceded by an assertion cell that fails with `NameError`/`AssertionError` until the function is correctly implemented. The notebook is re-executed end-to-end after each task to verify it runs top-to-bottom. The SC estimator is solved as a simplex-constrained QP via `cvxpy` (OSQP). Phase III uses `joblib` for parallel reps.

**Tech Stack:** Python 3.11, uv (env management), numpy, pandas, scipy, cvxpy (OSQP), matplotlib, joblib, jupytext, ipykernel, jupyter, pytest (smoke only).

**Spec:** `docs/superpowers/specs/2026-05-12-synth-control-inference-design.md`

---

## File Structure

| File | Status | Purpose |
|---|---|---|
| `40 - Synth Control Inference.py` | Create | Jupytext percent-format source — single source of truth for editing |
| `40 - Synth Control Inference.ipynb` | Create (synced) | The deliverable; auto-generated from the `.py` via `jupytext --sync` |
| `requirements-synthcontrol.txt` | Create | Pinned env for reproducibility |
| `.gitignore` | Modify (add line) | Exclude `.venv-synthcontrol/` and `__pycache__/` |
| `CLAUDE.md` | (already current) | No change |

The `.py` and `.ipynb` are paired by jupytext: edits go into `.py` (clean diffs), and `jupytext --sync` regenerates the `.ipynb`. Both files are committed; the `.ipynb` is what the user opens in Jupyter.

---

## Task 1: Environment + kernel registration

**Files:**
- Create: `.venv-synthcontrol/` (uv venv, gitignored)
- Create: `requirements-synthcontrol.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create uv venv**

Run from repo root:
```bash
uv venv .venv-synthcontrol --python 3.11
```
Expected: `Using CPython 3.11.x` and `Activate with: ...`

- [ ] **Step 2: Write requirements file**

Create `requirements-synthcontrol.txt`:
```
numpy>=1.26
pandas>=2.1
scipy>=1.11
cvxpy>=1.4
matplotlib>=3.8
joblib>=1.3
jupytext>=1.16
ipykernel>=6.28
jupyter>=1.0
pytest>=7.4
```

- [ ] **Step 3: Install dependencies**

Run:
```bash
uv pip install --python .venv-synthcontrol -r requirements-synthcontrol.txt
```
Expected: All packages resolve and install. cvxpy pulls OSQP as a default solver via wheels (no compilation needed on Windows).

- [ ] **Step 4: Register the Jupyter kernel at user level**

Run:
```bash
uv run --python .venv-synthcontrol python -m ipykernel install --user --name synthcontrol --display-name "Python (synthcontrol)"
```
Expected: `Installed kernelspec synthcontrol in <user-jupyter-kernel-dir>`

- [ ] **Step 5: Verify the kernel is visible to any Jupyter session**

Run:
```bash
jupyter kernelspec list
```
Expected: `synthcontrol` appears in the list with a path under the user-level kernel directory (Windows: `%APPDATA%\jupyter\kernels\synthcontrol`). If `jupyter` is not on PATH, run `uv run --python .venv-synthcontrol jupyter kernelspec list` — but the registration is still user-global; the kernel will appear when any other Jupyter is launched too.

- [ ] **Step 6: Update .gitignore**

Append to `.gitignore` (create file if it doesn't exist):
```
.venv-synthcontrol/
__pycache__/
.ipynb_checkpoints/
```
(Skip lines that are already present.)

- [ ] **Step 7: Commit**

```bash
git add requirements-synthcontrol.txt .gitignore
git commit -m "chore: add synthcontrol uv env and requirements"
```

---

## Task 2: Notebook scaffold (jupytext + section structure)

**Files:**
- Create: `40 - Synth Control Inference.py`
- Create: `40 - Synth Control Inference.ipynb` (auto-generated)

- [ ] **Step 1: Create the jupytext source with section headers and Setup cell**

Create `40 - Synth Control Inference.py`:
```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (synthcontrol)
#     language: python
#     name: synthcontrol
# ---

# %% [markdown]
# # Synth Control Inference
#
# Abadie-style placebo p-values and valid confidence intervals for synthetic control,
# verified by Monte Carlo on a synthetic factor-model panel.
#
# **Kernel:** select **`Python (synthcontrol)`** from the Jupyter kernel menu before running.

# %% [markdown]
# ## 1. Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from joblib import Parallel, delayed

RNG_SEED = 0
rng = np.random.default_rng(RNG_SEED)

# %% [markdown]
# ## 2. Synthetic DGP

# %% [markdown]
# ## 3. SC estimator

# %% [markdown]
# ## 4. Phase I — Abadie p-values

# %% [markdown]
# ## 5. Phase II — Confidence intervals

# %% [markdown]
# ## 6. Phase III — Coverage simulation

# %% [markdown]
# ## 7. Discussion
```

- [ ] **Step 2: Sync to .ipynb**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
```
Expected: `40 - Synth Control Inference.ipynb` created.

- [ ] **Step 3: Execute the notebook end-to-end as a smoke test**

Run:
```bash
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: No errors. The notebook just imports libraries and sets a seed; should complete in seconds.

- [ ] **Step 4: Commit**

```bash
git add "40 - Synth Control Inference.py" "40 - Synth Control Inference.ipynb"
git commit -m "feat(synth-ctrl): scaffold notebook with section structure"
```

---

## Task 3: Synthetic DGP (`simulate_panel`)

**Files:**
- Modify: `40 - Synth Control Inference.py` (add cells under "## 2. Synthetic DGP")

- [ ] **Step 1: Add a validation cell first (failing test)**

Append under section 2 in the `.py` file:
```python
# %% [markdown]
# ### Validation: simulate_panel

# %%
def _validate_simulate_panel():
    df, truth = simulate_panel(J=30, T=40, T0=30, r=2, sigma=0.5, tau=1.0, seed=42)
    # Long format with the right columns
    assert list(df.columns) == ["unit", "time", "y"], df.columns.tolist()
    # 31 units (1 treated + 30 donors) x 40 time periods
    assert df["unit"].nunique() == 31
    assert df["time"].nunique() == 40
    assert len(df) == 31 * 40
    # Treated id is 0; donors are 1..30
    assert truth["treated_id"] == 0
    assert set(df["unit"].unique()) == set(range(31))
    # Treatment effect added only to treated unit's post-period
    treated_post = df.query("unit == 0 and time >= 30")["y"].to_numpy()
    treated_pre = df.query("unit == 0 and time < 30")["y"].to_numpy()
    # Re-simulate with tau=0 and same seed to get the counterfactual
    df0, _ = simulate_panel(J=30, T=40, T0=30, r=2, sigma=0.5, tau=0.0, seed=42)
    cf_post = df0.query("unit == 0 and time >= 30")["y"].to_numpy()
    cf_pre = df0.query("unit == 0 and time < 30")["y"].to_numpy()
    np.testing.assert_allclose(treated_pre, cf_pre)  # pre-period unchanged by tau
    np.testing.assert_allclose(treated_post - cf_post, 1.0)  # post shifted by tau
    # Determinism
    df_a, _ = simulate_panel(seed=7)
    df_b, _ = simulate_panel(seed=7)
    pd.testing.assert_frame_equal(df_a, df_b)
    # Inside-hull: treated lambda is convex combo of donor lambdas
    lam = truth["lambda"]  # shape (31, r)
    # Solve W in simplex such that lam[1:] @ W ≈ lam[0]
    W = cp.Variable(30, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(lam[1:].T @ W - lam[0])),
               [cp.sum(W) == 1]).solve()
    assert W.value is not None
    np.testing.assert_allclose(lam[1:].T @ W.value, lam[0], atol=1e-6)
    print("simulate_panel OK")

_validate_simulate_panel()
```

- [ ] **Step 2: Sync and run; expect NameError**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: Execution fails with `NameError: name 'simulate_panel' is not defined` in the validation cell.

- [ ] **Step 3: Implement `simulate_panel`**

Insert above the validation cell, still under section 2:
```python
# %%
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
    rng_local = np.random.default_rng(seed)
    n_units = J + 1  # unit 0 = treated, 1..J = donors

    # Donor loadings: standard normal in r-dim factor space
    donor_lambda = rng_local.standard_normal((J, r))
    if inside_hull:
        # Treated loading = random convex combination of donors
        w = rng_local.dirichlet(np.ones(J))
        treated_lambda = w @ donor_lambda
    else:
        # Push treated outside the hull along a random direction
        direction = rng_local.standard_normal(r)
        direction /= np.linalg.norm(direction)
        treated_lambda = donor_lambda.mean(axis=0) + 3.0 * direction

    lam = np.vstack([treated_lambda[None, :], donor_lambda])  # (n_units, r)
    mu = rng_local.standard_normal(n_units)                   # unit fixed effects
    F = rng_local.standard_normal((T, r))                     # factor time series
    eps = rng_local.standard_normal((n_units, T)) * sigma     # idiosyncratic noise

    # y[i, t] = mu[i] + lam[i] @ F[t] + eps[i, t]
    y = mu[:, None] + lam @ F.T + eps

    # Add treatment effect to treated unit's post-period
    y[0, T0:] += tau

    # Long format
    units = np.repeat(np.arange(n_units), T)
    times = np.tile(np.arange(T), n_units)
    df = pd.DataFrame({"unit": units, "time": times, "y": y.reshape(-1)})

    truth = {
        "tau": tau, "mu": mu, "lambda": lam, "F": F,
        "treated_id": 0, "T0": T0,
    }
    return df, truth
```

- [ ] **Step 4: Sync and run; expect pass**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: No errors; final cell prints `simulate_panel OK`.

- [ ] **Step 5: Commit**

```bash
git add "40 - Synth Control Inference.py" "40 - Synth Control Inference.ipynb"
git commit -m "feat(synth-ctrl): factor-model DGP simulate_panel + validation"
```

---

## Task 4: SC estimator (`fit_sc`)

**Files:**
- Modify: `40 - Synth Control Inference.py` (add cells under "## 3. SC estimator")

- [ ] **Step 1: Add validation cell first (failing test)**

Append under section 3:
```python
# %% [markdown]
# ### Validation: fit_sc

# %%
def _validate_fit_sc():
    df, truth = simulate_panel(J=30, T=40, T0=30, sigma=0.0, tau=0.0, seed=11)
    # Noiseless inside-hull case: SC should perfectly recover the treated trajectory pre-period
    out = fit_sc(df, treated_id=0, T0=30)
    assert set(out.keys()) == {"gap", "W", "yhat"}
    assert isinstance(out["gap"], pd.Series)
    assert isinstance(out["W"], pd.Series)
    assert isinstance(out["yhat"], pd.Series)
    # gap and yhat are length T
    assert len(out["gap"]) == 40
    assert len(out["yhat"]) == 40
    # W is length J = 30, on the simplex
    assert len(out["W"]) == 30
    assert (out["W"] >= -1e-8).all()
    np.testing.assert_allclose(out["W"].sum(), 1.0, atol=1e-6)
    # Donor index is 1..30
    assert list(out["W"].index) == list(range(1, 31))
    # Noiseless: pre-period gap should be ~0
    pre_gap = out["gap"].iloc[:30]
    assert np.abs(pre_gap).max() < 1e-4, f"pre-period gap too large: {np.abs(pre_gap).max()}"
    # With tau=0, post-period gap should also be ~0
    post_gap = out["gap"].iloc[30:]
    assert np.abs(post_gap).max() < 1e-3, f"post-period gap too large: {np.abs(post_gap).max()}"

    # With noise and tau=2, post-period gap should center around 2
    df2, _ = simulate_panel(J=30, T=40, T0=30, sigma=0.3, tau=2.0, seed=11)
    out2 = fit_sc(df2, treated_id=0, T0=30)
    assert 1.0 < out2["gap"].iloc[30:].mean() < 3.0
    print("fit_sc OK")

_validate_fit_sc()
```

- [ ] **Step 2: Sync and run; expect NameError**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: `NameError: name 'fit_sc' is not defined`.

- [ ] **Step 3: Implement `fit_sc`**

Insert above the validation cell, under section 3:
```python
# %%
def fit_sc(df, treated_id, T0):
    """
    Fit Abadie SC matching on lagged outcomes only (no extra predictors).
    Solve simplex-constrained QP for donor weights W via cvxpy/OSQP.

    Returns dict with keys:
      gap:  pd.Series indexed by time, length T,  y_treated - y_synth
      W:    pd.Series indexed by donor_id, weights on simplex
      yhat: pd.Series indexed by time, the synthetic counterfactual
    """
    # Pivot to wide: rows = time, columns = unit
    wide = df.pivot(index="time", columns="unit", values="y").sort_index()
    times = wide.index.to_numpy()
    donor_ids = [u for u in wide.columns if u != treated_id]
    y_treated = wide[treated_id].to_numpy()       # shape (T,)
    Y_donors = wide[donor_ids].to_numpy()         # shape (T, J)
    J = Y_donors.shape[1]

    # Pre-period blocks
    pre_mask = times < T0
    y1_pre = y_treated[pre_mask]
    Y0_pre = Y_donors[pre_mask, :]

    # Solve simplex-constrained least squares for W
    W_var = cp.Variable(J, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(y1_pre - Y0_pre @ W_var))
    constraints = [cp.sum(W_var) == 1]
    cp.Problem(objective, constraints).solve(solver=cp.OSQP)
    W_val = np.clip(W_var.value, 0.0, None)
    W_val = W_val / W_val.sum()  # numerical renormalization

    yhat_full = Y_donors @ W_val
    gap_full = y_treated - yhat_full

    return {
        "gap":  pd.Series(gap_full,  index=times, name="gap"),
        "yhat": pd.Series(yhat_full, index=times, name="yhat"),
        "W":    pd.Series(W_val,     index=donor_ids, name="W"),
    }
```

- [ ] **Step 4: Sync and run; expect pass**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: `fit_sc OK`.

- [ ] **Step 5: Commit**

```bash
git add "40 - Synth Control Inference.py" "40 - Synth Control Inference.ipynb"
git commit -m "feat(synth-ctrl): cvxpy-based fit_sc estimator + validation"
```

---

## Task 5: Phase I — placebo distribution + RMSPE-ratio + p-value

**Files:**
- Modify: `40 - Synth Control Inference.py` (add cells under "## 4. Phase I — Abadie p-values")

- [ ] **Step 1: Add validation cell first (failing test)**

Append under section 4:
```python
# %% [markdown]
# ### Validation: placebo + rmspe_ratio + abadie_pvalue

# %%
def _validate_phase1():
    # rmspe_ratio: known input
    gap = pd.Series(np.array([0.1, -0.1, 0.1, -0.1, 2.0, 2.0]),
                    index=np.arange(6))
    r = rmspe_ratio(gap, T0=4)
    expected = np.sqrt((4.0 + 4.0) / 2) / np.sqrt((0.04 * 4) / 4)  # = 2.0 / 0.1 = 20.0
    np.testing.assert_allclose(r, expected, rtol=1e-10)

    # abadie_pvalue: floor at 1/(J+1)
    ratios = pd.Series([5.0, 1.0, 1.0, 1.0], index=[0, 1, 2, 3])
    p = abadie_pvalue(ratios, treated_idx=0)
    assert p == 1 / 4, f"expected 0.25, got {p}"
    # Tie at the top: treated tied with one placebo -> count includes both
    ratios_tie = pd.Series([2.0, 2.0, 1.0, 1.0], index=[0, 1, 2, 3])
    p_tie = abadie_pvalue(ratios_tie, treated_idx=0)
    assert p_tie == 2 / 4, f"expected 0.5, got {p_tie}"

    # placebo_distribution: shape and column set
    df, _ = simulate_panel(J=10, T=20, T0=15, sigma=0.3, tau=1.0, seed=3)
    gaps = placebo_distribution(df, T0=15)
    assert gaps.shape == (20, 11)
    assert set(gaps.columns) == set(range(11))

    # End-to-end on a panel with strong tau: treated should be in top decile
    df2, _ = simulate_panel(J=30, T=40, T0=30, sigma=0.3, tau=2.0, seed=5)
    gaps2 = placebo_distribution(df2, T0=30)
    ratios2 = gaps2.apply(lambda g: rmspe_ratio(g, T0=30))
    p2 = abadie_pvalue(ratios2, treated_idx=0)
    assert p2 <= 3 / 31, f"strong-effect p-value too large: {p2}"
    print("Phase I OK")

_validate_phase1()
```

- [ ] **Step 2: Sync and run; expect NameError**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: NameError on `rmspe_ratio` (or whichever function is referenced first).

- [ ] **Step 3: Implement the three functions**

Insert above the validation cell, under section 4:
```python
# %%
def rmspe_ratio(gap, T0):
    """Post/pre RMSPE ratio for one gap series.

    r = sqrt(mean(gap[t>=T0]^2)) / sqrt(mean(gap[t<T0]^2))
    """
    pre = gap.iloc[:T0].to_numpy()
    post = gap.iloc[T0:].to_numpy()
    pre_rmspe = float(np.sqrt(np.mean(pre ** 2)))
    post_rmspe = float(np.sqrt(np.mean(post ** 2)))
    if pre_rmspe == 0.0:
        return np.inf
    return post_rmspe / pre_rmspe


def placebo_distribution(df, T0):
    """For each unit j in {0, 1, ..., J}, refit SC with j as the treated unit
    and return the gap series. Returns DataFrame indexed by time, one column
    per "treated" identity.
    """
    units = sorted(df["unit"].unique())
    gaps = {}
    for j in units:
        out = fit_sc(df, treated_id=j, T0=T0)
        gaps[j] = out["gap"]
    return pd.DataFrame(gaps).sort_index()


def abadie_pvalue(ratios, treated_idx):
    """Exact one-sided rank p-value:
        p = #{j in 0..J : r_j >= r_treated} / (J + 1)
    Numerator includes the treated unit itself, so p >= 1/(J+1) always.
    Ties at the treated value are counted (conservative).
    """
    r_t = ratios.loc[treated_idx]
    n = len(ratios)
    return float((ratios >= r_t).sum()) / n
```

- [ ] **Step 4: Sync and run; expect pass**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: `Phase I OK`.

- [ ] **Step 5: Commit**

```bash
git add "40 - Synth Control Inference.py" "40 - Synth Control Inference.ipynb"
git commit -m "feat(synth-ctrl): Phase I placebo distribution, RMSPE ratio, p-value"
```

---

## Task 6: Phase I — worked example + diagnostic plot

**Files:**
- Modify: `40 - Synth Control Inference.py` (add cells under section 4, after the validation cell)

- [ ] **Step 1: Add the worked-example cell**

Append under section 4:
```python
# %% [markdown]
# ### Worked example: one panel at tau=1
#
# We simulate one panel with a true treatment effect of tau=1, fit synthetic
# control to every unit (treated + 30 placebos), compute the post/pre RMSPE
# ratio for each, and report the Abadie rank p-value.

# %%
WORKED_PANEL_SEED = 42
df_worked, truth_worked = simulate_panel(
    J=30, T=40, T0=30, r=2, sigma=0.5, tau=1.0, seed=WORKED_PANEL_SEED,
)

gaps_worked = placebo_distribution(df_worked, T0=30)
ratios_worked = gaps_worked.apply(lambda g: rmspe_ratio(g, T0=30))
treated_id_worked = truth_worked["treated_id"]
tau_hat = float(gaps_worked[treated_id_worked].iloc[30:].mean())
r_treated = float(ratios_worked.loc[treated_id_worked])
p_worked = abadie_pvalue(ratios_worked, treated_idx=treated_id_worked)

print(f"Estimated tau (mean post-period gap): {tau_hat:+.3f}")
print(f"Treated unit RMSPE ratio:              {r_treated:.3f}")
print(f"Abadie one-sided p-value:              {p_worked:.4f}  (floor = {1/31:.4f})")
```

- [ ] **Step 2: Add the diagnostic plot cell**

Append:
```python
# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: gap series, treated bold, placebos grey
ax = axes[0]
for col in gaps_worked.columns:
    if col == treated_id_worked:
        continue
    ax.plot(gaps_worked.index, gaps_worked[col], color="lightgrey", lw=0.8)
ax.plot(gaps_worked.index, gaps_worked[treated_id_worked],
        color="black", lw=2, label="Treated")
ax.axvline(30, color="red", ls="--", lw=1, label="T0")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("time")
ax.set_ylabel("gap (y - y_synth)")
ax.set_title("Treated vs. placebo gap series")
ax.legend(loc="upper left")

# Right: histogram of r_j, treated marked
ax = axes[1]
placebo_ratios = ratios_worked.drop(treated_id_worked)
ax.hist(placebo_ratios, bins=15, color="lightgrey", edgecolor="grey")
ax.axvline(r_treated, color="black", lw=2, label=f"Treated r = {r_treated:.2f}")
ax.set_xlabel("post/pre RMSPE ratio")
ax.set_ylabel("count")
ax.set_title("Placebo distribution of RMSPE ratio")
ax.legend()

fig.tight_layout()
plt.show()
```

- [ ] **Step 3: Sync and execute end-to-end**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: No errors. Plot is rendered and embedded in the `.ipynb`. Print output shows tau_hat near 1.0, p around 1/31 to 3/31.

- [ ] **Step 4: Commit**

```bash
git add "40 - Synth Control Inference.py" "40 - Synth Control Inference.ipynb"
git commit -m "feat(synth-ctrl): Phase I worked example + diagnostic plot"
```

---

## Task 7: Phase II — test-inversion CI

**Files:**
- Modify: `40 - Synth Control Inference.py` (add cells under "## 5. Phase II — Confidence intervals")

- [ ] **Step 1: Add validation cell first**

Append under section 5:
```python
# %% [markdown]
# ### Validation: test_inversion_ci

# %%
def _validate_test_inversion():
    df, truth = simulate_panel(J=20, T=30, T0=20, sigma=0.3, tau=1.0, seed=8)
    ci = test_inversion_ci(df, T0=20, treated_id=0, alpha=0.10, n_grid=21)
    # Returns a dict with the expected keys
    assert set(ci.keys()) >= {"ci_lo", "ci_hi", "tau_grid", "pvalues"}
    assert len(ci["tau_grid"]) == 21
    assert len(ci["pvalues"]) == 21
    # Grid is sorted ascending
    assert (np.diff(ci["tau_grid"]) > 0).all()
    # CI brackets the estimated tau (mean post-period gap, not necessarily the true tau)
    out = fit_sc(df, treated_id=0, T0=20)
    tau_hat = float(out["gap"].iloc[20:].mean())
    assert ci["ci_lo"] <= tau_hat <= ci["ci_hi"], \
        f"tau_hat {tau_hat} not in CI [{ci['ci_lo']}, {ci['ci_hi']}]"
    print("test_inversion_ci OK")

_validate_test_inversion()
```

- [ ] **Step 2: Sync and run; expect NameError**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: `NameError: name 'test_inversion_ci' is not defined`.

- [ ] **Step 3: Implement `test_inversion_ci`**

Insert above the validation cell:
```python
# %%
def test_inversion_ci(df, T0, treated_id, alpha=0.05, n_grid=51, grid=None):
    """Test-inversion CI: sweep candidate tau_0, subtract from treated post-period,
    refit + recompute Abadie p-value, keep tau_0 with p > alpha.

    Returns dict with keys: ci_lo, ci_hi, tau_grid, pvalues, tau_hat.
    The tau_grid is sorted ascending.
    """
    # First fit on the unaltered panel to anchor the grid
    out_obs = fit_sc(df, treated_id=treated_id, T0=T0)
    post_gap_obs = out_obs["gap"].iloc[T0:].to_numpy()
    tau_hat = float(post_gap_obs.mean())

    # Use placebo-mean-post-gap SD to scale the grid
    gaps_obs = placebo_distribution(df, T0=T0)
    placebo_means = gaps_obs.iloc[T0:].mean(axis=0).drop(treated_id)
    sigma_placebo = float(placebo_means.std(ddof=1))

    if grid is None:
        half_width = max(4.0 * sigma_placebo, 1e-3)
        grid = np.linspace(tau_hat - half_width, tau_hat + half_width, n_grid)
    else:
        grid = np.asarray(sorted(grid))

    pvalues = np.empty(len(grid))
    for k, tau0 in enumerate(grid):
        df_adj = df.copy()
        mask = (df_adj["unit"] == treated_id) & (df_adj["time"] >= T0)
        df_adj.loc[mask, "y"] = df_adj.loc[mask, "y"] - tau0
        gaps_adj = placebo_distribution(df_adj, T0=T0)
        ratios_adj = gaps_adj.apply(lambda g: rmspe_ratio(g, T0=T0))
        pvalues[k] = abadie_pvalue(ratios_adj, treated_idx=treated_id)

    in_ci = pvalues > alpha
    if not in_ci.any():
        ci_lo = ci_hi = float("nan")
    else:
        ci_lo = float(grid[in_ci].min())
        ci_hi = float(grid[in_ci].max())

    return {
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "tau_grid": grid, "pvalues": pvalues, "tau_hat": tau_hat,
    }
```

- [ ] **Step 4: Sync and run; expect pass**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: `test_inversion_ci OK`.

- [ ] **Step 5: Apply to the worked panel**

Append after the validation cell:
```python
# %% [markdown]
# Apply test-inversion CI to the worked panel:

# %%
ci_inv = test_inversion_ci(df_worked, T0=30, treated_id=0, alpha=0.10, n_grid=51)
print(f"Test-inversion 90% CI: [{ci_inv['ci_lo']:+.3f}, {ci_inv['ci_hi']:+.3f}]  "
      f"(tau_hat = {ci_inv['tau_hat']:+.3f})")
```

- [ ] **Step 6: Sync, execute, commit**

```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
git add "40 - Synth Control Inference.py" "40 - Synth Control Inference.ipynb"
git commit -m "feat(synth-ctrl): Phase II test-inversion CI"
```

---

## Task 8: Phase II — conformal CI (CWZ moving-block)

**Files:**
- Modify: `40 - Synth Control Inference.py` (add cells under section 5, after test-inversion)

- [ ] **Step 1: Add validation cell first**

Append:
```python
# %% [markdown]
# ### Validation: conformal_ci

# %%
def _validate_conformal():
    df, _ = simulate_panel(J=15, T=30, T0=20, sigma=0.3, tau=1.0, seed=9)
    ci = conformal_ci(df, T0=20, treated_id=0, alpha=0.10, n_grid=15, q=1)
    assert set(ci.keys()) >= {"ci_lo", "ci_hi", "tau_grid", "pvalues"}
    assert len(ci["tau_grid"]) == 15
    # P-values are in [pmin, 1] where pmin = 1 / n_windows
    n_windows = 30 - 10 + 1  # T - T1 + 1 = 21
    assert (ci["pvalues"] >= 1 / n_windows - 1e-12).all()
    assert (ci["pvalues"] <= 1.0 + 1e-12).all()
    # CI is non-empty (alpha=0.10 with 21 windows leaves a wide acceptance region)
    assert not np.isnan(ci["ci_lo"]) and not np.isnan(ci["ci_hi"])
    assert ci["ci_lo"] <= ci["ci_hi"]
    print("conformal_ci OK")

_validate_conformal()
```

- [ ] **Step 2: Sync and run; expect NameError**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: `NameError: name 'conformal_ci' is not defined`.

- [ ] **Step 3: Implement `conformal_ci`**

Insert above the validation cell:
```python
# %%
def _moving_block_pvalue(residuals, T0, q=1):
    """Chernozhukov-Wuthrich-Zhu moving-block permutation test.

    Statistic: S_q(window) = (mean over window of |u_t|^q)^(1/q)
    Compare actual post-period window against all length-T1 windows in the residual series.
    P-value = rank of actual statistic among all (T - T1 + 1) windows, including itself.
    """
    T = len(residuals)
    T1 = T - T0  # post-period length
    if T1 <= 0:
        raise ValueError("post-period must be non-empty")
    res = np.asarray(residuals)

    def Sq(window):
        return float(np.mean(np.abs(window) ** q) ** (1.0 / q))

    # Actual post-period window starts at T0
    s_actual = Sq(res[T0:T0 + T1])
    # All admissible windows (including the actual one)
    starts = np.arange(0, T - T1 + 1)
    s_all = np.array([Sq(res[s:s + T1]) for s in starts])
    # Rank including the actual statistic (it's already in s_all at index T0)
    return float((s_all >= s_actual).sum()) / float(len(s_all))


def conformal_ci(df, T0, treated_id, alpha=0.05, n_grid=51, grid=None, q=1):
    """Conformal CI via moving-block permutation of SC residuals (CWZ 2021).

    For each tau_0: subtract tau_0 from treated post-period, fit SC on the
    full T-length adjusted treated series (not just pre-period), compute
    residuals, and compare the post-period window's S_q statistic against
    all moving-block windows of the same length.

    Returns dict with keys: ci_lo, ci_hi, tau_grid, pvalues.
    """
    # Anchor the grid using the test-inversion grid logic for comparability
    out_obs = fit_sc(df, treated_id=treated_id, T0=T0)
    tau_hat = float(out_obs["gap"].iloc[T0:].mean())
    gaps_obs = placebo_distribution(df, T0=T0)
    placebo_means = gaps_obs.iloc[T0:].mean(axis=0).drop(treated_id)
    sigma_placebo = float(placebo_means.std(ddof=1))
    if grid is None:
        half_width = max(4.0 * sigma_placebo, 1e-3)
        grid = np.linspace(tau_hat - half_width, tau_hat + half_width, n_grid)
    else:
        grid = np.asarray(sorted(grid))

    pvalues = np.empty(len(grid))
    for k, tau0 in enumerate(grid):
        df_adj = df.copy()
        mask = (df_adj["unit"] == treated_id) & (df_adj["time"] >= T0)
        df_adj.loc[mask, "y"] = df_adj.loc[mask, "y"] - tau0
        # Fit SC on the FULL adjusted series, not just pre-period
        # We accomplish this by passing T0 = T (so all periods are "pre" for the fit)
        T = df["time"].nunique()
        out_full = fit_sc(df_adj, treated_id=treated_id, T0=T)
        residuals = out_full["gap"].to_numpy()
        pvalues[k] = _moving_block_pvalue(residuals, T0=T0, q=q)

    in_ci = pvalues > alpha
    if not in_ci.any():
        ci_lo = ci_hi = float("nan")
    else:
        ci_lo = float(grid[in_ci].min())
        ci_hi = float(grid[in_ci].max())

    return {
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "tau_grid": grid, "pvalues": pvalues,
    }
```

- [ ] **Step 4: Sync and run; expect pass**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: `conformal_ci OK`.

- [ ] **Step 5: Apply to the worked panel + interpretive note**

Append after the validation cell:
```python
# %% [markdown]
# Apply conformal CI to the worked panel and compare with test-inversion:

# %%
ci_conf = conformal_ci(df_worked, T0=30, treated_id=0, alpha=0.10, n_grid=51, q=1)
print(f"Test-inversion 90% CI: [{ci_inv['ci_lo']:+.3f}, {ci_inv['ci_hi']:+.3f}]")
print(f"Conformal      90% CI: [{ci_conf['ci_lo']:+.3f}, {ci_conf['ci_hi']:+.3f}]")

# %% [markdown]
# The two intervals may legitimately disagree because they rest on different
# exchangeability assumptions (donor-units exchangeable with treated under the null
# vs. residual blocks exchangeable across time), not because one is "wrong."
```

- [ ] **Step 6: Sync, execute, commit**

```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
git add "40 - Synth Control Inference.py" "40 - Synth Control Inference.ipynb"
git commit -m "feat(synth-ctrl): Phase II conformal CI with moving-block permutation"
```

---

## Task 9: Phase III — coverage_sim (smoke run)

**Files:**
- Modify: `40 - Synth Control Inference.py` (add cells under "## 6. Phase III — Coverage simulation")

- [ ] **Step 1: Add validation cell first**

Append under section 6:
```python
# %% [markdown]
# ### Validation: coverage_sim (small smoke run)

# %%
def _validate_coverage_sim():
    out = coverage_sim(R=8, taus=(0.0, 1.0), alpha=0.10, n_jobs=1,
                      sim_kwargs=dict(J=10, T=20, T0=15, sigma=0.3),
                      ci_kwargs=dict(n_grid=11))
    assert isinstance(out, pd.DataFrame)
    expected_cols = {"rep", "tau", "method", "pvalue", "ci_lo", "ci_hi", "covered"}
    assert expected_cols <= set(out.columns), out.columns.tolist()
    assert set(out["method"].unique()) == {"abadie_p", "test_inversion", "conformal"}
    # 8 reps x 2 taus x 3 methods = 48 rows
    assert len(out) == 8 * 2 * 3
    # `covered` is bool/int and well-defined for the CI methods at least
    ci_rows = out[out["method"].isin({"test_inversion", "conformal"})]
    assert ci_rows["covered"].notna().all()
    print("coverage_sim OK")

_validate_coverage_sim()
```

- [ ] **Step 2: Sync and run; expect NameError**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: `NameError: name 'coverage_sim' is not defined`.

- [ ] **Step 3: Implement `coverage_sim`**

Insert above the validation cell:
```python
# %%
def _one_rep(rep_idx, taus, alpha, sim_kwargs, ci_kwargs):
    """Run one MC rep across all taus. Reuses the same factor draws across taus."""
    rows = []
    for tau in taus:
        df, truth = simulate_panel(tau=tau, seed=rep_idx, **sim_kwargs)
        T0 = truth["T0"]
        treated_id = truth["treated_id"]

        # Abadie p-value (one-sided, ratio-based)
        gaps = placebo_distribution(df, T0=T0)
        ratios = gaps.apply(lambda g: rmspe_ratio(g, T0=T0))
        p_abadie = abadie_pvalue(ratios, treated_idx=treated_id)
        rows.append(dict(rep=rep_idx, tau=tau, method="abadie_p",
                         pvalue=p_abadie, ci_lo=np.nan, ci_hi=np.nan,
                         covered=(p_abadie > alpha)))

        # Test-inversion CI
        ci_inv_r = test_inversion_ci(df, T0=T0, treated_id=treated_id,
                                     alpha=alpha, **ci_kwargs)
        rows.append(dict(rep=rep_idx, tau=tau, method="test_inversion",
                         pvalue=np.nan,
                         ci_lo=ci_inv_r["ci_lo"], ci_hi=ci_inv_r["ci_hi"],
                         covered=(ci_inv_r["ci_lo"] <= tau <= ci_inv_r["ci_hi"])))

        # Conformal CI
        ci_conf_r = conformal_ci(df, T0=T0, treated_id=treated_id,
                                 alpha=alpha, **ci_kwargs)
        rows.append(dict(rep=rep_idx, tau=tau, method="conformal",
                         pvalue=np.nan,
                         ci_lo=ci_conf_r["ci_lo"], ci_hi=ci_conf_r["ci_hi"],
                         covered=(ci_conf_r["ci_lo"] <= tau <= ci_conf_r["ci_hi"])))
    return rows


def coverage_sim(R=2000, taus=(0.0, 0.5, 1.0, 2.0), alpha=0.05, n_jobs=-1,
                 sim_kwargs=None, ci_kwargs=None):
    """Monte Carlo coverage simulation. Within each rep, the panel uses
    seed=rep_idx and the same factor draws are reused across taus (only the
    additive shift changes), so rows of the result at different taus are paired.

    Returns long DataFrame with columns:
      rep, tau, method, pvalue, ci_lo, ci_hi, covered
    """
    sim_kwargs = sim_kwargs or dict(J=30, T=40, T0=30, r=2, sigma=0.5)
    ci_kwargs = ci_kwargs or dict(n_grid=51)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_one_rep)(rep, taus, alpha, sim_kwargs, ci_kwargs)
        for rep in range(R)
    )
    flat = [row for rep_rows in results for row in rep_rows]
    return pd.DataFrame(flat)
```

- [ ] **Step 4: Sync and run; expect pass**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace "40 - Synth Control Inference.ipynb"
```
Expected: `coverage_sim OK`. Smoke run takes a few seconds.

- [ ] **Step 5: Commit smoke version**

```bash
git add "40 - Synth Control Inference.py" "40 - Synth Control Inference.ipynb"
git commit -m "feat(synth-ctrl): Phase III coverage_sim (validated on smoke run)"
```

---

## Task 10: Phase III — full run + tables + plots

**Files:**
- Modify: `40 - Synth Control Inference.py` (add cells after the smoke-run validation in section 6)
- Create: `coverage_sim_results.parquet` (cached output of the full run; gitignored)
- Modify: `.gitignore`

- [ ] **Step 1: Gitignore the cached results**

Append to `.gitignore`:
```
coverage_sim_results.parquet
```

- [ ] **Step 2: Add the full-run cell with cache**

Append under section 6, after the smoke validation:
```python
# %% [markdown]
# ### Full coverage simulation
#
# R=2000 reps across tau in {0.0, 0.5, 1.0, 2.0}, parallelized.
# This takes ~5-15 minutes depending on cores. Cached to disk so re-execution
# of the notebook is fast.

# %%
import os

CACHE_PATH = "coverage_sim_results.parquet"

if os.path.exists(CACHE_PATH):
    cov_df = pd.read_parquet(CACHE_PATH)
    print(f"Loaded cached coverage_sim from {CACHE_PATH} (rows={len(cov_df)})")
else:
    cov_df = coverage_sim(R=2000, taus=(0.0, 0.5, 1.0, 2.0), alpha=0.05, n_jobs=-1)
    cov_df.to_parquet(CACHE_PATH)
    print(f"Saved coverage_sim to {CACHE_PATH} (rows={len(cov_df)})")
```

- [ ] **Step 3: Add tables (size, power, coverage)**

Append:
```python
# %% [markdown]
# ### Size and power table (Abadie p-value)

# %%
abadie = cov_df[cov_df["method"] == "abadie_p"].copy()
# `covered` is True when p > alpha, i.e. failed to reject. So rejection rate = 1 - mean(covered).
size_power = abadie.groupby("tau")["covered"].agg(
    n="size",
    accept_rate="mean",
).assign(
    reject_rate=lambda d: 1 - d["accept_rate"],
    se_reject=lambda d: np.sqrt(d["reject_rate"] * (1 - d["reject_rate"]) / d["n"]),
)[["n", "reject_rate", "se_reject"]]
print("Abadie p-value rejection rate by tau (alpha = 0.05):")
print(size_power.round(4))

# %% [markdown]
# ### Coverage table (CI methods)

# %%
cis = cov_df[cov_df["method"].isin({"test_inversion", "conformal"})].copy()
coverage = cis.groupby(["method", "tau"])["covered"].agg(
    n="size",
    coverage="mean",
).assign(
    se=lambda d: np.sqrt(d["coverage"] * (1 - d["coverage"]) / d["n"]),
)[["n", "coverage", "se"]]
print("Empirical coverage by method and tau (target = 0.95):")
print(coverage.round(4))
```

- [ ] **Step 4: Add the coverage plot**

Append:
```python
# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
for method, sub in coverage.reset_index().groupby("method"):
    ax.errorbar(sub["tau"], sub["coverage"], yerr=2 * sub["se"],
                marker="o", capsize=3, label=method)
ax.axhline(0.95, color="black", ls="--", lw=1, label="nominal 0.95")
ax.set_xlabel("true tau")
ax.set_ylabel("empirical coverage")
ax.set_title("Coverage of synth-control CIs (R=2000, alpha=0.05)")
ax.set_ylim(0.80, 1.02)
ax.legend()
fig.tight_layout()
plt.show()
```

- [ ] **Step 5: Sync and execute**

Run:
```bash
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "40 - Synth Control Inference.ipynb"
```
Expected: First run is slow (5-15 min). Final tables and plot rendered.

- [ ] **Step 6: Commit**

```bash
git add "40 - Synth Control Inference.py" "40 - Synth Control Inference.ipynb" .gitignore
git commit -m "feat(synth-ctrl): Phase III full coverage sim, tables, coverage plot"
```

---

## Task 11: Discussion + final end-to-end check

**Files:**
- Modify: `40 - Synth Control Inference.py` (add cells under "## 7. Discussion")

- [ ] **Step 1: Add the discussion markdown cell**

Append under section 7 (replace placeholder bracketed bits with values from your run):
```python
# %% [markdown]
# ## 7. Discussion
#
# **Did the Abadie p-value hit nominal size?** Look at the rejection rate at tau=0
# in the size/power table. With J=30 donors, the smallest possible p-value from
# the rank-based test is 1/31 ≈ 0.032, and the test rejects at alpha=0.05 when
# p ≤ 0.05. Under the null we expect the rejection rate to be close to 1/31
# (the floor) ≈ 0.032, *not* 0.05 — the test is conservative because of grid
# coarseness. Empirically you should see ~3% rejection at tau=0.
#
# **Power.** Rejection rate at tau in {0.5, 1.0, 2.0} should rise sharply.
# At tau=2 (well above the noise scale sigma=0.5), power should be near 1.
#
# **Which CI was tighter?** Compare interval widths in the cached `cov_df` —
# averaging `ci_hi - ci_lo` per method shows whether test-inversion or conformal
# is more efficient under the well-specified inside-hull DGP.
#
# **Where does each method degrade?** Both methods rely on exchangeability
# assumptions that this DGP satisfies (donors share the factor structure with
# the treated unit; residuals from a correctly-specified factor fit are roughly
# stationary). Misspecification scenarios — treated unit *outside* the convex
# hull, heavy-tailed noise, or a structural break in the factors — are listed
# in the spec's "Open follow-ups" and are the natural next experiments.
#
# **One caveat on identifiability.** With J=30 donors and r=2 latent factors,
# the donor weights W are massively under-identified noiselessly: any face of
# the simplex containing the treated unit's loading vector works. Inference
# here doesn't need W to be unique — every test statistic is a function of
# the gap series, which is invariant to the choice of W within the admissible
# set. Don't read the bar plot of W as recovering "the true weights."
```

- [ ] **Step 2: Final end-to-end execution from a clean state**

Delete the cache and re-execute to confirm the notebook runs top-to-bottom from scratch:
```bash
rm -f coverage_sim_results.parquet
uv run --python .venv-synthcontrol jupytext --to ipynb "40 - Synth Control Inference.py"
uv run --python .venv-synthcontrol jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "40 - Synth Control Inference.ipynb"
```
Expected: All cells execute without error. Tables and plots render.

- [ ] **Step 3: Commit**

```bash
git add "40 - Synth Control Inference.py" "40 - Synth Control Inference.ipynb"
git commit -m "feat(synth-ctrl): discussion section + final end-to-end check"
```

---

## Self-review checklist

- **Spec coverage:** Setup → Task 1+2; DGP → Task 3; estimator → Task 4; Phase I (placebo, RMSPE, p-value, worked example, plot) → Tasks 5–6; Phase II (test-inversion, conformal) → Tasks 7–8; Phase III (coverage_sim, tables, plot) → Tasks 9–10; Discussion → Task 11. No spec section is unaddressed.
- **No placeholders:** All cells contain actual code, every step shows commands and expected output, no "TBD" / "fill in details."
- **Type / signature consistency:** `simulate_panel` returns `(df, truth)` everywhere; `fit_sc` returns dict with keys `gap`/`W`/`yhat` everywhere; `placebo_distribution` returns wide DataFrame consumed by `apply(rmspe_ratio)`; `test_inversion_ci` and `conformal_ci` both return dicts with `ci_lo`/`ci_hi`/`tau_grid`/`pvalues`; `coverage_sim` returns long DataFrame consumed in Task 10.
- **TDD adapted to notebook:** Each function task is "validation cell first → run → fail → implement → run → pass" — the notebook idiom of inline assertion cells.

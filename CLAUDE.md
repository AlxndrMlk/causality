# Project conventions

This is a notebook-driven causal-inference repo. Each topic lives in its own numbered notebook at the root (e.g. `34 - Decision-Making Errors Pt 1.ipynb`), with a dedicated environment.

## Environments

Historically envs are conda `.yml` files (`econml-dowhy-py38.yml`, `causality_pysensemakr.yml`, etc.). For **new** notebook projects, use **uv** with a per-project venv and a registered Jupyter kernel:

```bash
uv venv .venv-<project>
uv pip install <deps> ipykernel
uv run python -m ipykernel install --user \
    --name <project> --display-name "Python (<project>)"
```

`--user` registers the kernel in the user-level Jupyter directory so it shows up in any Jupyter session (Lab, Notebook, VS Code) without needing `uv run` to launch — Jupyter spawns the venv's python directly via the absolute path baked into `kernel.json`. The first markdown cell of the notebook should name the kernel so the user knows which to select.

## Specs

Design docs for non-trivial notebook projects live in `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`.

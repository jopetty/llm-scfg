# AGENTS.md

This file records project-specific guidance for future coding agents working in
`llm-scfg`.

## Environment and tooling

- Use `uv` for Python commands and dependency management.
  - Prefer `uv run python ...`
  - Prefer `uv sync` / `uv sync --group cluster`
  - Prefer `uv add` for new dependencies
  - Never use `pip(3)`, `conda`, or `python(3)` directly
- Run checks with:
  - `uv run prek --all-files`
- `prek.toml` is the authoritative hook config.
  - Do not try to run `pre-commit` directly; instead run `prek`
  - `uv run prek ...` will be run before commits
- `pyproject.toml` defines:
  - base dependencies for generation, analysis, notebooks
  - `cluster` dependency group for `vllm` and `wandb`
  - `dev` dependency group for `prek` and `ty`

## Notebook and analysis conventions

- Prefer using `notebooks/aesthetics.py` rather than ad hoc plot styling.
  - When figures are saved, they should always be saved with the `aes.save_figure(...)` wrapper; never call `plt.savefig(...)` directly.
- Paper-facing figures should be saved to `paper/figures/`, not
  `notebooks/figures/`; this should always be set as a `FIGURES_DIR: Path` constant in the notebook, and passed to `aes.save_figure(FIGURES_DIR / "figure_name")`. Do not include a filetype extension in the save call; `aes.save_figure` will handle this.
- The paper repo is a submodule; do not assume it should be modified unless the
  task clearly calls for it.
- Notebook caches under `notebooks/cache/` are generated artifacts and should
  not be committed.
- If you ever make or edit a figure that has multiple rows/columns, always use a
  `gridspec` to manage the layout instead of manually calling subplots.

## Git / commit conventions learned from this project

- There is a github actions workflow preventing direct pushes to `main`; if you are currently on `main`, first create a new branch for your work and then switch to it; if already on a new branch, do not create another branch off this unless specifically directed to.
- Group commits into sensible chunks when possible.
- Before pushing, run `uv run prek --all-files`.
- Do not commit generated notebook cache artifacts.

## Practical defaults for future agents

- If a task mentions notebooks, inspect `notebooks/aesthetics.py` early.
- If a task touches cluster execution, inspect `README.md`, `pyproject.toml`,
  `prek.toml`, and the scripts under `scripts/`.
- If a task touches error analysis, update both:
  - `notebooks/error_analysis.py`
  - downstream notebook/dashboard consumers when needed
- If a task touches paper figures, check whether the save destination and legend
  placement conventions in the paper-facing notebooks need to be preserved.

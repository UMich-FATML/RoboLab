# RoboLab — Agent Map

RoboLab is a benchmark of language-guided robotic manipulation tasks built on
NVIDIA IsaacSim/IsaacLab. A **task** (scene + instruction + termination) is
combined with robot/observation/action/simulation configs at *environment
registration time* — there is no per-task code to read; there is a library of
composable pieces.

## Layout

| Path | What it is |
|------|------------|
| `robolab/` | the package: `core/environments` (registration & run loop), `core/world` (state queries), `core/task` (conditionals & events), `core/events` (recorders, resets), `robots/` (per-robot obs/action/scene pieces), `core/utils/isaaclab_compat.py` (see rule 1) |
| `policies/` | policy backends (`pi0_family/`, …) — each pairs a client with a runner |
| `tests/` | pytest suite; `test_run_empty` and `test_frame_contract` boot IsaacSim and need a GPU |
| `docs/` | concept docs — `docs/README.md` is the index; `docs/design.md` records *why* |
| `docs/plans/` | execution plans (active + completed) and the tech-debt list |
| `examples/`, `dashboard/`, `analysis/` | supporting tooling |

## Start here

- **Concepts & terminology**: [docs/README.md](docs/README.md) — read before touching task/env code.
- **Frames & quaternion contract**: [docs/frames.md](docs/frames.md) — RoboLab speaks `(w, x, y, z)` everywhere (one exception: recorded camera extrinsics are `quat_w_ros`). Every pose-consuming change must respect this.
- **Decisions & rationale**: [docs/design.md](docs/design.md) — why the compat layer exists, why stacks are uv conflict groups, etc.
- **Running & verifying on GPUs**: [docs/running.md](docs/running.md) — per-stack venvs, `--junit-xml`, camera-less smoke runs, Slurm patterns.

## Toolchain

uv only (never pip/conda against project deps; `uv.lock` is local-only). Three
mutually exclusive IsaacSim stacks install via extras — see the README install
section for the exact per-venv commands.

```bash
uv run ruff check .                     # lint (isort-style import order; config in pyproject.toml)
uv run python scripts/check_docs.py     # docs/map lint: links resolve, no orphan docs
uv run --extra test pytest tests/       # full suite (GPU + IsaacSim)
```

Enable the git hooks once per clone (docs lint + ruff run on commit):

```bash
git config core.hooksPath .githooks
```

## Hard rules

1. **All IsaacLab-version handling lives in
   `robolab/core/utils/isaaclab_compat.py`.** No other file may branch on the
   installed isaaclab version or convert quaternion order inline — call the
   compat helpers at the boundary instead. The 2.x path of every helper is a
   no-op guard; keep it that way (non-regression on 2.x is a merge condition).
2. **The quaternion contract is `(w, x, y, z)`** ([docs/frames.md](docs/frames.md)).
   Authored configs, recorded data, and public APIs speak wxyz; the installed
   stack's order is an implementation detail hidden by rule 1.
3. **`openpi-client` is not in any extra** (its `numpy<2` pin conflicts with
   isaaclab 3.0's `numpy>=2`). `tests/test_volo_interop.py` skips without it;
   the Docker eval image installs it `--no-deps`. Do not add it back to
   `pyproject.toml` without checking that pin.
4. **Batch scripts under the local-only slurm/ tree are never committed**
   (git-excluded; it does not exist in a fresh clone — never link to it or
   backtick it as a path). Cluster submission knowledge belongs in
   [docs/running.md](docs/running.md).

## When you finish

- Verify per [docs/running.md](docs/running.md) before claiming done — the
  full suite on the affected stack, `--junit-xml` always (isaaclab 3.0
  hard-exits before pytest's failure summary).
- Commit at feature scale with a why-focused body; see `git log` for the house
  style. See [docs/plans/README.md](docs/plans/README.md) for when a change
  also needs a plan or a design-doc entry.

# Design — Core Beliefs & Decisions

Why RoboLab works the way it does. Each entry records the decision, the
reason, and what was rejected. Add an entry when a decision is made that a
future contributor would otherwise re-litigate or silently break. House rule
([AGENTS.md](../AGENTS.md) rule 1) enforces the most load-bearing one.

## Core beliefs

1. **Composition over per-task code.** A benchmark of N tasks × M robots × K
   observation/action variants must not be N×M×K code paths. Tasks bind a
   scene to an instruction and termination criteria; everything else is
   combined at registration time from libraries ([README](../README.md),
   [environment registration](environment_registration.md)). Anything that
   grows a per-task branch is a design smell.

2. **One version boundary.** IsaacSim/IsaacLab releases break APIs at every
   major (quaternion order, sensor classes, render settings — see the decision
   log). The blast radius is contained by converting *only* in
   `robolab/core/utils/isaaclab_compat.py`; the rest of the codebase speaks
   RoboLab's stable contract (wxyz quaternions, `docs/frames.md`). Rejected:
   scattering `if isaaclab_version()` checks — they rot, and 2.x non-regression
   becomes unauditable.

3. **Non-regression is a merge condition.** Every stack the repo claims to
   support (see the README extras table) must pass the full suite. A port that
   greens 3.0 while breaking 2.x is not done. This is why every compat helper's
   2.x path is a no-op guard, and why verification runs both stacks
   ([running](running.md)).

4. **The map is checked, not trusted.** Docs rot; pointers lie. The docs lint
   (`scripts/check_docs.py`, run by the pre-commit hook) fails on dead links
   and orphan docs so the index stays true mechanically. The same goes for
   rules: a hard rule a grep can judge gets a check (`scripts/check_rules.py`)
   whose message says how to fix it, and a lint nobody runs is not a lint —
   the hook fails rather than skips when ruff is unavailable.

## Decision log

| Date | Decision | Why / rejected alternative |
|------|----------|---------------------------|
| 2026-10 | **Generated tasks render instruction and termination from one symbolic goal; LLMs only author templates** | RoboLab's LLM task generation measured 57–60 % instruction/termination alignment on color and counting because the two were written separately. Both are now projections of the same goal facts, proven solvable by backward chaining; per-scene generation is seeded and deterministic (manifest = canonical, USDA = build product). Rejected: per-scene LLM generation (unaligned, unreproducible, unprovable). See [batch generation](scene_gen_pipeline.md). |
| 2026-10 | **Scene generation reasons from mesh geometry (runtime hull reader); physics settle confirms, not constructs** | Bounding-box placement mis-rests bottom-origin assets by up to 6.5 cm, drops `place-in` objects from above the mouth and treats rims as supports, so settle *built* the scene and facts became unknowable until then. Heightfield drops over mesh-derived stable poses author scenes at rest (matches settled benchmark z to mm). `hull_check.read_prim_mesh_points` is the single geometry definition shared with the runtime predicates. Rejected: bbox heuristics + drop-and-settle; a CPU physics engine (different contact model, new dependency — fallback if settle rejections stay high). |
| 2026-10 | **RoboLab keeps wxyz everywhere; isaaclab 3.0's xyzw is converted at the boundary** | Recorded data and authored configs must stay stable across stack upgrades — flipping the public contract would invalidate every downstream checkpoint and analysis. Rejected: adopting xyzw in 3.0 venvs (two conventions in flight, silent corruption of recorded datasets). See [frames](frames.md). |
| 2026-10 | **`isaaclab 3.0` contact/articulation isinstance checks resolve base classes via the compat layer** | 3.0 turns `ContactSensor`/`Articulation` into physics-backend factories: constructed assets subclass `Base*`, not the facade, so `isinstance(x, ContactSensor)` matches *nothing* — sensors silently disappear from the world dict. Rejected: duck-typing on attributes (weaker, still version-coupled). |
| 2026-10 | **`as_torch()` wraps data buffers before isaaclab's scripted math** | 3.0 data buffers are warp `ProxyArray`s; indexing works via a deprecation bridge but torch-scripted functions (`quat_inv` inside `subtract_frame_transforms`) reject them outright. Rejected: relying on the bridge (deprecated, and the failure only appears at runtime on GPU). |
| 2026-10 | **Render presets vendored in `robolab/core/environments/_render_presets.py`** | isaaclab 3.0 stops shipping `rendering_modes/*.kit` presets and removes `SimulationCfg.render`; the `--rendering-type` CLI survives by writing the preset's carb settings directly before `gym.make`. Values are the 2.3.2.post1 presets (provenance in the module docstring). Rejected: dropping the flag (user-visible break) or pinning carb settings per camera (loses the global CLI). |
| 2026-10 | **Three stacks as uv conflict-group extras, one lockfile** | IsaacSim 5.0/5.1/6.1 wheels are mutually exclusive; conflict groups let one lockfile serve all three, selected per venv. Rejected: three branches or three lockfiles (drift, cherry-pick tax). `uv.lock` stays local-only like the venvs. |
| 2026-10 | **`openpi-client` out of the extras; Docker installs it `--no-deps`** | Its `numpy<2` pin cannot co-resolve with isaaclab 3.0's `numpy>=2`; uv overrides are global so a scoped override is impossible. `test_volo_interop` uses `importorskip` so the suite stays collectable everywhere. Rejected: dropping 3.0 support or forking openpi-client (out of scope for a benchmark repo). |
| 2026-10 | **`ROBOLAB_TEST_DISABLE_CAMERAS=1` smoke mode** | Some driver generations can only boot IsaacSim headless-without-cameras on parts of a cluster; the two env-constructing tests skip and the rest still exercises registration, configs, and the compat layer. Rejected: skipping the whole suite there (loses 214 checks for free). |
| 2026-09 | **`docs/frames.md` contract: three frames, root pose recorded per step** | Env-local and robot-root coincide only for Franka-family embodiments; floor-standing robots broke that assumption. Consumers must read the recorded `robot_root_pose` channel instead of assuming it. |

## Related

- [Plans & tech debt](plans/README.md) — active work and known debt.
- [Running & verifying](running.md) — how the beliefs above are checked in practice.

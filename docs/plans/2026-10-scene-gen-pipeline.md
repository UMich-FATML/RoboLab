# Batch Scene/Task Generation Pipeline

**Status: completed** (2026-10, branch `scene-gen-pipeline`, commits
`e30a031..23956bf`)

## Goal

First runnable slice of a plan to scale RoboLab from 120 hand-authored tasks
to ~10k generated ones: a seeded, LLM-free pipeline from committed templates
to settled scenes and solvable tasks, smoke-tested on 20 tasks. Design and
usage: [batch generation](../scene_gen_pipeline.md); decisions:
[design](../design.md) (two 2026-10 entries).

## Success criteria

1. `scripts/scenegen.py generate --num-tasks 20 --seed 0` → 20 scenes + tasks +
   manifest, deterministic per seed, ≥5 templates and ≥5 goal families.
2. `validate-cpu` passes for all 20 (prims, contact/termination objects,
   USDA ↔ manifest poses, plan replay); every intended `in` holds under the
   runtime hull model before any GPU.
3. GPU: settle keeps ≥90 % of objects within 2 cm; settle-rejected scenes are
   topped up; `verify_task_valid` passes for all 20; empty episodes run for 3.
4. `tests_nosim/` green on CPU; `tests/` unaffected on the isaac60 stack.

## Decisions taken while implementing

- **Heightfield drop instead of bbox + PhysicalSolver for z/containment.**
  The bbox recipe was off by 6.5 cm for bottom-origin assets; the drop matches
  physics-settled benchmark scenes to millimetres. `SpatialSolver` is still
  used (unmodified) for table-level xy.
- **Tip margin, not just COM-over-patch.** A remote standing on end is
  statically stable but tips in physics; poses need margin/COM-height ≥ 0.2.
- **`in` means COM below the rim.** A banana bridging a mug rim passed a
  "bottom below rim" test.
- **Containment may "settle".** Round fruit on a curved bowl floor next to a
  sibling rolls but stays in; such placements are flagged and checked by the
  relation, not by displacement.
- **Referring expressions gate goals.** Name → color → side of table; if
  nothing disambiguates, the goal is rejected.
- **Assets with non-identity root xforms are excluded** (the scene's authored
  xform replaces root ops).

## Progress

- [x] Shared mesh reader extracted (`hull_check.read_prim_mesh_points`), verified
      identical to the old loop
- [x] Geometry (stable poses, heightmaps, cavity, support), catalog + ontology
- [x] Logic layer (rules, backward chainer, plan replay, DAG)
- [x] Templates (6), sampler, grounded placer, static check, USDA writer
- [x] Quest (8 families), renderer, pipeline + CLI, GPU validation script
- [x] 20 tasks generated; `validate-cpu` 0 failures; `tests_nosim` 23 passed
- [x] GPU settle + `check-settled`, `verify_task_valid`, empty episodes
- [x] Full suite on isaac60 and camera-less isaac51 (`--junit-xml`)

## Outcome

- **Generation:** 20 live tasks from 21 attempts (1 `hosted_drop` rejection),
  6 templates, 7 goal families (`sort_two_bins` renders in tests but was not
  drawn in this seed), 13 simple / 6 moderate / 1 complex.
- **Grounding held:** physics settle moved all 102 objects < 2 cm (mean
  0.10 cm on the table, 0.41 cm in containers, 0.47 cm on supports, 0.11 cm
  lying; worst 1.89 cm) — zero settle rejections. Settle confirmed the
  authored scenes instead of constructing them.
- **Simulator checks (isaac60, camera node):** `verify_task_valid` 20/20;
  empty episodes 3/3, none terminating without a policy.
- **Non-regression** (the mesh-reader refactor touches a runtime path):
  isaac60 full suite 227/227; isaac51 camera-less 216 passed, 3 by-design skips.
- **Found by the GPU loop and fixed:** 3 assets carry their rigid body on a
  child prim (env build fails, settle moves the child) — now excluded by the
  catalog; `check-settled` refuses files physics never touched; orientation is
  only checked for objects with discrete rest poses.
- **Screenshots (follow-up):** `core/utils/render_utils.py` imported
  `omni.isaac.core` / `isaacsim.core.api`, which are deprecated-only on 5.1 /
  6.1. It now uses only version-stable APIs (pxr, omni.usd, timeline,
  replicator), so it needs no compat-layer branch; thumbnails of the 20 scenes
  are in `assets/scenes/generated/_images/`. Verified on isaac60 only — 5.x
  camera rendering cannot run on this cluster (tech-debt item 3).
- **Not done:** reference plans are not executed (tech-debt item 5).

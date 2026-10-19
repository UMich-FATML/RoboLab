# Batch Scene & Task Generation

`scripts/scenegen.py` generates RoboLab scenes *and* tasks in bulk, without
LLM calls, from committed templates. It is the batch counterpart of the
interactive [`/robolab-scenegen`](../skills/robolab-scenegen/) and
[`/robolab-taskgen`](../skills/robolab-taskgen/) skills. Code:
[`robolab/scene_gen/`](../robolab/scene_gen/); rationale:
[design](design.md) (decision log, 2026-10).

## Quick start

```bash
export OMNI_KIT_ACCEPT_EULA=Y         # 6.x stack; pxr is needed, a GPU is not
PY=.venv-60/bin/python                # any stack venv with pxr works for the CPU steps

$PY scripts/scenegen.py build-geometry                # mesh geometry per asset (cached, ~10 s)
$PY scripts/scenegen.py generate --num-tasks 20 --seed 0
$PY scripts/scenegen.py validate-cpu                  # static checks, no simulator
# GPU node (see running.md for batch patterns):
$PY assets/scenes/_utils/settle_scenes.py --scene assets/scenes/generated --replace \
    --screenshot --screenshot-dir assets/scenes/generated/_images
$PY scripts/scenegen.py check-settled                 # relation-aware settle check; rejects broken scenes
$PY scripts/scenegen.py generate --num-tasks 20 --seed 0   # tops up rejected scenes (then settle those)
$PY scripts/scenegen_validate_sim.py --episodes 3 --junit-xml junit_scenegen.xml
OMNI_KIT_ACCEPT_EULA=Y .venv-60/bin/pytest tests_nosim -q   # CPU unit tests
```

Outputs: `assets/scenes/generated/gen_<template>_<seed>.usda`,
`robolab/tasks/generated/gen_<template>_<seed>.py`, and
`assets/scenes/generated/_manifest.jsonl` + `README.md`. The manifest (seed,
objects, poses, facts, goal, plan) is canonical; USDA and task files are
regenerable build products. Generated tasks live outside
`DEFAULT_TASK_SUBFOLDERS`, so they are not part of the benchmark or of
`tests/test_tasks_valid.py`; register them with
`auto_register_droid_envs(task_dirs=["generated"])`.

## Stages

```
template YAML ─seeded sampler─▶ objects per slot ─SpatialSolver─▶ table xy/yaw
   ─heightfield drop─▶ resting poses (table, in, on) ─static_check─▶ facts ─▶ USDA
facts ─goal family─▶ goal facts ─backward chaining─▶ plan + subtask DAG
goal facts ─render twice─▶ termination + subtasks (code)  and  default/vague/specific (language)
```

| Stage | Module | Rejects when |
|---|---|---|
| sample | `templates.py` | not enough catalog objects for a slot's count |
| table layout | `scene_builder._spatial_solve` (existing `SpatialSolver`, unmodified) | solver can't separate footprints |
| table drop | `placer.HeightField.drop` | object would rest on another object, or tip |
| hosted drop | `placer.place_relative` | no stable spot in/on the host (single layer) |
| static check | `scene_builder.static_check` | an intended `in` fails the runtime hull model |
| quest | `quest.py` + `logic/` | no goal family yields an unambiguous, plannable goal |
| settle (GPU) | `pipeline.check_settled` | an intended fact broke during physics |

## Grounding: why authored scenes are already at rest

Bounding boxes cannot author stable scenes: prim origins sit at the bottom for
some datasets and at the center for others (a 6.5 cm error for `wooden_bowl`
under the `z = dims/2` recipe), a bowl's top is a hollow rim, and objects rest
only in discrete poses. `geometry.py` reads each asset's mesh with the same
reader the runtime predicates use (`hull_check.read_prim_mesh_points`) and
derives:

- **stable poses** — hull face clusters whose contact patch contains the COM,
  each with a tip margin (support margin ÷ COM height); tippy poses
  (a remote on its end) are never authored on a surface;
- **top/bottom heightmaps** per pose (4 mm grid);
- **cavity** (open-top containers: floor, rim, depth) and **support** (flat,
  well-filled plateau: plates, box tops — rims are rejected by fill ratio);
- **graspable**: thinnest dimension ≤ 7.5 cm (Robotiq 2F-85), or an open
  container ≤ 20 cm across (grasped by the wall).

Placement is one operation: lower the object's bottom heightmap onto the
scene's top heightmap. That gives the exact rest height on the table, a bowl's
curved floor or a plate's dished interior (validated: `wooden_bowl` z=0.0029
and `bowl` z=0.0305 vs 0.0028/0.0303 in physics-settled benchmark scenes),
what it rests on, and quasi-static stability. `in` additionally requires the
object's COM below the rim (a banana bridging a mug rim sags into it but is
not *in* it). Affordances (`fits_in`, `stackable_on`) are trial placements, so
the chainer can only use relations the geometry allows.

Physics settle then *confirms*: table/`on` objects must move < 2 cm and keep
their upright-ness; `in` objects must still be inside under the runtime hull
model (round fruit may roll on a curved floor; such spots are flagged
`settles` in the manifest).

## Tasks: goal facts, chaining, rendering

Goal families (each one competency axis, one termination):

| family | goal facts | termination | attributes |
|---|---|---|---|
| `put_in` | `in(o, c)` | `object_in_container` | semantics / color / spatial (referring expression) |
| `put_on` | `on(o, s)` (plate/box) | `object_on_top` | as above |
| `take_out` | `on(o, table)` from `in(o, c)` | `object_outside_of` | as above |
| `stack` | `on(b2, b1), on(b3, b2)` | `stacked(order="bottom_to_top")` | stacking, color |
| `spatial` | `left_of/right_of/in_front_of/behind(o, r)` | `object_left_of` & co. (robot frame) | spatial |
| `count_k_of_n` | `in(o_i, c)` for K of N same-category | `object_in_container(logical="choose", K)` (exactly K) | counting |
| `reorient` | `upright(o)` for lying objects | `object_upright` | reorientation |
| `sort_two_bins` | two whole categories into two bins | `object_groups_in_containers` | sorting |

`logic.Chainer` works backwards from the goal facts (goal regression over the
rules in `logic/rules.py`); the plan is replayed forward to prove solvability
and grouped into stages of independent objects (`plan_to_dag`) for subtasks.
Instructions use the shortest unambiguous referring expression (display name;
else color; else side of the table) — if none exists the goal is rejected, so
`default`, `specific` and the termination always name the same objects.

## Templates

Committed YAML under [`assets/scene_templates/`](../assets/scene_templates/)
(schema in its README). Templates name classes, relations, regions, pose
(`upright`/`lying`/`any`) and count ranges; the semantic ontology
(`assets/objects/scenegen_ontology.yaml`: display name, category, color)
decides which catalog objects may be sampled. Physical roles are never
hand-set. An LLM may author more templates in the same schema; it never touches
per-scene generation.

## Limits (follow-ups)

- Logical solvability only: reference plans are not yet executed with a
  scripted IK controller, so "demonstrably solvable" is future work.
- One layer inside containers; no piles.
- Assets with non-identity root xforms (6 in the catalog) or missing files (3)
  are excluded by `catalog.py` — the authored prim xform would replace a root
  transform. None of them is in the ontology today.
- Vague instructions are template text, not paraphrases.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch driver: generate, validate on CPU, and check settled scenes.

Every attempt — accepted or rejected — appends one JSON line to the manifest,
so acceptance rates per stage/template/difficulty are always reconstructible.
The manifest (predicates + seed) is canonical; USDA and task files are
regenerable build products.
"""

from __future__ import annotations

import ast
import collections
import json
import math
import random
import time
import zlib
from pathlib import Path

import numpy as np

from robolab.constants import PACKAGE_DIR, SCENE_DIR, TASK_DIR
from robolab.scene_gen.catalog import Catalog
from robolab.scene_gen.geometry import quat_to_matrix
from robolab.scene_gen.logic import RULES, Prop, State, validate
from robolab.scene_gen.placer import TABLE_ID
from robolab.scene_gen.quest import FAMILIES, NoGoal, sample_goal
from robolab.scene_gen.render_task import class_name, instructions, render_task_source, termination_objects
from robolab.scene_gen.scene_builder import Rejected, build_scene, layout_signature, runtime_in, write_usda
from robolab.scene_gen.templates import load_templates

DEFAULT_SCENE_OUT = Path(SCENE_DIR) / "generated"
DEFAULT_TASK_OUT = Path(TASK_DIR) / "generated"
MANIFEST = "_manifest.jsonl"
SETTLE_MAX_DISPLACEMENT = 0.02
UPRIGHT_TILT_DEG = 15.0


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(Path(PACKAGE_DIR)))


def read_manifest(scene_out: Path) -> list[dict]:
    path = Path(scene_out) / MANIFEST
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_rows(rows: list[dict]) -> dict[str, dict]:
    """Last row per scene name (status updates are appended, never rewritten)."""
    out = {}
    for r in rows:
        if r.get("scene_name"):
            out[r["scene_name"]] = r
    return out


def _append(scene_out: Path, row: dict) -> None:
    with open(Path(scene_out) / MANIFEST, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def generate(num_tasks: int, seed: int = 0, scene_out: Path = DEFAULT_SCENE_OUT, task_out: Path = DEFAULT_TASK_OUT,
             max_attempts: int | None = None, templates=None, log=print) -> dict:
    """Generate until ``num_tasks`` scenes are live (accepted and not settle-rejected).

    Re-running continues where the manifest left off (top-up after settle
    rejections): seeds already attempted are skipped.
    """
    scene_out, task_out = Path(scene_out), Path(task_out)
    scene_out.mkdir(parents=True, exist_ok=True)
    task_out.mkdir(parents=True, exist_ok=True)
    templates = templates or load_templates()
    catalog = Catalog()
    _ = catalog.objects
    rows = read_manifest(scene_out)
    latest = latest_rows(rows)
    live = {k for k, r in latest.items() if r["status"] == "accepted"}
    attempted = {r["seed"] for r in rows}
    signatures = {r["signature"] for r in latest.values() if r.get("signature")}
    fam_count = collections.Counter(r["goal"]["family"] for r in latest.values() if r["status"] == "accepted")
    tmpl_count = collections.Counter(r["template"] for r in latest.values() if r["status"] == "accepted")
    max_attempts = max_attempts or 20 * num_tasks
    stats = collections.Counter()
    i, t0 = 0, time.time()
    while len(live) < num_tasks and i < max_attempts:
        s = seed * 10000 + i
        i += 1
        if s in attempted:
            continue
        # Template coverage first: the least-used template goes next.
        t = min(templates, key=lambda tt: (tmpl_count[tt.name], (zlib.crc32(tt.name.encode()) + s) % 97))
        row = {"seed": s, "template": t.name, "time": round(time.time() - t0, 2)}
        try:
            scene = build_scene(t, catalog, s)
        except Rejected as e:
            stats[e.stage] += 1
            _append(scene_out, {**row, "status": "rejected", "stage": e.stage, "reason": e.reason})
            # A template that keeps failing must not starve the others.
            tmpl_count[t.name] += 0.25
            continue
        sig = layout_signature(scene)
        if sig in signatures:
            stats["duplicate"] += 1
            _append(scene_out, {**row, "status": "rejected", "stage": "dedupe", "reason": "duplicate layout"})
            continue
        rng = random.Random(s)
        fams = sorted(t.goals, key=lambda f: (fam_count[f], rng.random()))  # goal-family coverage first
        try:
            goal = sample_goal(scene, rng, families=fams)
        except NoGoal as e:
            stats["quest"] += 1
            _append(scene_out, {**row, "status": "rejected", "stage": "quest", "reason": str(e)})
            continue
        usda = scene_out / f"{scene.name}.usda"
        task_py = task_out / f"{scene.name}.py"
        write_usda(scene, usda)
        src = render_task_source(scene, goal)
        task_py.write_text(src)
        score, label = goal.difficulty()
        _append(scene_out, {
            **row, "status": "accepted", "stage": "generated", "scene_name": scene.name,
            "scene": _rel(usda), "task": _rel(task_py), "task_class": class_name(scene), "signature": sig,
            "objects": {p: {"asset": o.name, "usd_path": o.usd_path, "slot": next(k for k, v in scene.slots.items() if p in v),
                            **scene.placements[p].to_json()} for p, o in scene.objects.items()},
            "relations": {p: list(v) for p, v in scene.relations.items()},
            "initial_facts": scene.state.to_strings(),
            "goal": {"family": goal.family, "facts": [str(f) for f in goal.facts], "targets": goal.targets,
                     "hosts": goal.hosts, "attributes": sorted(set(goal.attributes)),
                     "params": {k: v for k, v in goal.params.items() if k != "groups"}},
            "plan": goal.plan.to_strings(), "dag": goal.dag, "difficulty": {"score": score, "label": label},
            "instruction": instructions(scene, goal),
        })
        live.add(scene.name)
        signatures.add(sig)
        fam_count[goal.family] += 1
        tmpl_count[t.name] += 1
        stats["accepted"] += 1
        log(f"[{len(live)}/{num_tasks}] {scene.name}: {goal.family} ({label}) — {instructions(scene, goal)['default']}")
    write_summary(scene_out)
    return {"live": len(live), "attempts": i, **stats}


# ---------------------------------------------------------------------------
# validate-cpu
# ---------------------------------------------------------------------------

def parse_action(text: str):
    p = Prop.parse(text)
    rule = next(r for r in RULES if r.name == p.name)
    return rule.ground({v: a for (v, _), a in zip(rule.params, p.args)})


def read_usda_poses(path: Path) -> dict[str, tuple[np.ndarray, tuple]]:
    """Authored pose (translate, orient wxyz) of every direct child of /world.

    Payloads stay unloaded, so prims are typeless (their Xform type comes from
    the payload) and UsdGeom.Xformable would silently return identity; read the
    xformOp attributes directly. /world itself is identity in base_empty.usda.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    out = {}
    for prim in stage.GetPrimAtPath("/world").GetAllChildren():  # default predicate hides unloaded prims
        t = prim.GetAttribute("xformOp:translate").Get()
        q = prim.GetAttribute("xformOp:orient").Get()
        if t is None:
            continue
        quat = (1.0, 0.0, 0.0, 0.0) if q is None else (q.GetReal(), *q.GetImaginary())
        out[prim.GetName()] = (np.array(t, dtype=np.float64), tuple(float(c) for c in quat))
    return out


def validate_cpu(scene_out: Path = DEFAULT_SCENE_OUT, log=print) -> list[str]:
    """Static checks on every live scene/task. Returns a list of failures (empty = pass)."""
    from robolab.core.task.task_utils import find_task_files

    failures = []
    latest = latest_rows(read_manifest(scene_out))
    live = [r for r in latest.values() if r["status"] == "accepted"]
    bench_classes = set()
    for f in find_task_files(str(Path(TASK_DIR) / "benchmark")):
        bench_classes |= {n.name for n in ast.parse(Path(f).read_text()).body if isinstance(n, ast.ClassDef)}
    seen = set()
    for r in live:
        name = r["scene_name"]
        usda, task = Path(PACKAGE_DIR) / r["scene"], Path(PACKAGE_DIR) / r["task"]
        if not usda.exists() or not task.exists():
            failures.append(f"{name}: missing {'usda' if not usda.exists() else 'task'} file")
            continue
        src = task.read_text()
        tree = ast.parse(src)
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith("Task")]
        if classes != [r["task_class"]] or r["task_class"] in seen or r["task_class"] in bench_classes:
            failures.append(f"{name}: task class {classes} missing or not unique")
        seen.add(r["task_class"])
        contact = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                       if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "contact_object_list")
        poses = read_usda_poses(usda)
        if missing := set(contact) - set(poses):
            failures.append(f"{name}: contact objects not prims in scene: {sorted(missing)}")
        if missing := termination_objects(src) - set(contact):
            failures.append(f"{name}: termination objects not in contact list: {sorted(missing)}")
        for prim, o in r["objects"].items():
            pos, q = poses[prim]
            if np.abs(pos - [o["x"], o["y"], o["z"]]).max() > 1e-4 or _quat_angle(q, o["quat_wxyz"]) > 1e-2:  # quatf float32 + 6-dp rounding ~2e-3 rad
                failures.append(f"{name}: USDA pose of {prim} differs from manifest")
        state = State(Prop.parse(f) for f in r["initial_facts"])
        if not validate(state, [parse_action(a) for a in r["plan"]], [Prop.parse(f) for f in r["goal"]["facts"]]):
            failures.append(f"{name}: reference plan does not replay to the goal")
    log(f"validate-cpu: {len(live)} live scenes, {len(failures)} failures")
    return failures


def _quat_angle(a, b) -> float:
    d = abs(sum(x * y for x, y in zip(a, b)))
    return 2 * math.acos(min(1.0, d))


# ---------------------------------------------------------------------------
# check-settled
# ---------------------------------------------------------------------------

def _tilt_deg(q) -> float:
    z = quat_to_matrix(q)[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, z[2]))))


def check_settled(scene_out: Path = DEFAULT_SCENE_OUT, max_displacement: float = SETTLE_MAX_DISPLACEMENT,
                  apply: bool = True, log=print) -> dict:
    """Compare settled USDAs to authored poses; reject scenes whose intended facts broke.

    Table / ``on`` objects must move < ``max_displacement`` and keep their
    upright-ness; ``in`` objects must still be inside their container under the
    runtime hull model (they may legitimately roll on a curved floor). Rejected
    scenes get a ``settle_rejected`` manifest row and their task file removed.
    """
    catalog = Catalog()
    latest = latest_rows(read_manifest(scene_out))
    live = [r for r in latest.values() if r["status"] == "accepted"]
    disp = collections.defaultdict(list)
    report = {"checked": 0, "rejected": [], "unsettled": [], "per_relation": {}}
    for r in live:
        usda = Path(PACKAGE_DIR) / r["scene"]
        settled = read_usda_poses(usda)
        moved = max(float(np.abs(settled[k][0] - [o["x"], o["y"], o["z"]]).max()) for k, o in r["objects"].items())
        if moved < 1e-6:  # bit-identical to the authored poses: physics never ran on this file
            report["unsettled"].append(r["scene_name"])
            continue
        problems = []
        placed = {}
        for prim, o in r["objects"].items():
            pos, q = settled[prim]
            d = float(np.linalg.norm(pos - [o["x"], o["y"], o["z"]]))
            rel = r["relations"].get(prim, ["table", TABLE_ID])[0]
            kind = "lying" if (rel == "table" and not o["upright"]) else rel
            if rel == "in" and o.get("settles"):
                kind = "in_settles"
            disp[kind].append(d)
            co = catalog.objects[o["asset"]]
            placed[prim] = (_SettledPose(*pos, q), q, co)
            if pos[2] < -0.05:
                problems.append(f"{prim} fell off the table (z={pos[2]:.3f})")
            elif rel in ("table", "on") and d > max_displacement:
                problems.append(f"{prim} moved {d * 100:.1f} cm")
            # Orientation is a fact only for objects with discrete rest poses (mugs, boxes);
            # a round orange "tilting" on the table changes nothing.
            discrete = any(not co.geom.poses[i].upright for i in co.geom.table_poses())
            if discrete and o["upright"] and _tilt_deg(q) > UPRIGHT_TILT_DEG:
                problems.append(f"{prim} tipped over ({_tilt_deg(q):.0f} deg)")
            if discrete and not o["upright"] and rel == "table" and _tilt_deg(q) < UPRIGHT_TILT_DEG:
                problems.append(f"{prim} rolled upright")
        for prim, (rel, host) in r["relations"].items():
            if rel != "in":
                continue
            (p, _, po), (h, _, ho) = placed[prim], placed[host]
            if not runtime_in(p, po, h, ho):
                problems.append(f"in({prim}, {host}) no longer holds")
        report["checked"] += 1
        if problems:
            report["rejected"].append({"scene": r["scene_name"], "problems": problems})
            if apply:
                task = Path(PACKAGE_DIR) / r["task"]
                task.unlink(missing_ok=True)
                _append(scene_out, {**r, "status": "settle_rejected", "stage": "settle", "reason": "; ".join(problems)})
        elif apply:
            _append(scene_out, {**r, "status": "accepted", "stage": "settled"})
    for kind, v in sorted(disp.items()):
        a = np.array(v)
        report["per_relation"][kind] = {"n": len(v), "mean_cm": round(a.mean() * 100, 2),
                                        "p90_cm": round(float(np.percentile(a, 90)) * 100, 2),
                                        "max_cm": round(a.max() * 100, 2),
                                        "frac_lt_2cm": round(float((a < 0.02).mean()), 3)}
    all_d = np.concatenate([np.array(v) for v in disp.values()]) if disp else np.zeros(0)
    report["all_objects"] = {"n": int(all_d.size), "frac_lt_2cm": round(float((all_d < 0.02).mean()), 3) if all_d.size else None}
    log(json.dumps(report, indent=2))
    write_summary(scene_out)
    return report


class _SettledPose:
    """Duck-typed pose for runtime_in: position + quaternion straight from the USDA."""

    def __init__(self, x, y, z, q):
        self.x, self.y, self.z, self.quat_wxyz = float(x), float(y), float(z), q


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def write_summary(scene_out: Path = DEFAULT_SCENE_OUT) -> None:
    rows = read_manifest(scene_out)
    latest = latest_rows(rows)
    live = sorted((r for r in latest.values() if r["status"] == "accepted"), key=lambda r: r["scene_name"])
    stages = collections.Counter(r["stage"] for r in rows if r["status"] == "rejected")
    settle_rej = [r for r in latest.values() if r["status"] == "settle_rejected"]
    lines = [
        "# Generated scenes",
        "",
        "Built by `scripts/scenegen.py` (see `docs/scene_gen_pipeline.md`). Do not edit by hand:",
        "`_manifest.jsonl` (seed + predicates + facts + plan) is canonical; USDA and task files are",
        "regenerable build products. Thumbnails (after settle `--screenshot`): `_images/<scene>.png`.",
        "",
        f"- live scenes: **{len(live)}**; attempts: {len({r['seed'] for r in rows})}; "
        f"pre-GPU rejections by stage: {dict(stages)}; settle rejections: {len(settle_rej)}",
        f"- goal families: {dict(collections.Counter(r['goal']['family'] for r in live))}",
        f"- difficulty: {dict(collections.Counter(r['difficulty']['label'] for r in live))}",
        f"- templates: {dict(collections.Counter(r['template'] for r in live))}",
        "",
        "| scene | family | difficulty | settled | instruction (default) |",
        "| :-- | :-- | :-- | :-- | :-- |",
    ]
    for r in live:
        lines.append(f"| {r['scene_name']} | {r['goal']['family']} | {r['difficulty']['label']} "
                     f"| {'yes' if r['stage'] == 'settled' else 'no'} | {r['instruction']['default']} |")
    (Path(scene_out) / "README.md").write_text("\n".join(lines) + "\n")


__all__ = ["generate", "validate_cpu", "check_settled", "write_summary", "FAMILIES"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Template + seed -> grounded scene: placements, symbolic state, USDA.

Pipeline per scene (any failure raises ``Rejected`` with a reason; there is no
repair loop — the caller overgenerates):

1. sample objects for the template's slots (``templates.sample_objects``);
2. table-level xy/yaw with RoboLab's existing ``SpatialSolver`` (unmodified),
   fed each object's *posed footprint* radius;
3. rest every table object on the heightfield (``placer.HeightField.drop``) —
   it must rest only on the table and be stable;
4. rest ``in:``/``on:`` objects on their hosts (``placer.place_relative``);
5. ``static_check``: re-verify every intended ``in`` with the runtime's own
   convex-hull containment model (``hull_check``), before any GPU;
6. lift placements into facts (``logic``), write the USDA.
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robolab.constants import PACKAGE_DIR, SCENE_DIR
from robolab.scene_gen.catalog import Catalog, CatalogObject
from robolab.scene_gen.geometry import CELL, quat_to_matrix
from robolab.scene_gen.logic import P, State
from robolab.scene_gen.placer import (
    TABLE_BOUNDS,
    TABLE_ID,
    HeightField,
    Placement,
    footprint_center_radius,
    place_relative,
)
from robolab.scene_gen.templates import REGIONS, SampledObject, SamplingError, Template, sample_objects

BASE_SCENE = Path(SCENE_DIR) / "base_empty.usda"


class Rejected(RuntimeError):
    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason


@dataclass
class SceneInstance:
    template: Template
    seed: int
    objects: dict[str, CatalogObject]  # prim -> catalog object
    slots: dict[str, list[str]]  # slot -> prims
    placements: dict[str, Placement]
    relations: dict[str, tuple[str, str]]  # prim -> (relation, host) for in:/on: objects
    state: State = None
    fits: dict = field(default_factory=dict)  # (obj, container) -> bool
    stackable: dict = field(default_factory=dict)  # (obj, surface) -> bool

    @property
    def name(self) -> str:
        return f"gen_{self.template.name}_{self.seed:04d}"


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _spatial_solve(table_objs: list[SampledObject], rng: random.Random) -> dict[str, tuple[float, float, float]]:
    """Footprint-center xy + yaw for table objects via the existing SpatialSolver."""
    from robolab.scene_gen.llm_scene_gen import ObjectState, PlaceOnBasePredicate, SpatialSolver

    states, dims, centers = {}, {}, {}
    for so in table_objs:
        cx, cy, r = footprint_center_radius(so.obj.geom, so.pose_index)
        x0, x1, y0, y1 = REGIONS[so.slot.region]
        yaw = rng.uniform(0, 360)
        st = ObjectState(name=so.prim)
        st.predicates.append(PlaceOnBasePredicate(so.prim, x=rng.uniform(x0, x1), y=rng.uniform(y0, y1), yaw=yaw))
        states[so.prim] = st
        dims[so.prim] = (2 * r, 2 * r, so.obj.geom.poses[so.pose_index].extents[2])
        centers[so.prim] = (cx, cy)
    solver = SpatialSolver(table_bounds=TABLE_BOUNDS, collision_margin=0.02)
    with contextlib.redirect_stdout(io.StringIO()):
        ok, msg = solver.solve(states, dims)
    if not ok:
        raise Rejected("spatial_solver", msg)
    out = {}
    for prim, st in states.items():
        cx, cy = centers[prim]
        c, s = math.cos(math.radians(st.yaw)), math.sin(math.radians(st.yaw))
        # The solver positions the footprint center; convert to the prim origin.
        out[prim] = (st.x - (c * cx - s * cy), st.y - (s * cx + c * cy), st.yaw)
    return out


def build_scene(template: Template, catalog: Catalog, seed: int) -> SceneInstance:
    rng = random.Random(seed)
    random.seed(seed)  # SpatialSolver uses the global RNG
    np.random.seed(seed % (2**32))
    try:
        sampled = sample_objects(template, catalog, rng)
    except SamplingError as e:
        raise Rejected("sample", str(e)) from None

    scene = SceneInstance(template, seed, {s.prim: s.obj for s in sampled}, {}, {}, {})
    for s in sampled:
        scene.slots.setdefault(s.slot.slot, []).append(s.prim)

    hf = HeightField()
    table_objs = [s for s in sampled if s.slot.relation == "table"]
    poses = _spatial_solve(table_objs, rng)
    # Large footprints first: a small object placed first never props up a big one.
    for so in sorted(table_objs, key=lambda s: -max(s.obj.geom.poses[s.pose_index].extents[:2])):
        x, y, yaw = poses[so.prim]
        d = hf.drop(so.obj.geom, so.pose_index, x, y, yaw)
        if d is None:
            raise Rejected("table_drop", f"{so.prim} off the grid")
        if d.supports != {TABLE_ID}:
            raise Rejected("table_drop", f"{so.prim} rests on {sorted(d.supports)}")
        if not d.stable:
            raise Rejected("table_drop", f"{so.prim} unstable on table")
        p = Placement(so.prim, so.obj.geom, so.pose_index, x, y, d.z, yaw, d.supports)
        hf.stamp(p)
        scene.placements[so.prim] = p

    for so in (s for s in sampled if s.slot.relation in ("in", "on")):
        hosts = scene.slots.get(so.slot.host, [])
        if not hosts:
            raise Rejected("hosted_drop", f"slot {so.slot.slot}: host slot {so.slot.host} is empty")
        host = scene.placements[rng.choice(hosts)]
        if so.slot.relation == "in" and not host.geom.cavity:
            raise Rejected("hosted_drop", f"{host.name} is not a container")
        siblings = {k for k, (rel, h) in scene.relations.items() if h == host.name}
        p = place_relative(hf, so.prim, so.obj.geom, 0, host, so.slot.relation, rng, siblings=siblings)
        if p is None:
            raise Rejected("hosted_drop", f"no stable spot for {so.prim} {so.slot.relation} {host.name}")
        hf.stamp(p)
        scene.placements[so.prim] = p
        scene.relations[so.prim] = (so.slot.relation, host.name)

    static_check(scene)
    scene.state = derive_state(scene, catalog)
    return scene


# ---------------------------------------------------------------------------
# Affordances (trial placements on an empty table) and symbolic state
# ---------------------------------------------------------------------------

def _trial(obj: CatalogObject, host: CatalogObject, relation: str) -> bool:
    hf = HeightField()
    d = hf.drop(host.geom, 0, 0.55, 0.0, 0.0)
    hp = Placement(host.name, host.geom, 0, 0.55, 0.0, d.z, 0.0, d.supports)
    hf.stamp(hp)
    p = place_relative(hf, obj.name, obj.geom, 0, hp, relation, random.Random(0), yaws=[0.0, 45.0, 90.0])
    return p is not None and (relation == "on" or not p.settles)


_AFFORDANCE_CACHE: dict = {}


def affordance(obj: CatalogObject, host: CatalogObject, relation: str) -> bool:
    """fits_in / stackable_on from mesh geometry: can obj rest stably in/on host alone?"""
    key = (obj.name, host.name, relation)
    if key not in _AFFORDANCE_CACHE:
        _AFFORDANCE_CACHE[key] = _trial(obj, host, relation)
    return _AFFORDANCE_CACHE[key]


def derive_state(scene: SceneInstance, catalog: Catalog) -> State:
    facts = [P("free_gripper"), P("table", TABLE_ID), P("surface", TABLE_ID)]
    occupied = set()
    for prim, p in scene.placements.items():
        rel = scene.relations.get(prim)
        if rel is None:
            facts.append(P("on", prim, TABLE_ID))
        else:
            facts.append(P(rel[0], prim, rel[1]))
            occupied.add(rel[1])
    for prim, o in scene.objects.items():
        p = scene.placements[prim]
        if o.graspable:
            facts += [P("movable", prim), P("graspable", prim)]
        if o.is_container:
            facts.append(P("container", prim))
        if o.is_support:
            facts.append(P("surface", prim))
        if prim not in occupied:
            facts.append(P("clear", prim))
        if p.upright:
            facts.append(P("upright", prim))
        facts += [P("color", prim, o.color), P("category", prim, o.category), P("size", prim, o.size)]
    for prim, o in scene.objects.items():
        if not o.graspable:
            continue
        for host_prim, h in scene.objects.items():
            if host_prim == prim:
                continue
            if h.is_container and affordance(o, h, "in"):
                facts.append(P("fits_in", prim, host_prim))
            if h.is_support and affordance(o, h, "on"):
                facts.append(P("stackable_on", prim, host_prim))
    facts.append(P("clear", TABLE_ID))
    return State(facts)


# ---------------------------------------------------------------------------
# Static check with the runtime's containment model
# ---------------------------------------------------------------------------

_HULLS: dict = {}


def asset_hull(o: CatalogObject):
    """``hull_check.LocalHull`` of the asset, exactly as the runtime predicates build it."""
    if o.name not in _HULLS:
        from pxr import Usd

        from robolab.core.task.hull_check import build_local_hull, read_prim_mesh_points

        stage = Usd.Stage.Open(str(Path(PACKAGE_DIR) / o.usd_path))
        _HULLS[o.name] = build_local_hull(read_prim_mesh_points(stage.GetDefaultPrim()))
    return _HULLS[o.name]


def runtime_in(obj: Placement, oo: CatalogObject, host: Placement, ho: CatalogObject) -> bool:
    """Mirror of ``predicate_logic.in_opentop_container`` evaluated on authored poses."""
    import torch

    from robolab.core.task.hull_check import point_in_hull

    oh, ch = asset_hull(oo), asset_hull(ho)
    c_obj = oh.centroid.numpy().astype(np.float64)
    w = quat_to_matrix(obj.quat_wxyz) @ c_obj + np.array([obj.x, obj.y, obj.z])
    local = quat_to_matrix(host.quat_wxyz).T @ (w - np.array([host.x, host.y, host.z]))
    return bool(point_in_hull(torch.tensor(local, dtype=torch.float32), ch.planes_open_top))


def static_check(scene: SceneInstance) -> None:
    for prim, (rel, host) in scene.relations.items():
        p, h = scene.placements[prim], scene.placements[host]
        if rel == "in" and not runtime_in(p, scene.objects[prim], h, scene.objects[host]):
            raise Rejected("static_check", f"in({prim}, {host}) false under the runtime hull model")
        if rel == "on" and p.bottom_z() < h.top_z() - 0.05 - CELL:
            raise Rejected("static_check", f"on({prim}, {host}) rests far below the host top")
    for prim, p in scene.placements.items():
        if p.supports == {TABLE_ID}:
            x0, x1, y0, y1 = TABLE_BOUNDS
            if not (x0 <= p.x <= x1 and y0 <= p.y <= y1):
                raise Rejected("static_check", f"{prim} outside table bounds")


# ---------------------------------------------------------------------------
# USDA
# ---------------------------------------------------------------------------

def layout_signature(scene: SceneInstance) -> str:
    """Dedupe key: multiset of (category, relation, host category, 5 cm cell)."""
    items = []
    for prim, p in scene.placements.items():
        rel, host = scene.relations.get(prim, ("on", TABLE_ID))
        hcat = scene.objects[host].category if host in scene.objects else host
        items.append(f"{scene.objects[prim].category}|{rel}|{hcat}|{int(p.x / 0.05)},{int(p.y / 0.05)}")
    return ";".join(sorted(items))


def write_usda(scene: SceneInstance, out_path: Path) -> None:
    out_path = Path(out_path)
    depth = len(out_path.resolve().parent.relative_to(Path(PACKAGE_DIR) / "assets").parts)
    base = BASE_SCENE.read_text()
    if depth > 1:  # base_empty's payloads are relative to assets/scenes/
        prefix = "../" * (depth - 1)
        base = re.sub(r"@(\.\./)", lambda m: "@" + prefix + m.group(1), base)
    up = "../" * depth
    blocks = []
    for prim in sorted(scene.placements, key=lambda k: (k in scene.relations, k)):
        p = scene.placements[prim]
        payload = up + scene.objects[prim].usd_path.removeprefix("assets/")
        w, x, y, z = p.quat_wxyz  # USD quatf literal order is (real, i, j, k) == (w, x, y, z)
        blocks.append(
            f'    def "{prim}" (\n'
            f"        prepend payload = @{payload}@\n"
            f"    )\n"
            f"    {{\n"
            f"        quatf xformOp:orient = ({w:.7f}, {x:.7f}, {y:.7f}, {z:.7f})\n"
            f"        float3 xformOp:scale = (1, 1, 1)\n"
            f"        double3 xformOp:translate = ({p.x:.6f}, {p.y:.6f}, {p.z:.6f})\n"
            f'        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]\n'
            f"    }}\n"
        )
    idx = base.rstrip().rfind("}")
    text = base[:idx].rstrip("\n") + "\n\n" + "\n".join(blocks) + "}\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)

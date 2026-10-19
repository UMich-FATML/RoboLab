# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene templates (committed YAML) and the seeded object sampler.

A template names object classes, relations and count ranges
(``assets/scene_templates/README.md``). ``sample_objects`` draws concrete
catalog objects for every slot; geometry roles in ``require`` (container /
support / graspable) come from the mesh, so a template can only ask for
physically meaningful roles.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from robolab.constants import ASSET_DIR
from robolab.scene_gen.catalog import Catalog, CatalogObject

TEMPLATE_DIR = Path(ASSET_DIR) / "scene_templates"
REGIONS = {
    # (x_min, x_max, y_min, y_max) in the robot/world frame: +x away from the robot, +y left.
    "center": (0.45, 0.65, -0.12, 0.12),
    "left": (0.35, 0.70, 0.12, 0.38),
    "right": (0.35, 0.70, -0.38, -0.12),
    "front": (0.62, 0.78, -0.30, 0.30),
    "back": (0.32, 0.45, -0.30, 0.30),
    "any": (0.32, 0.78, -0.38, 0.38),
}
GOAL_FAMILIES = ("put_in", "put_on", "take_out", "stack", "spatial", "count_k_of_n", "reorient", "sort_two_bins")


@dataclass
class Slot:
    slot: str
    select: dict
    count: tuple
    placement: str = "table"
    region: str = "any"
    pose: str = "upright"
    require: dict = field(default_factory=dict)

    @property
    def host(self) -> str | None:
        return self.placement.split(":", 1)[1] if ":" in self.placement else None

    @property
    def relation(self) -> str:
        return self.placement.split(":", 1)[0]


@dataclass
class Template:
    name: str
    theme: str
    slots: list
    goals: list

    @staticmethod
    def load(path: str | Path) -> "Template":
        d = yaml.safe_load(open(path))
        slots = [Slot(slot=s["slot"], select=s.get("select", {}), count=tuple(s.get("count", (1, 1))),
                      placement=s.get("placement", "table"), region=s.get("region", "any"),
                      pose=s.get("pose", "upright"), require=s.get("require", {})) for s in d["slots"]]
        t = Template(d["name"], d.get("theme", ""), slots, list(d.get("goals", [])))
        t.validate()
        return t

    def validate(self) -> None:
        names = [s.slot for s in self.slots]
        if len(set(names)) != len(names):
            raise ValueError(f"template {self.name}: duplicate slot names")
        for s in self.slots:
            if s.relation not in ("table", "in", "on"):
                raise ValueError(f"template {self.name}: slot {s.slot} has bad placement {s.placement!r}")
            if s.host and s.host not in names[: names.index(s.slot)]:
                raise ValueError(f"template {self.name}: slot {s.slot} references unknown/later slot {s.host}")
            if s.region not in REGIONS:
                raise ValueError(f"template {self.name}: slot {s.slot} has unknown region {s.region}")
            if s.pose not in ("upright", "lying", "any"):
                raise ValueError(f"template {self.name}: slot {s.slot} has bad pose {s.pose}")
        for g in self.goals:
            if g not in GOAL_FAMILIES:
                raise ValueError(f"template {self.name}: unknown goal family {g}")


def load_templates(directory: str | Path = TEMPLATE_DIR) -> list[Template]:
    return [Template.load(p) for p in sorted(Path(directory).glob("*.yaml"))]


@dataclass
class SampledObject:
    prim: str  # unique prim name in the scene
    obj: CatalogObject
    slot: Slot
    pose_index: int


class SamplingError(RuntimeError):
    pass


def _pick_pose(o: CatalogObject, want: str, rng: random.Random) -> int:
    robust = o.geom.table_poses()
    lying = [i for i in robust if not o.geom.poses[i].upright]
    if want == "lying":
        if not lying:
            raise SamplingError(f"{o.name} has no robust lying pose")
        return rng.choice(lying)
    if want == "upright" and 0 in robust:
        return 0
    if want == "any" and robust:
        return 0 if 0 in robust and rng.random() < 0.6 else rng.choice(robust)
    if robust:  # authored pose too tippy/unstable: use the most robust pose
        return max(robust, key=lambda i: o.geom.poses[i].tip_ratio)
    raise SamplingError(f"{o.name} has no robust table pose")


def sample_objects(template: Template, catalog: Catalog, rng: random.Random) -> list[SampledObject]:
    """Draw concrete objects for every slot (without replacement within a scene)."""
    used: set[str] = set()
    out: list[SampledObject] = []
    for slot in template.slots:
        lo, hi = slot.count
        n = rng.randint(lo, hi)
        if n == 0:
            continue
        sel = slot.select
        req = slot.require
        pool = catalog.select(names=sel.get("names"), category=sel.get("category"), color=sel.get("color"),
                              container=req.get("container"), support=req.get("support"),
                              graspable=req.get("graspable"))
        if slot.relation != "table":
            pool = [o for o in pool if o.graspable and o.geom.poses[0].stable]
        else:  # must have a pose that is hard to tip on a flat table
            pool = [o for o in pool if o.geom.table_poses()]
        if slot.pose == "lying":
            pool = [o for o in pool if any(not o.geom.poses[i].upright for i in o.geom.table_poses())]
        pool = [o for o in pool if o.name not in used]
        if len(pool) < lo:
            raise SamplingError(f"slot {slot.slot}: only {len(pool)} candidates for count >= {lo}")
        for o in rng.sample(pool, min(n, len(pool))):
            used.add(o.name)
            pose = 0 if slot.relation != "table" else _pick_pose(o, slot.pose, rng)
            out.append(SampledObject(o.name, o, slot, pose))
    return out

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quest generation: sample goal facts over a scene, prove them solvable by chaining.

Each goal family maps to one competency axis and one RoboLab termination. The
family picks targets from the scene's symbolic state (so only affordances the
mesh geometry allowed are usable), ``logic.Chainer`` backward-chains a plan —
the proof of solvability — and ``plan_to_dag`` yields the subtask stages.
Referring expressions are chosen so the instruction is unambiguous: if the
display name is shared, color must disambiguate (a color-axis task), otherwise
the goal is rejected.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from robolab.constants import DIFFICULTY_THRESHOLDS, SKILL_WEIGHTS
from robolab.scene_gen.logic import Chainer, P, Plan, Prop, plan_to_dag
from robolab.scene_gen.logic.rules import SPATIAL_RELATIONS
from robolab.scene_gen.placer import TABLE_ID
from robolab.scene_gen.scene_builder import SceneInstance


class NoGoal(RuntimeError):
    pass


@dataclass
class Goal:
    family: str
    facts: list[Prop]
    targets: list[str]  # manipulated objects (for choose-K: the candidate set)
    hosts: list[str]  # containers / surfaces / references
    attributes: list[str]
    params: dict = field(default_factory=dict)  # family-specific (K, relation, order, groups, ...)
    plan: Plan | None = None
    dag: list = field(default_factory=list)

    @property
    def num_subtasks(self) -> int:
        return self.params.get("K", len(self.manipulated))

    @property
    def manipulated(self) -> list[str]:
        return self.params.get("manipulated", self.targets)

    def difficulty(self) -> tuple[int, str]:
        """Mirror of subtask_utils.compute_difficulty_score (count + hardest skill)."""
        score = self.num_subtasks + max((SKILL_WEIGHTS.get(a, 0) for a in self.attributes), default=0)
        lo, hi = DIFFICULTY_THRESHOLDS
        return score, "simple" if score <= lo else ("moderate" if score <= hi else "complex")


# ---------------------------------------------------------------------------
# Referring expressions
# ---------------------------------------------------------------------------

def refer(scene: SceneInstance, prim: str) -> tuple[str, list[str]] | None:
    """Shortest unambiguous noun phrase for ``prim`` and the visual axis it uses."""
    o = scene.objects[prim]
    same = [p for p, x in scene.objects.items() if p != prim and x.display == o.display]
    if not same:
        return o.display, ["semantics"]
    if not [p for p in same if scene.objects[p].color == o.color] and o.color not in o.display:
        return f"{o.color} {o.display}", ["color"]
    if len(same) == 1:  # two look-alikes: disambiguate by side, as seen from the robot
        dy = scene.placements[prim].y - scene.placements[same[0]].y
        if abs(dy) > 0.15:
            return f"{'left' if dy > 0 else 'right'} {o.display}", ["spatial"]
    return None


def _ref_or_raise(scene, prim):
    r = refer(scene, prim)
    if r is None:
        raise NoGoal(f"{prim} cannot be referred to unambiguously")
    return r


def _axes(scene, *prims) -> list[str]:
    axes = set()
    for p in prims:
        axes |= set(_ref_or_raise(scene, p)[1])
    return sorted(axes)


def _facts(scene, name):
    return scene.state.query(name)


def _on_table_movables(scene) -> list[str]:
    st = scene.state
    return [p for p in scene.objects if P("movable", p) in st and P("on", p, TABLE_ID) in st and P("clear", p) in st]


def _relation_holds(scene, rel: str, o: str, ref: str) -> bool:
    """Robot-frame 45° cone test, as object_left_of & co. evaluate it (+y left, +x away)."""
    a, b = scene.placements[o], scene.placements[ref]
    dx, dy = a.x - b.x, a.y - b.y
    return {"left_of": dy > abs(dx), "right_of": -dy > abs(dx),
            "behind": dx > abs(dy), "in_front_of": -dx > abs(dy)}[rel]


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

def _put_in(scene, rng):
    st = scene.state
    cands = [(o, c.args[1]) for o in _on_table_movables(scene) for c in _facts(scene, "fits_in")
             if c.args[0] == o]
    rng.shuffle(cands)
    for o, c in cands:
        if refer(scene, o) and refer(scene, c):
            return Goal("put_in", [P("in", o, c)], [o], [c], _axes(scene, o, c))
    raise NoGoal("no (object, container) pair with fits_in")


def _put_on(scene, rng):
    st = scene.state
    cands = [(o, s.args[1]) for o in _on_table_movables(scene) for s in _facts(scene, "stackable_on")
             if s.args[0] == o and P("clear", s.args[1]) in st
             and scene.objects[s.args[1]].category in ("plate", "box")]  # blocks belong to `stack`
    rng.shuffle(cands)
    for o, s in cands:
        if refer(scene, o) and refer(scene, s):
            return Goal("put_on", [P("on", o, s)], [o], [s], _axes(scene, o, s))
    raise NoGoal("no (object, plate/box) pair with stackable_on")


def _take_out(scene, rng):
    st = scene.state
    cands = [(f.args[0], f.args[1]) for f in _facts(scene, "in")
             if P("graspable", f.args[0]) in st and P("clear", f.args[0]) in st]
    rng.shuffle(cands)
    for o, c in cands:
        if refer(scene, o) and refer(scene, c):
            return Goal("take_out", [P("on", o, TABLE_ID)], [o], [c], _axes(scene, o, c))
    raise NoGoal("nothing graspable inside a container")


def _stack(scene, rng):
    st = scene.state
    blocks = [p for p in _on_table_movables(scene) if scene.objects[p].category == "block" and P("surface", p) in st]
    if len(blocks) < 2:
        raise NoGoal("fewer than 2 stackable blocks")
    n = min(len(blocks), rng.choice([2, 3]))
    tower = rng.sample(blocks, n)
    facts = [P("on", tower[i + 1], tower[i]) for i in range(n - 1)]
    if not all(P("stackable_on", tower[i + 1], tower[i]) in st for i in range(n - 1)):
        raise NoGoal("tower not stackable")
    return Goal("stack", facts, tower, [], ["stacking", "color"],
                {"order": "bottom_to_top", "manipulated": tower[1:]})


def _spatial(scene, rng):
    movs = _on_table_movables(scene)
    refs = [p for p in scene.objects if P("on", p, TABLE_ID) in scene.state]
    cands = [(o, r, rel) for o in movs for r in refs if r != o for rel in SPATIAL_RELATIONS
             if not _relation_holds(scene, rel, o, r)]
    rng.shuffle(cands)
    for o, r, rel in cands:
        if refer(scene, o) and refer(scene, r):
            return Goal("spatial", [P(rel, o, r)], [o], [r], sorted(set(_axes(scene, o, r)) | {"spatial"}),
                        {"relation": rel})
    raise NoGoal("no spatial pair")


def _count_k_of_n(scene, rng):
    st = scene.state
    movs = _on_table_movables(scene)
    conts = [p for p in scene.objects if P("container", p) in st and refer(scene, p)]
    rng.shuffle(conts)
    for c in conts:
        by_cat: dict[str, list[str]] = {}
        for o in movs:
            if P("fits_in", o, c) in st and o != c:
                by_cat.setdefault(scene.objects[o].category, []).append(o)
        groups = [g for g in by_cat.values() if len(g) >= 3]
        if not groups:
            continue
        group = sorted(rng.choice(groups))
        k = rng.randint(1, len(group) - 1)
        chosen = rng.sample(group, k)
        return Goal("count_k_of_n", [P("in", o, c) for o in chosen], group, [c], ["counting", "semantics"],
                    {"K": k, "manipulated": chosen, "category": scene.objects[group[0]].category})
    raise NoGoal("no container with >= 3 same-category fitting objects")


def _reorient(scene, rng):
    st = scene.state
    lying = [p for p in scene.objects if P("movable", p) in st and P("upright", p) not in st
             and P("on", p, TABLE_ID) in st]
    if not lying:
        raise NoGoal("nothing lying over")
    cats = {scene.objects[p].category for p in lying}
    cat = rng.choice(sorted(cats))
    targets = sorted(p for p in lying if scene.objects[p].category == cat)
    return Goal("reorient", [P("upright", o) for o in targets], targets, [], ["reorientation"], {"category": cat})


def _sort_two_bins(scene, rng):
    st = scene.state
    movs = _on_table_movables(scene)
    conts = [p for p in scene.objects if P("container", p) in st and P("movable", p) not in st and refer(scene, p)]
    if len(conts) < 2:
        raise NoGoal("fewer than 2 fixed containers")
    c1, c2 = rng.sample(conts, 2)
    by_cat: dict[str, list[str]] = {}
    for o in movs:
        by_cat.setdefault(scene.objects[o].category, []).append(o)
    cats = [k for k, v in by_cat.items() if v]
    if len(cats) < 2:
        raise NoGoal("fewer than 2 categories to sort")
    k1, k2 = rng.sample(sorted(cats), 2)
    # A group is a whole category ("put the cans in ..."), so every member must fit.
    g1, g2 = sorted(by_cat[k1]), sorted(by_cat[k2])
    if len(g1) + len(g2) > 5 or not all(P("fits_in", o, c1) in st for o in g1) \
            or not all(P("fits_in", o, c2) in st for o in g2):
        raise NoGoal("sort groups do not fit their bins")
    facts = [P("in", o, c1) for o in g1] + [P("in", o, c2) for o in g2]
    return Goal("sort_two_bins", facts, g1 + g2, [c1, c2], ["sorting", "semantics"],
                {"groups": [(k1, g1, c1), (k2, g2, c2)]})


FAMILIES = {
    "put_in": _put_in, "put_on": _put_on, "take_out": _take_out, "stack": _stack, "spatial": _spatial,
    "count_k_of_n": _count_k_of_n, "reorient": _reorient, "sort_two_bins": _sort_two_bins,
}


def sample_goal(scene: SceneInstance, rng: random.Random, families: list[str] | None = None) -> Goal:
    """Try goal families in order (``families`` as given, else the template's, shuffled); first that plans wins."""
    fams = list(families) if families else rng.sample(scene.template.goals, len(scene.template.goals))
    errors = []
    for fam in fams:
        try:
            goal = FAMILIES[fam](scene, rng)
        except NoGoal as e:
            errors.append(f"{fam}: {e}")
            continue
        plan = Chainer(rng=random.Random(rng.random())).plan(scene.state, goal.facts)
        if plan is None:
            errors.append(f"{fam}: chainer found no plan")
            continue
        goal.plan, goal.dag = plan, plan_to_dag(plan)
        return goal
    raise NoGoal("; ".join(errors))

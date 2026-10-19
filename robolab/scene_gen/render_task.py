# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a goal twice — as executable success criteria and as language.

Both renderings are projections of the same goal facts, so instruction and
termination agree by construction (RoboLab's LLM task generation measured only
57-60 % alignment on color/counting because the two were written separately).
Output files follow the shape of ``robolab/tasks/benchmark/*.py``.
"""

from __future__ import annotations

import re

from robolab.scene_gen.placer import TABLE_ID
from robolab.scene_gen.quest import Goal, refer
from robolab.scene_gen.scene_builder import SceneInstance

CATEGORY_NOUN = {
    "fruit": ("piece of fruit", "fruits"), "vegetable": ("vegetable", "vegetables"),
    "canned food": ("can", "cans"), "boxed food": ("food box", "food boxes"), "dairy": ("dairy item", "dairy items"),
    "condiment": ("condiment", "condiments"), "block": ("block", "blocks"), "toy": ("toy", "toys"),
    "mug": ("mug", "mugs"), "bowl": ("bowl", "bowls"), "plate": ("plate", "plates"), "bin": ("bin", "bins"),
    "box": ("box", "boxes"), "utensil": ("utensil", "utensils"), "electronics": ("device", "devices"),
    "stationery": ("office item", "office items"), "accessory": ("accessory", "accessories"),
    "tool": ("tool", "tools"), "bread": ("bread item", "bread items"), "snack": ("snack", "snacks"),
}
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
REL_TEXT = {"left_of": "to the left of", "right_of": "to the right of",
            "in_front_of": "in front of", "behind": "behind"}
REL_FUNC = {"left_of": "object_left_of", "right_of": "object_right_of",
            "in_front_of": "object_in_front_of", "behind": "object_behind"}


def class_name(scene: SceneInstance) -> str:
    return "Gen" + "".join(w.capitalize() for w in scene.template.name.split("_")) + f"{scene.seed:04d}Task"


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

def _the(scene, prim) -> str:
    return "the " + refer(scene, prim)[0]


def _noun(scene, prim, plural=False) -> str:
    sing, pl = CATEGORY_NOUN.get(scene.objects[prim].category, (scene.objects[prim].category,) * 2)
    return pl if plural else sing


def _side(scene, prim) -> str:
    rel = scene.relations.get(prim)
    if rel:
        return f"{'inside' if rel[0] == 'in' else 'on'} the {scene.objects[rel[1]].display}"
    y = scene.placements[prim].y
    return "on the left side of the table" if y > 0.12 else ("on the right side of the table" if y < -0.12
                                                            else "in the middle of the table")


def _specific(scene, prim) -> str:
    o = scene.objects[prim]
    adj = [] if o.color in refer(scene, prim)[0] or o.color == "multicolor" else [o.color]
    if o.size != "medium":
        adj.insert(0, o.size)
    return f"the {' '.join(adj + [refer(scene, prim)[0]])} {_side(scene, prim)}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def instructions(scene: SceneInstance, g: Goal) -> dict[str, str]:
    t = g.targets
    if g.family in ("put_in", "put_on"):
        o, h = t[0], g.hosts[0]
        prep = "in" if g.family == "put_in" else "on"
        return {
            "default": f"Put {_the(scene, o)} {prep} {_the(scene, h)}",
            "vague": f"Put the {_noun(scene, o)} {prep} the {_noun(scene, h)}",
            "specific": f"Pick up {_specific(scene, o)} and place it {'inside' if prep == 'in' else 'on top of'} "
                        f"{_the(scene, h)}",
        }
    if g.family == "take_out":
        o, c = t[0], g.hosts[0]
        return {
            "default": f"Take {_the(scene, o)} out of {_the(scene, c)}",
            "vague": f"Take the {_noun(scene, o)} out",
            "specific": f"Pick up {_the(scene, o)} from inside {_the(scene, c)} and put it down on the table",
        }
    if g.family == "stack":
        names = [refer(scene, p)[0] for p in t]
        return {
            "default": f"Stack the blocks from bottom to top: {_join(names)}",
            "vague": f"Stack {NUMBER[len(t)]} blocks in the order {_join(names)}",
            "specific": f"Build a single tower with {_the(scene, t[0])} at the bottom, then "
                        + ", then ".join(_the(scene, p) for p in t[1:]) + " on top",
        }
    if g.family == "spatial":
        o, r, rel = t[0], g.hosts[0], g.params["relation"]
        return {
            "default": f"Put {_the(scene, o)} {REL_TEXT[rel]} {_the(scene, r)}",
            "vague": f"Move the {_noun(scene, o)} {REL_TEXT[rel]} the {_noun(scene, r)}",
            "specific": f"Pick up {_specific(scene, o)} and place it on the table {REL_TEXT[rel]} "
                        f"{_the(scene, r)}, as seen from the robot",
        }
    if g.family == "count_k_of_n":
        k, c = g.params["K"], g.hosts[0]
        word = _noun(scene, t[0], plural=k > 1)
        return {
            "default": f"Put exactly {NUMBER[k]} {word} in {_the(scene, c)}",
            "vague": f"Put {NUMBER[k]} {word} in the {_noun(scene, c)}",
            "specific": f"Choose {NUMBER[k]} of the {NUMBER[len(t)]} {_noun(scene, t[0], plural=True)} on the table "
                        f"and place exactly {NUMBER[k]} of them inside {_the(scene, c)}, leaving the rest out",
        }
    if g.family == "reorient":
        word = _noun(scene, t[0], plural=len(t) > 1)
        it = "them" if len(t) > 1 else "it"
        return {
            "default": f"Set the {word} upright" if len(t) > 1 else f"Set {_the(scene, t[0])} upright",
            "vague": f"Fix the {word} that fell over",
            "specific": f"Pick up {_join([_specific(scene, p) for p in t])}, which "
                        f"{'are' if len(t) > 1 else 'is'} lying on {'their' if len(t) > 1 else 'its'} side, "
                        f"and stand {it} upright on the table",
        }
    if g.family == "sort_two_bins":
        parts = [f"the {_noun(scene, grp[0], plural=len(grp) > 1)} in {_the(scene, c)}" for _, grp, c in g.params["groups"]]
        return {
            "default": "Put " + " and ".join(parts),
            "vague": "Sort the groceries into the bins by type",
            "specific": "Sort the items: " + "; ".join(
                f"place {_join([_the(scene, p) for p in grp])} inside {_the(scene, c)}" for _, grp, c in g.params["groups"]),
        }
    raise ValueError(g.family)


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------

def _success_and_subtasks(scene: SceneInstance, g: Goal) -> tuple[str, dict, list[str], set[str]]:
    """(termination func, params, subtask expressions, conditional imports)."""
    t, imports = g.targets, set()
    if g.family == "put_in":
        imports |= {"object_in_container", "pick_and_place"}
        return ("object_in_container", {"object": t, "container": g.hosts[0], "logical": "all",
                                        "require_gripper_detached": True},
                [f"pick_and_place(object={t!r}, container={g.hosts[0]!r}, logical='all', score=1.0)"], imports)
    if g.family == "put_on":
        imports |= {"object_on_top", "pick_and_place_on_surface"}
        return ("object_on_top", {"object": t, "reference_object": g.hosts[0], "logical": "all",
                                  "require_gripper_detached": True},
                [f"pick_and_place_on_surface(object={t!r}, surface={g.hosts[0]!r}, logical='all', score=1.0)"],
                imports)
    if g.family == "take_out":
        imports |= {"object_outside_of", "object_grabbed", "object_dropped"}
        o, c = t[0], g.hosts[0]
        sub = (f"Subtask(name='take_out', conditions={{{o!r}: [\n"
               f"            partial(object_grabbed, object={o!r}),\n"
               f"            partial(object_outside_of, object={o!r}, container={c!r}, tolerance=0.0, "
               f"require_gripper_detached=True),\n"
               f"            partial(object_dropped, object={o!r}),\n"
               f"        ]}}, logical='all', score=1.0)")
        return ("object_outside_of", {"object": t, "container": c, "logical": "all", "tolerance": 0.0,
                                      "require_gripper_detached": True}, [sub], imports)
    if g.family == "stack":
        imports |= {"stacked"}
        n = len(t) - 1
        subs = [f"Subtask(conditions=partial(stacked, objects={[t[i], t[i + 1]]!r}, order='bottom_to_top'), "
                f"score={round(1.0 / n, 4)})" for i in range(n)]
        return "stacked", {"objects": t, "order": "bottom_to_top"}, subs, imports
    if g.family == "spatial":
        rel, o, r = g.params["relation"], t[0], g.hosts[0]
        f = REL_FUNC[rel]
        imports |= {f, "object_grabbed", "object_dropped"}
        sub = (f"Subtask(name='{rel}', conditions=[\n"
               f"            partial(object_grabbed, object={o!r}),\n"
               f"            partial({f}, object={o!r}, reference_object={r!r}, frame_of_reference='robot', "
               f"mirrored=False),\n"
               f"            partial(object_dropped, object={o!r}),\n"
               f"        ], logical='all', score=1.0)")
        return (f, {"object": o, "reference_object": r, "frame_of_reference": "robot", "mirrored": False,
                    "require_gripper_detached": True}, [sub], imports)
    if g.family == "count_k_of_n":
        imports |= {"object_in_container", "pick_and_place"}
        k = g.params["K"]
        return ("object_in_container", {"object": t, "container": g.hosts[0], "logical": "choose", "K": k,
                                        "require_gripper_detached": True},
                [f"pick_and_place(object={t!r}, container={g.hosts[0]!r}, logical='choose', K={k}, score=1.0)"],
                imports)
    if g.family == "reorient":
        imports |= {"object_upright", "object_grabbed"}
        conds = ", ".join(
            f"{o!r}: [partial(object_grabbed, object={o!r}), partial(object_upright, object={o!r}, up_axis='z', "
            f"require_gripper_detached=True)]" for o in t)
        sub = f"Subtask(name='reorient', conditions={{{conds}}}, logical='all', score=1.0)"
        return ("object_upright", {"object": t, "up_axis": "z", "logical": "all", "require_gripper_detached": True},
                [sub], imports)
    if g.family == "sort_two_bins":
        imports |= {"object_groups_in_containers", "pick_and_place_grouped"}
        groups = [{"object": grp, "container": c, "logical": "all", "require_gripper_detached": True}
                  for _, grp, c in g.params["groups"]]
        sub_groups = [{"object": grp, "container": c} for _, grp, c in g.params["groups"]]
        return ("object_groups_in_containers", {"groups": groups},
                [f"pick_and_place_grouped(groups={sub_groups!r}, logical='all', score=1.0)"], imports)
    raise ValueError(g.family)


def render_task_source(scene: SceneInstance, g: Goal) -> str:
    cls = class_name(scene)
    base = cls.removesuffix("Task")
    func, params, subtasks, imports = _success_and_subtasks(scene, g)
    instr = instructions(scene, g)
    contact = sorted(scene.objects) + [TABLE_ID]
    n = len(g.manipulated)
    episode = min(50 + 25 * n + (25 if g.family in ("stack", "reorient") else 0), 150)
    score, label = g.difficulty()
    uses_partial = any("partial(" in s for s in subtasks)
    uses_subtask = any(s.startswith("Subtask(") for s in subtasks)
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "",
        f'"""Generated by scripts/scenegen.py — template {scene.template.name}, seed {scene.seed}.',
        "",
        f"Goal facts: {', '.join(str(f) for f in g.facts)}",
        f"Reference plan: {'; '.join(g.plan.to_strings())}",
        f"Difficulty: {label} (score {score}). Do not edit by hand; regenerate instead.",
        '"""',
        "",
        "from dataclasses import dataclass",
    ]
    if uses_partial:
        lines.append("from functools import partial")
    lines += [
        "",
        "import isaaclab.envs.mdp as mdp",
        "from isaaclab.managers import TerminationTermCfg as DoneTerm",
        "from isaaclab.utils import configclass",
        "",
        "from robolab.core.scenes.utils import import_scene",
        f"from robolab.core.task.conditionals import {', '.join(sorted(imports))}",
    ]
    if uses_subtask:
        lines.append("from robolab.core.task.subtask import Subtask")
    lines += [
        "from robolab.core.task.task import Task",
        "",
        "",
        "@configclass",
        f"class {base}Terminations:",
        "    time_out = DoneTerm(func=mdp.time_out, time_out=True)",
        f"    success = DoneTerm(func={func}, params={params!r})",
        "",
        "",
        "@dataclass",
        f"class {cls}(Task):",
        f"    contact_object_list = {contact!r}",
        f'    scene = import_scene("{scene.name}.usda", contact_object_list)',
        f"    terminations = {base}Terminations",
        "    instruction = {",
        *[f"        {k!r}: {v!r}," for k, v in instr.items()],
        "    }",
        f"    episode_length_s: int = {episode}",
        f"    attributes = {sorted(set(g.attributes))!r}",
        "    subtasks = [",
        *[f"        {s}," for s in subtasks],
        "    ]",
        "",
    ]
    return "\n".join(lines)


def termination_objects(src: str) -> set[str]:
    """Object names a rendered task's success params reference (for CPU validation)."""
    m = re.search(r"success = DoneTerm\(func=\w+, params=(\{.*\})\)", src)
    import ast

    params = ast.literal_eval(m.group(1))
    names = set()
    for key in ("object", "container", "objects", "reference_object", "surface"):
        v = params.get(key)
        names |= {v} if isinstance(v, str) else set(v or [])
    for grp in params.get("groups", []):
        names |= set(grp["object"]) | {grp["container"]}
    return names

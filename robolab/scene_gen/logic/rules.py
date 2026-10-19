# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab manipulation rules (tabletop pick-and-place; no rooms, doors or cooking).

Fact vocabulary
---------------
Types (unary):    movable(o) container(c) surface(s) table(t)
Relations:        on(o, s)   in(o, c)   holding(o)   free_gripper
                  left_of / right_of / in_front_of / behind (o, ref)
State:            upright(o)   clear(s)  (nothing rests on s)
Static, from mesh geometry (geometry.py) — never added by a rule:
                  graspable(o)  fits_in(o, c)  stackable_on(o, s)
Attributes:       color(o, c)  category(o, k)  size(o, z)
"""

from robolab.scene_gen.logic.core import P, Rule

_O, _H, _C, _S, _R = "?o", "?h", "?c", "?s", "?r"

TAKE_FROM_SURFACE = Rule(
    "take", ((_O, "movable"), (_H, "surface")),
    pre=(P("free_gripper"), P("graspable", _O), P("on", _O, _H), P("clear", _O)),
    add=(P("holding", _O), P("clear", _H)),
    delete=(P("free_gripper"), P("on", _O, _H)),
)
TAKE_FROM_CONTAINER = Rule(
    "take_out", ((_O, "movable"), (_H, "container")),
    pre=(P("free_gripper"), P("graspable", _O), P("in", _O, _H), P("clear", _O)),
    add=(P("holding", _O),),
    delete=(P("free_gripper"), P("in", _O, _H)),
)
PUT_ON_TABLE = Rule(
    "put_on_table", ((_O, "movable"), (_H, "table")),
    pre=(P("holding", _O),),
    add=(P("on", _O, _H), P("free_gripper")),
    delete=(P("holding", _O),),
)
PUT_ON = Rule(
    "put_on", ((_O, "movable"), (_S, "surface")),
    pre=(P("holding", _O), P("stackable_on", _O, _S), P("clear", _S)),
    neg=(P("table", _S),),
    add=(P("on", _O, _S), P("free_gripper")),
    delete=(P("holding", _O), P("clear", _S)),
)
PUT_IN = Rule(
    "put_in", ((_O, "movable"), (_C, "container")),
    pre=(P("holding", _O), P("fits_in", _O, _C)),
    add=(P("in", _O, _C), P("free_gripper")),
    delete=(P("holding", _O),),
)
REORIENT = Rule(
    "reorient", ((_O, "movable"),),
    pre=(P("holding", _O),),
    neg=(P("upright", _O),),
    add=(P("upright", _O),),
)


def _put_relative(rel: str) -> Rule:
    return Rule(
        f"put_{rel}", ((_O, "movable"), (_R, "movable"), (_H, "table")),
        pre=(P("holding", _O), P("on", _R, _H)),
        add=(P(rel, _O, _R), P("on", _O, _H), P("free_gripper")),
        delete=(P("holding", _O),),
    )


SPATIAL_RELATIONS = ("left_of", "right_of", "in_front_of", "behind")
RULES = (TAKE_FROM_SURFACE, TAKE_FROM_CONTAINER, PUT_ON_TABLE, PUT_ON, PUT_IN, REORIENT,
         *(_put_relative(r) for r in SPATIAL_RELATIONS))
STATIC_FACTS = {"graspable", "fits_in", "stackable_on", "movable", "container", "surface", "table",
                "color", "category", "size"}

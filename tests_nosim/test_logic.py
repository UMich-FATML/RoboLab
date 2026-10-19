# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

from robolab.scene_gen.logic import Chainer, P, State, plan_to_dag, validate
from robolab.scene_gen.pipeline import parse_action

BASE = [P("free_gripper"), P("table", "table"), P("surface", "table"), P("clear", "table")]


def obj(name, *extra):
    return [P("movable", name), P("graspable", name), P("on", name, "table"), P("clear", name), *extra]


def fruit_bowl(fits=True):
    facts = BASE + obj("lemon") + obj("banana") + [P("container", "bowl"), P("on", "bowl", "table")]
    if fits:
        facts.append(P("fits_in", "lemon", "bowl"))
    return State(facts)


def test_put_in_plan_is_minimal():
    plan = Chainer(rng=random.Random(0)).plan(fruit_bowl(), [P("in", "lemon", "bowl")])
    assert plan.to_strings() == ["take(lemon, table)", "put_in(lemon, bowl)"]


def test_static_affordances_gate_goals():
    assert Chainer().plan(fruit_bowl(fits=False), [P("in", "lemon", "bowl")]) is None
    assert Chainer().plan(fruit_bowl(), [P("in", "banana", "bowl")]) is None


def test_reorient_releases_the_object():
    plan = Chainer().plan(State(BASE + obj("mug")), [P("upright", "mug")])
    assert plan.to_strings() == ["take(mug, table)", "reorient(mug)", "put_on_table(mug, table)"]


def test_stack_orders_dag_and_plan():
    facts = list(BASE)
    for b in "rgb":
        facts += obj(b, P("surface", b))
        facts += [P("stackable_on", b, o) for o in "rgb" if o != b]
    for seed in range(3):
        plan = Chainer(rng=random.Random(seed)).plan(State(facts), [P("on", "g", "r"), P("on", "b", "g")])
        assert len(plan) == 4
        assert plan_to_dag(plan) == [["g"], ["b"]]


def test_independent_objects_share_a_stage():
    facts = BASE + obj("a") + obj("b") + [P("container", "c"), P("on", "c", "table"),
                                          P("fits_in", "a", "c"), P("fits_in", "b", "c")]
    plan = Chainer().plan(State(facts), [P("in", "a", "c"), P("in", "b", "c")])
    assert len(plan_to_dag(plan)) == 1  # apple-then-orange == orange-then-apple


def test_validate_rejects_inapplicable_plans():
    s = fruit_bowl()
    assert validate(s, [parse_action("take(lemon, table)"), parse_action("put_in(lemon, bowl)")], [P("in", "lemon", "bowl")])
    assert not validate(s, [parse_action("put_in(lemon, bowl)")], [P("in", "lemon", "bowl")])

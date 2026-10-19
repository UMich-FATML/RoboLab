# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward chaining from goal facts to an executable plan.

Like TextWorld's quest generator, a task starts from the *goal* and works
backwards: for an unmet goal fact pick a rule whose effect produces it, then
recursively achieve that rule's preconditions in the current state (goal
regression / means-ends analysis), finally apply the action. Static facts
(affordances from mesh geometry) cannot be produced by any rule, so the chainer
can never propose a physically impossible goal. The returned plan is replayed
forward to prove it reaches the goal — the proof of solvability — and is turned
into a dependency DAG for RoboLab's (parallel-within-stage) subtasks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from robolab.scene_gen.logic.core import Action, Prop, Rule, State, is_var, substitute, unify
from robolab.scene_gen.logic.rules import RULES, STATIC_FACTS

_SERIAL = {"free_gripper", "holding"}  # facts that only serialize a single gripper


@dataclass
class Plan:
    actions: list[Action]
    final: State

    def __len__(self) -> int:
        return len(self.actions)

    def to_strings(self) -> list[str]:
        return [str(a) for a in self.actions]


class Chainer:
    def __init__(self, rules=RULES, rng: random.Random | None = None, max_depth: int = 6):
        self.rules = rules
        self.rng = rng or random.Random(0)
        self.max_depth = max_depth

    # -- public ---------------------------------------------------------------
    def plan(self, state: State, goals: list[Prop], attempts: int = 8) -> Plan | None:
        """Plan achieving all ``goals`` jointly from ``state``; tries goal orders."""
        goals = list(goals)
        for _ in range(attempts):
            # A task ends with the object released: append free_gripper last.
            res = self._achieve_all(state, goals + [Prop("free_gripper")], [], 0)
            if res is not None:
                actions, final = res
                if final.holds(goals) and validate(state, actions, goals):
                    return Plan(actions, final)
            self.rng.shuffle(goals)
        return None

    # -- search ---------------------------------------------------------------
    def _achieve_all(self, state, goals, protected, depth):
        actions = []
        for g in goals:
            if g in state:
                protected = protected + [g]
                continue
            res = self._achieve(state, g, protected, depth)
            if res is None:
                return None
            acts, state = res
            actions += acts
            protected = protected + [g]
        if not state.holds(protected):
            return None
        return actions, state

    def _achieve(self, state: State, goal: Prop, protected, depth):
        if goal in state:
            return [], state
        if depth >= self.max_depth or goal.name in STATIC_FACTS:
            return None
        candidates = []
        for rule in self.rules:
            for eff in rule.add:
                binding = unify(eff, goal, {})
                if binding is not None:
                    candidates += [(cost, tie, rule, b) for cost, tie, b in self._ranked(rule, binding, state)]
        candidates.sort(key=lambda t: (t[0], t[1]))
        for _, _, rule, full in candidates[:32]:
            res = self._try_rule(state, rule, full, protected, depth)
            if res is not None:
                return res
        return None

    def _try_rule(self, state, rule: Rule, binding, protected, depth):
        action = rule.ground(binding)
        if any(n in state for n in action.neg):
            return None
        pre = action.pre
        # Static and type preconditions must already hold — no rule can create them.
        if any(p.name in STATIC_FACTS and p not in state for p in pre):
            return None
        # Achieve "holding" last: everything else (e.g. clearing) needs a free gripper.
        dynamic = sorted((p for p in pre if p not in state), key=lambda p: p.name == "holding")
        res = self._achieve_all(state, dynamic, [p for p in protected], depth + 1)
        if res is None:
            return None
        acts, st = res
        if not st.is_applicable(action):
            return None
        st2 = st.apply(action)
        if not st2.holds(protected):
            return None
        return acts + [action], st2

    def _ranked(self, rule: Rule, partial: dict, state: State, limit: int = 24):
        """Complete bindings, fewest unmet preconditions first (random tie-break).

        Prefers grounding ``take(o, h)`` with the ``h`` the object is actually on,
        instead of first moving it there — keeps plans minimal.
        """
        scored = []
        for b in self._bindings(rule, partial, state):
            a = rule.ground(b)
            pre = a.pre
            if any(p.name in STATIC_FACTS and p not in state for p in pre):
                continue
            scored.append((sum(p not in state for p in pre), self.rng.random(), b))
            if len(scored) > 400:
                break
        scored.sort(key=lambda t: (t[0], t[1]))
        return scored[:limit]

    def _bindings(self, rule: Rule, partial: dict, state: State):
        """Complete ``partial`` over the rule's typed params; entities are distinct."""
        free = [(v, t) for v, t in rule.params if v not in partial]
        if not free:
            yield dict(partial)
            return
        v, t = free[0]
        used = set(partial.values())
        for ent in self._shuffled(state.entities_of(t)):
            if ent in used:
                continue
            yield from self._bindings(rule, {**partial, v: ent}, state)

    def _shuffled(self, seq):
        seq = list(seq)
        self.rng.shuffle(seq)
        return seq


def validate(state: State, actions: list[Action], goals: list[Prop]) -> bool:
    """Replay ``actions`` forward from ``state``; True iff every step applies and goals hold."""
    try:
        for a in actions:
            state = state.apply(a)
    except ValueError:
        return False
    return state.holds(goals)


def object_of(action: Action) -> str:
    return action.args[0]


def plan_to_dag(plan: Plan) -> list[list[str]]:
    """Group the plan into sequential stages of independent objects.

    Each manipulated object contributes a ladder (its consecutive actions). A
    ladder depends on another if it places onto/relative to an object that the
    other ladder moves (stack blue before putting red on blue). Gripper
    serialization (free_gripper/holding) is not a dependency: apple-then-orange
    equals orange-then-apple. Returns stages of object names in dependency order.
    """
    order: list[str] = []
    hosts: dict[str, set[str]] = {}
    for a in plan.actions:
        o = object_of(a)
        if o not in order:
            order.append(o)
        hosts.setdefault(o, set()).update(arg for arg in a.args[1:])
    deps = {o: {h for h in hosts[o] if h in hosts and h != o} for o in order}
    stages, done = [], set()
    while len(done) < len(order):
        layer = [o for o in order if o not in done and deps[o] <= done]
        if not layer:  # cycle (should not happen) — fall back to plan order
            layer = [o for o in order if o not in done]
        stages.append(layer)
        done |= set(layer)
    return stages


__all__ = ["Chainer", "Plan", "plan_to_dag", "validate", "is_var", "substitute"]

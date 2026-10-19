# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed facts, rules and states — a minimal TextWorld-style logic core.

Facts are ground propositions ``name(arg, ...)`` over scene entities. A
``Rule`` has typed parameters, positive/negative preconditions and add/delete
effects; grounding it with a binding yields an ``Action`` that transforms a
``State``. Static facts (affordances such as ``fits_in``) are ordinary facts no
rule ever adds, so chaining can only use what the geometry allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, NamedTuple


class Prop(NamedTuple):
    name: str
    args: tuple = ()

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    @staticmethod
    def parse(text: str) -> "Prop":
        text = text.strip()
        if "(" not in text:
            return Prop(text, ())
        name, rest = text.split("(", 1)
        args = tuple(a.strip() for a in rest.rstrip(")").split(",") if a.strip())
        return Prop(name.strip(), args)


def P(name: str, *args: str) -> Prop:
    return Prop(name, tuple(args))


def is_var(term: str) -> bool:
    return term.startswith("?")


def substitute(prop: Prop, binding: dict[str, str]) -> Prop:
    return Prop(prop.name, tuple(binding.get(a, a) for a in prop.args))


def unify(pattern: Prop, fact: Prop, binding: dict[str, str]) -> dict[str, str] | None:
    """Extend ``binding`` so that ``pattern`` matches ``fact``, or return None."""
    if pattern.name != fact.name or len(pattern.args) != len(fact.args):
        return None
    out = dict(binding)
    for p, f in zip(pattern.args, fact.args):
        if is_var(p):
            if out.get(p, f) != f:
                return None
            out[p] = f
        elif p != f:
            return None
    return out


@dataclass(frozen=True)
class Rule:
    name: str
    params: tuple  # ((var, type), ...) — types are unary facts, e.g. ("?o", "movable")
    pre: tuple = ()  # positive preconditions
    neg: tuple = ()  # negative preconditions (must be absent)
    add: tuple = ()
    delete: tuple = ()

    def ground(self, binding: dict[str, str]) -> "Action":
        args = tuple(binding[v] for v, _ in self.params)
        return Action(self, args, binding)


@dataclass(frozen=True)
class Action:
    rule: Rule
    args: tuple
    binding: dict = field(compare=False, hash=False)

    def __str__(self) -> str:
        return f"{self.rule.name}({', '.join(self.args)})"

    @property
    def pre(self) -> list[Prop]:
        return [substitute(p, self.binding) for p in self.rule.pre] + \
               [Prop(t, (self.binding[v],)) for v, t in self.rule.params]

    @property
    def neg(self) -> list[Prop]:
        return [substitute(p, self.binding) for p in self.rule.neg]

    @property
    def add(self) -> list[Prop]:
        return [substitute(p, self.binding) for p in self.rule.add]

    @property
    def delete(self) -> list[Prop]:
        return [substitute(p, self.binding) for p in self.rule.delete]


class State:
    """An immutable set of ground facts."""

    __slots__ = ("facts",)

    def __init__(self, facts: Iterable[Prop] = ()):
        self.facts = frozenset(facts)

    def __contains__(self, prop: Prop) -> bool:
        return prop in self.facts

    def __iter__(self):
        return iter(self.facts)

    def __len__(self) -> int:
        return len(self.facts)

    def holds(self, props: Iterable[Prop]) -> bool:
        return all(p in self.facts for p in props)

    def is_applicable(self, action: Action) -> bool:
        return self.holds(action.pre) and not any(n in self.facts for n in action.neg)

    def apply(self, action: Action) -> "State":
        if not self.is_applicable(action):
            missing = [str(p) for p in action.pre if p not in self.facts] + \
                      [f"not {n}" for n in action.neg if n in self.facts]
            raise ValueError(f"{action} not applicable: {missing}")
        return State((self.facts - set(action.delete)) | set(action.add))

    def entities_of(self, type_name: str) -> list[str]:
        return sorted(p.args[0] for p in self.facts if p.name == type_name and len(p.args) == 1)

    def query(self, name: str) -> list[Prop]:
        return sorted(p for p in self.facts if p.name == name)

    def to_strings(self) -> list[str]:
        return sorted(str(p) for p in self.facts)

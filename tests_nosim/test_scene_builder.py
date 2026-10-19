# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from robolab.scene_gen.logic import P
from robolab.scene_gen.scene_builder import Rejected, build_scene, runtime_in
from robolab.scene_gen.templates import load_templates

TEMPLATES = {t.name: t for t in load_templates()}


def _first_scene(catalog, name, want=lambda s: True):
    for seed in range(40):
        try:
            s = build_scene(TEMPLATES[name], catalog, seed)
        except Rejected:
            continue
        if want(s):
            return s
    pytest.fail(f"no scene for {name}")


def test_in_relations_hold_under_the_runtime_hull_model(catalog):
    s = _first_scene(catalog, "fruit_bowl", lambda s: s.relations)
    for prim, (rel, host) in s.relations.items():
        assert rel == "in" and P("in", prim, host) in s.state
        assert runtime_in(s.placements[prim], s.objects[prim], s.placements[host], s.objects[host])
        assert s.placements[prim].com_world()[2] < s.placements[host].z + s.objects[host].geom.cavity["rim_z"]


def test_build_is_deterministic(catalog):
    a = build_scene(TEMPLATES["pantry_bins"], catalog, 3)
    b = build_scene(TEMPLATES["pantry_bins"], catalog, 3)
    assert {k: v.to_json() for k, v in a.placements.items()} == {k: v.to_json() for k, v in b.placements.items()}


def test_lying_slots_are_not_upright(catalog):
    s = _first_scene(catalog, "mugs_reorient")
    for prim in s.slots["mug_lying"]:
        assert not s.placements[prim].upright and P("upright", prim) not in s.state


def test_table_objects_rest_on_the_table_only(catalog):
    s = _first_scene(catalog, "desk_clutter_bin")
    for prim, p in s.placements.items():
        if prim not in s.relations:
            assert p.supports == {"table"}

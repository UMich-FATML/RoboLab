# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import pytest

from robolab.scene_gen.templates import Slot, Template, load_templates, sample_objects


def test_all_templates_load():
    names = {t.name for t in load_templates()}
    assert {"fruit_bowl", "pantry_bins", "blocks_stacking", "breakfast_plate", "desk_clutter_bin",
            "mugs_reorient"} <= names


def test_sampling_is_deterministic(catalog):
    t = next(t for t in load_templates() if t.name == "fruit_bowl")
    a = [(s.prim, s.pose_index) for s in sample_objects(t, catalog, random.Random(7))]
    b = [(s.prim, s.pose_index) for s in sample_objects(t, catalog, random.Random(7))]
    assert a == b and len({p for p, _ in a}) == len(a)


def test_hosted_slots_only_draw_graspable_objects(catalog):
    t = next(t for t in load_templates() if t.name == "fruit_bowl")
    for seed in range(10):
        for s in sample_objects(t, catalog, random.Random(seed)):
            if s.slot.relation != "table":
                assert s.obj.graspable


def test_invalid_template_rejected():
    with pytest.raises(ValueError):
        Template("bad", "", [Slot("a", {}, (1, 1), placement="in:missing")], ["put_in"]).validate()
    with pytest.raises(ValueError):
        Template("bad", "", [Slot("a", {}, (1, 1))], ["juggle"]).validate()

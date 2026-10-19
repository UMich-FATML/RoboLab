# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mesh-derived geometry: the grounding the scene generator relies on."""

import pytest

from robolab.scene_gen.placer import HeightField
from robolab.scene_gen.scene_builder import affordance


@pytest.mark.parametrize("name, settled_z", [
    ("wooden_bowl", 0.0028),  # origin at the bottom (hot3d) — assets/scenes/cooking_table.usda
    ("bowl", 0.0303),  # origin at the center (ycb) — assets/scenes/banana_bowl.usda
])
def test_rest_height_matches_physics_settled_benchmark_scenes(catalog, name, settled_z):
    g = catalog.objects[name].geom
    d = HeightField().drop(g, 0, 0.55, 0.0, 0.0)
    assert d.supports == {"table"} and d.stable
    assert abs(d.z - settled_z) < 0.005  # bbox recipe z=dims/2 is off by 6.5 cm for wooden_bowl


def test_roles_come_from_the_mesh(catalog):
    o = catalog.objects
    assert o["wooden_bowl"].is_container and o["bin_a04"].is_container and o["mug"].is_container
    assert o["plate_large"].is_support and not o["plate_large"].is_container
    assert not o["wooden_bowl"].is_support  # a rim is not a support surface
    assert not o["orange_01"].is_container and not o["orange_01"].is_support


def test_graspability(catalog):
    o = catalog.objects
    assert o["banana"].graspable  # thin axis 3.7 cm
    assert o["mug"].graspable  # open container grasped by the wall
    assert not o["wooden_bowl"].graspable and not o["cubebox_a02"].graspable


def test_stable_poses(catalog):
    mug = catalog.objects["mug"].geom
    assert mug.poses[0].upright and any(not p.upright for p in mug.poses[1:])
    glasses = catalog.objects["glasses"].geom
    assert 0 not in glasses.table_poses() and glasses.table_poses()  # authored pose not robust
    remote = catalog.objects["remote_control"].geom
    assert 0 not in remote.table_poses()  # standing on end: stable but tippy


def test_affordances(catalog):
    o = catalog.objects
    assert affordance(o["lemon_02"], o["bowl"], "in")
    assert not affordance(o["banana"], o["bowl"], "in")
    assert not affordance(o["banana"], o["mug"], "in")  # bridges the rim; COM stays above it
    assert affordance(o["red_block"], o["plate_large"], "on")

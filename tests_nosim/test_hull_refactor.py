# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""read_prim_mesh_points (shared by runtime predicates and scene_gen) equals the original per-point loop."""

import numpy as np
from pxr import Gf, Usd, UsdGeom

from robolab.constants import SCENE_DIR
from robolab.core.task.hull_check import read_prim_mesh_points


def _reference(prim):
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    inv = xc.GetLocalToWorldTransform(prim).RemoveScaleShear().GetInverse()
    out = []
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not child.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(child)
        if mesh.GetPurposeAttr().Get() not in (None, UsdGeom.Tokens.default_):
            continue
        m = xc.GetLocalToWorldTransform(child)
        for p in mesh.GetPointsAttr().Get() or []:
            q = inv.Transform(Gf.Vec3d(m.Transform(p)))
            out.append([q[0], q[1], q[2]])
    return np.asarray(out)


def test_matches_reference_on_posed_scene_objects():
    stage = Usd.Stage.Open(f"{SCENE_DIR}/banana_bowl.usda")
    for name in ("bowl", "banana"):
        prim = stage.GetPrimAtPath(f"/world/{name}")
        np.testing.assert_allclose(read_prim_mesh_points(prim), _reference(prim), atol=1e-12)

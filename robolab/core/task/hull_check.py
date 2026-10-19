# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convex-hull containment primitive (pure torch).

A convex polytope is the intersection of half-spaces ``{x : n·x + d <= 0}``
over its outward face equations. Dropping a face from the input plane set
yields an unbounded polytope along that face's normal — used to model
"open-top" containers, where the rim cap is removed so the polytope extends
infinitely along the container's local opening direction.
"""

from typing import NamedTuple

import numpy as np
import torch


class LocalHull(NamedTuple):
    """Convex hull of a body's mesh, expressed in body-local frame.

    All tensors live on the same device (typically GPU). Built once per body
    at scene-load and cached; never recomputed.

    Fields:
        vertices: (V, 3) — convex hull vertices, used on the *object* side of a
            containment test (object's points transformed into container-local
            frame and tested against the container's planes).
        centroid: (3,) — mean of ``vertices``. Precomputed so the per-step
            ``_obj_centroid_in_container`` doesn't re-reduce V verts each call.
        planes_full: (F, 4) — closed-hull outward face equations, used by
            ``enclosed`` semantics where the object must be fully bounded.
        planes_open_top: (F_kept, 4) — top faces dropped via ``open_top_planes``,
            used by ``inside`` / ``outside_of`` for open-top containers where
            "inside" includes the air column above the rim.
    """
    vertices: torch.Tensor
    centroid: torch.Tensor
    planes_full: torch.Tensor
    planes_open_top: torch.Tensor


def build_local_hull(
    points_np: np.ndarray,
    device: torch.device | str = "cpu",
    open_top_threshold: float = 0.7,
) -> LocalHull:
    """Compute the convex hull of ``points_np`` and package as a ``LocalHull``.

    Pure function — no IsaacSim dependencies. ``scipy.spatial.ConvexHull``
    handles QuickHull; the heavy work happens here, once per body. Resulting
    tensors are pushed to ``device`` for downstream point-in-hull math.

    Args:
        points_np: (P, 3) array of mesh points in body-local frame.
        device: torch device for the returned tensors.
        open_top_threshold: passed to ``open_top_planes`` (axis=2).

    Returns:
        LocalHull with vertices, full planes, and open-top planes.

    Raises:
        scipy.spatial.qhull.QhullError if ``points_np`` is degenerate
        (coplanar, < 4 distinct points, etc.). Caller decides on fallback.
    """
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points_np)
    verts = torch.tensor(points_np[hull.vertices], dtype=torch.float32, device=device)
    centroid = verts.mean(dim=0)
    planes_full = torch.tensor(hull.equations, dtype=torch.float32, device=device)
    planes_ot = open_top_planes(planes_full, axis=2, threshold=open_top_threshold)
    return LocalHull(vertices=verts, centroid=centroid, planes_full=planes_full, planes_open_top=planes_ot)


def point_in_hull(points: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask over points lying inside the convex polytope.

    Args:
        points: shape ``(..., 3)``, in the same frame as ``planes``.
        planes: shape ``(F, 4)``, outward face equations ``(nx, ny, nz, d)``.
            A point ``x`` is inside iff ``n·x + d <= 0`` for every face.

    Returns:
        Boolean tensor of shape ``(...,)``. The closed half-space test
        includes the boundary (``<= 0``).
    """
    signed = points @ planes[:, :3].T + planes[:, 3]   # (..., F)
    return signed.amax(dim=-1) <= 0


def open_top_planes(planes: torch.Tensor, axis: int = 2, threshold: float = 0.7) -> torch.Tensor:
    """Drop faces whose outward normal projects ``>= threshold`` onto ``+axis``.

    With default ``axis=2`` and ``threshold=0.7``, this removes faces facing
    within ~45° of straight up — the rim cap of an open-top container in the
    standard +z-up authoring convention. The resulting polytope is unbounded
    along the container's local +z, modelling the "open air above the rim".

    Args:
        planes: shape ``(F, 4)``.
        axis: index of the local axis defining "up". Default 2 (z).
        threshold: minimum normal component to drop. Faces with
            ``planes[:, axis] >= threshold`` are removed.

    Returns:
        ``(F_kept, 4)`` plane subset. Caller is responsible for caching.
    """
    keep = planes[:, axis] < threshold
    return planes[keep]


def read_prim_mesh_points(prim) -> np.ndarray:
    """Concatenated mesh points under ``prim``, in the prim's rotated frame at world scale.

    Pure pxr (no IsaacSim): reads ``points`` from descendant ``UsdGeom.Mesh``
    prims (traversing instance proxies, default purpose only), composes each
    mesh's local-to-world transform, then undoes only the prim's rotation and
    translation — explicitly NOT its scale — so points stay in meters even for
    assets authored in other units.

    This is the single definition of "an object's geometry" shared by the
    runtime predicates (``predicate_logic._read_local_mesh_points``) and the
    offline scene generator (``robolab.scene_gen.geometry``).

    Raises:
        ValueError: if no mesh geometry exists under ``prim``.
    """
    from pxr import Gf, Usd, UsdGeom

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prim_xform = xform_cache.GetLocalToWorldTransform(prim)
    prim_world_to_local = prim_xform.RemoveScaleShear().GetInverse()

    chunks: list[np.ndarray] = []
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not child.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(child)
        purpose = mesh.GetPurposeAttr().Get()
        if purpose not in (None, UsdGeom.Tokens.default_):
            continue
        points = mesh.GetPointsAttr().Get()
        if points is None or len(points) == 0:
            continue
        mesh_to_prim = xform_cache.GetLocalToWorldTransform(child) * prim_world_to_local
        m = np.array(mesh_to_prim, dtype=np.float64)  # row-vector convention: p' = [p,1] @ m
        pts = np.asarray(points, dtype=np.float64)
        homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ m
        chunks.append(homo[:, :3] / homo[:, 3:4])
    if not chunks:
        raise ValueError(f"read_prim_mesh_points: no mesh geometry found under prim '{prim.GetPath()}'")
    return np.concatenate(chunks, axis=0)

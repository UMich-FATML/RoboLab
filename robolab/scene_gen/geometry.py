# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-asset geometry derived from the USD mesh — the grounding of scene generation.

Bounding boxes are not enough to author stable scenes: prim origins sit at the
bottom for some datasets and at the center for others, a bowl's "top" is a
hollow rim, and objects only rest in a discrete set of poses. This module reads
the actual mesh (the same reader the runtime predicates use,
``hull_check.read_prim_mesh_points``) and derives, per asset:

* **stable poses** — hull face clusters whose contact patch contains the COM;
  each pose carries top/bottom heightmaps of the posed object;
* **rest geometry** — the authored-frame bottom, so the rest height is correct
  for every origin convention;
* **cavity** — for open-top containers, the floor height, rim height and depth;
* **support surface** — the flat plateau other objects can rest on (plates,
  boxes), distinguished from rings (bowl rims) by fill ratio.

Placement then reduces to one operation (``placer.HeightField.drop``): rest an
object's bottom heightmap on the scene's top heightmap. Pure pxr + numpy +
scipy; no IsaacSim, runs on CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

CELL = 0.004  # heightmap resolution, meters
CONTACT_TOL = 0.002  # cells within this gap of the rest height count as contact
CAVITY_MIN_DEPTH = 0.025
SURFACE_MIN_AREA = 0.002  # m^2
GRIPPER_MAX_WIDTH = 0.075  # Robotiq 2F-85 stroke (85 mm) minus margin
CONTAINER_MAX_GRASP = 0.20  # open containers up to this footprint are grasped by the wall
UPRIGHT_COS = math.cos(math.radians(10.0))
GEOMETRY_VERSION = 2
MIN_TIP_RATIO = 0.2  # poses easier to tip than this are not authored on a surface


# ---------------------------------------------------------------------------
# Quaternion helpers — RoboLab contract is (w, x, y, z) (docs/frames.md).
# ---------------------------------------------------------------------------

def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_from_yaw(yaw_deg: float):
    h = math.radians(yaw_deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_matrix(m: np.ndarray):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    n = math.sqrt(sum(c * c for c in q))
    q = tuple(c / n for c in q)
    return q if q[0] >= 0 else tuple(-c for c in q)


def rotation_to_down(n: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector ``n`` to (0, 0, -1)."""
    target = np.array([0.0, 0.0, -1.0])
    n = n / np.linalg.norm(n)
    v = np.cross(n, target)
    c = float(np.dot(n, target))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


# ---------------------------------------------------------------------------
# Heightmaps
# ---------------------------------------------------------------------------

@dataclass
class HeightMap:
    """Top/bottom surface of a posed object on a regular grid in its own xy frame.

    ``top[i, j]`` / ``bot[i, j]`` are the max / min z over the cell centered at
    ``(x0 + (i + 0.5) * CELL, y0 + (j + 0.5) * CELL)``; NaN where the object has
    no geometry. Coordinates are relative to the prim origin.
    """
    x0: float
    y0: float
    top: np.ndarray
    bot: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        return ~np.isnan(self.top)

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        nx, ny = self.top.shape
        xs = self.x0 + (np.arange(nx) + 0.5) * CELL
        ys = self.y0 + (np.arange(ny) + 0.5) * CELL
        return np.meshgrid(xs, ys, indexing="ij")


def heightmap_from_points(pts: np.ndarray) -> HeightMap:
    x0 = math.floor(pts[:, 0].min() / CELL) * CELL - CELL
    y0 = math.floor(pts[:, 1].min() / CELL) * CELL - CELL
    nx = int(math.ceil((pts[:, 0].max() - x0) / CELL)) + 2
    ny = int(math.ceil((pts[:, 1].max() - y0) / CELL)) + 2
    ix = np.clip(((pts[:, 0] - x0) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - y0) / CELL).astype(int), 0, ny - 1)
    flat = ix * ny + iy
    top = np.full(nx * ny, -np.inf)
    bot = np.full(nx * ny, np.inf)
    np.maximum.at(top, flat, pts[:, 2])
    np.minimum.at(bot, flat, pts[:, 2])
    top[np.isinf(top)] = np.nan
    bot[np.isinf(bot)] = np.nan
    return HeightMap(x0, y0, top.reshape(nx, ny), bot.reshape(nx, ny))


# ---------------------------------------------------------------------------
# Mesh reading
# ---------------------------------------------------------------------------

def read_mesh_triangles(prim) -> tuple[np.ndarray, np.ndarray]:
    """(V, 3) points and (T, 3) triangle indices under ``prim``, in the prim frame.

    Same traversal and frame convention as ``hull_check.read_prim_mesh_points``
    (instance proxies, default purpose, prim rotation/translation removed,
    scale kept); additionally returns fan-triangulated faces for surface sampling.
    """
    from pxr import Usd, UsdGeom

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    to_prim = xc.GetLocalToWorldTransform(prim).RemoveScaleShear().GetInverse()
    verts, tris, offset = [], [], 0
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not child.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(child)
        if mesh.GetPurposeAttr().Get() not in (None, UsdGeom.Tokens.default_):
            continue
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or counts is None or indices is None:
            continue
        m = np.array(xc.GetLocalToWorldTransform(child) * to_prim, dtype=np.float64)
        p = np.asarray(points, dtype=np.float64)
        homo = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ m
        verts.append(homo[:, :3] / homo[:, 3:4])
        counts = np.asarray(counts)
        indices = np.asarray(indices)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for k in range(3, int(counts.max()) + 1):  # fan-triangulate faces of size >= k
            sel = counts >= k
            s = starts[sel]
            tris.append(np.stack([indices[s], indices[s + k - 2], indices[s + k - 1]], axis=1) + offset)
        offset += len(p)
    if not verts:
        raise ValueError(f"no mesh geometry under {prim.GetPath()}")
    return np.concatenate(verts), np.concatenate(tris).astype(np.int64)


def sample_surface(verts: np.ndarray, tris: np.ndarray, seed: int = 0) -> np.ndarray:
    """Area-weighted surface samples dense enough for CELL-resolution heightmaps."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.minimum(np.ceil(area / (CELL * CELL / 6.0)).astype(np.int64), 4000)
    rng = np.random.default_rng(seed)
    idx = np.repeat(np.arange(len(tris)), n)
    r1 = np.sqrt(rng.random(len(idx)))
    r2 = rng.random(len(idx))
    pts = (1 - r1)[:, None] * a[idx] + (r1 * (1 - r2))[:, None] * b[idx] + (r1 * r2)[:, None] * c[idx]
    return np.concatenate([verts, pts])


# ---------------------------------------------------------------------------
# Derived geometry
# ---------------------------------------------------------------------------

def hull_com(points: np.ndarray):
    """(hull, volume centroid) of the convex hull — uniform-density COM estimate."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    ref = points[hull.vertices].mean(axis=0)
    tet = points[hull.simplices]  # (F, 3, 3)
    vol = np.abs(np.einsum("ij,ij->i", tet[:, 0] - ref, np.cross(tet[:, 1] - ref, tet[:, 2] - ref))) / 6.0
    cent = (tet.sum(axis=1) + ref) / 4.0
    return hull, (cent * vol[:, None]).sum(axis=0) / vol.sum()


def support_margin(xs: np.ndarray, ys: np.ndarray, com_xy: np.ndarray) -> float:
    """Signed distance from the COM projection to the edge of the contact patch's hull.

    Each contact cell contributes its CELL-square corners. Positive = inside
    (statically stable); the value over the COM height is the tip-over margin.
    """
    from scipy.spatial import ConvexHull, QhullError

    if len(xs) == 0:
        return -1.0
    h = CELL / 2
    pts = np.stack([xs, ys], axis=1)
    corners = np.concatenate([pts + [sx * h, sy * h] for sx in (-1, 1) for sy in (-1, 1)])
    try:
        hull = ConvexHull(corners)
    except QhullError:
        return -1.0
    eq = hull.equations  # n·p + d <= 0 inside, n unit
    return float(-(eq[:, :2] @ com_xy + eq[:, 2]).max())


def contact_stable(hm: HeightMap, com_xy: np.ndarray) -> tuple[bool, float]:
    """(stable, margin) for the posed object resting on a flat plane at its lowest point.

    Contact cells are those whose bottom lies within CONTACT_TOL of the plane;
    a half-cell tolerance absorbs discretization.
    """
    gap = hm.bot - np.nanmin(hm.bot)
    contact = (gap <= CONTACT_TOL) & ~np.isnan(gap)
    X, Y = hm.cell_centers()
    m = support_margin(X[contact], Y[contact], com_xy)
    return m >= -CELL / 2, m


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def _fill_ratio(region: np.ndarray, hm: HeightMap) -> float:
    from scipy.spatial import ConvexHull, QhullError

    if region.sum() < 4:
        return 0.0
    X, Y = hm.cell_centers()
    pts = np.stack([X[region], Y[region]], axis=1)
    try:
        # Hull of cell *centers* under-counts by ~half a cell around the perimeter.
        hull_area = ConvexHull(pts).volume + 2 * CELL * CELL * math.sqrt(region.sum())
    except QhullError:
        return 0.0
    return float(region.sum() * CELL * CELL / max(hull_area, 1e-9))


def detect_cavity(hm: HeightMap) -> dict | None:
    """Open-top cavity in the upright top heightmap: an enclosed low region well below the rim."""
    from scipy import ndimage

    occ = hm.mask
    if occ.sum() < 20:
        return None
    rim_z = float(np.nanmax(hm.top))
    filled = ndimage.binary_fill_holes(occ)
    # A cavity floor may be missing from the top map only if it is open at the bottom; require geometry.
    low = occ & (hm.top < rim_z - CAVITY_MIN_DEPTH)
    comp = _largest_component(low)
    if comp.sum() * CELL * CELL < 0.12 * filled.sum() * CELL * CELL:
        return None
    # Walled: most of the component's perimeter must be bounded by walls, not the
    # footprint edge (handle cutouts in bin end-walls leak a little).
    boundary = filled & ~ndimage.binary_erosion(filled)
    perimeter = comp & ~ndimage.binary_erosion(comp)
    if (comp & boundary).sum() > 0.25 * perimeter.sum():
        return None
    floor_z = float(np.nanmin(hm.top[comp]))
    floor = comp & (hm.top <= floor_z + 0.015)
    return {
        "rim_z": rim_z,
        "floor_z": floor_z,
        "depth": rim_z - floor_z,
        "interior_area": float(comp.sum() * CELL * CELL),
        "floor_area": float(floor.sum() * CELL * CELL),
    }


def detect_support(hm: HeightMap) -> dict | None:
    """Highest flat, well-filled plateau of the top heightmap (box tops, plate interiors)."""
    occ = hm.mask
    foot = occ.sum() * CELL * CELL
    top = float(np.nanmax(hm.top))
    bottom = float(np.nanmin(hm.bot))
    z = top
    while z > bottom + 0.004 - 1e-9:  # plate interiors sit well below the rim
        band = occ & (hm.top <= z + 1e-9) & (hm.top >= z - 0.006)
        comp = _largest_component(band)
        area = comp.sum() * CELL * CELL
        if area >= SURFACE_MIN_AREA and area >= 0.35 * foot and _fill_ratio(comp, hm) >= 0.8:
            return {"z": float(np.nanmax(hm.top[comp])), "area": float(area), "fill": _fill_ratio(comp, hm)}
        z -= 0.002
    return None


@dataclass
class StablePose:
    quat_wxyz: tuple
    upright: bool
    stable: bool
    support_area: float
    com: list  # COM in the posed frame (prim origin relative)
    extents: list  # posed AABB size (x, y, z)
    tip_ratio: float = 0.0  # support margin / COM height above the contact plane
    hm: HeightMap | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in
                ("quat_wxyz", "upright", "stable", "support_area", "com", "extents", "tip_ratio")}


@dataclass
class AssetGeometry:
    name: str
    usd_path: str
    extents: list
    bottom_z: float
    top_z: float
    com: list
    min_dim: float
    graspable: bool
    rolls: bool
    poses: list  # list[StablePose]; poses[0] is the authored (upright) pose
    cavity: dict | None
    support: dict | None

    @property
    def is_container(self) -> bool:
        return self.cavity is not None

    def summary(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "poses"}
        d["poses"] = [p.to_json() for p in self.poses]
        return d

    def table_poses(self, min_tip_ratio: float = MIN_TIP_RATIO) -> list[int]:
        """Pose indices robust enough to author on a flat surface (stable, hard to tip)."""
        return [i for i, p in enumerate(self.poses) if p.stable and p.tip_ratio >= min_tip_ratio]


def _pose_from_points(pts: np.ndarray, R: np.ndarray, com: np.ndarray, upright: bool) -> StablePose:
    posed = pts @ R.T
    c = R @ com
    hm = heightmap_from_points(posed)
    stable, margin = contact_stable(hm, c[:2])
    contact = (hm.bot - np.nanmin(hm.bot)) <= CONTACT_TOL
    ext = posed.max(axis=0) - posed.min(axis=0)
    com_h = max(c[2] - posed[:, 2].min(), 1e-3)
    return StablePose(quat_from_matrix(R), upright, bool(stable), float(contact.sum() * CELL * CELL),
                      c.tolist(), ext.tolist(), float(max(margin, 0.0) / com_h), hm)


def compute_asset_geometry(name: str, usd_path: str, max_poses: int = 4) -> AssetGeometry:
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    prim = stage.GetDefaultPrim()
    if not prim:
        raise ValueError(f"{usd_path}: no default prim")
    verts, tris = read_mesh_triangles(prim)
    pts = sample_surface(verts, tris)
    hull, com = hull_com(verts if len(verts) >= 4 else pts)

    poses = [_pose_from_points(pts, np.eye(3), com, upright=True)]
    # Candidate resting faces: coplanar hull facet clusters, largest first.
    eq = hull.equations
    tri = verts[hull.simplices] if len(verts) >= 4 else pts[hull.simplices]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    order = np.argsort(-areas)
    used = np.zeros(len(eq), bool)
    clusters = []
    for i in order:
        if used[i]:
            continue
        same = (~used) & (eq[:, :3] @ eq[i, :3] > math.cos(math.radians(4))) & (np.abs(eq[:, 3] - eq[i, 3]) < 0.003)
        used |= same
        clusters.append((areas[same].sum(), eq[i, :3]))
    clusters.sort(key=lambda c: -c[0])
    for area, n in clusters[:12]:
        if area < 4e-4 or n @ np.array([0, 0, -1.0]) > UPRIGHT_COS:
            continue  # tiny face, or the authored bottom (already pose 0)
        p = _pose_from_points(pts, rotation_to_down(n), com, upright=False)
        if p.stable:
            poses.append(p)
        if len(poses) > max_poses:
            break

    up = poses[0]
    ext = (verts.max(axis=0) - verts.min(axis=0)).tolist()
    cavity = detect_cavity(up.hm)
    # Solid objects are grasped across their thinnest dimension; open containers
    # (mugs, small bowls, pails) across their wall/rim, so footprint size bounds them.
    graspable = min(ext) <= GRIPPER_MAX_WIDTH or (cavity is not None and max(ext[:2]) <= CONTAINER_MAX_GRASP)
    return AssetGeometry(
        name=name,
        usd_path=str(usd_path),
        extents=ext,
        bottom_z=float(verts[:, 2].min()),
        top_z=float(verts[:, 2].max()),
        com=com.tolist(),
        min_dim=float(min(ext)),
        graspable=bool(graspable),
        rolls=bool(up.support_area < 1e-4),
        poses=poses,
        cavity=cavity,
        support=None if cavity else detect_support(up.hm),
    )


# ---------------------------------------------------------------------------
# Geometry database (summary JSON committed, heightmaps cached locally)
# ---------------------------------------------------------------------------

def _cache_key(usd_path: str) -> str:
    st = os.stat(usd_path)
    raw = f"{GEOMETRY_VERSION}|{usd_path}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class GeometryDB:
    """Lazily computed, disk-cached ``AssetGeometry`` per catalog object."""

    def __init__(self, repo_root: str | Path, cache_dir: str | Path | None = None):
        self.root = Path(repo_root)
        self.cache_dir = Path(cache_dir) if cache_dir else self.root / "assets" / "objects" / ".scenegen_cache"
        self._mem: dict[str, AssetGeometry] = {}

    def get(self, name: str, usd_path: str) -> AssetGeometry:
        if name in self._mem:
            return self._mem[name]
        abspath = str(self.root / usd_path)
        key = _cache_key(abspath)
        npz = self.cache_dir / f"{name}.{key}.npz"
        if npz.exists():
            g = _load(npz)
        else:
            g = compute_asset_geometry(name, usd_path=abspath)
            g.usd_path = usd_path
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            _save(g, npz)
        self._mem[name] = g
        return g


def _save(g: AssetGeometry, path: Path) -> None:
    arrays = {}
    for i, p in enumerate(g.poses):
        arrays[f"top{i}"] = p.hm.top.astype(np.float32)
        arrays[f"bot{i}"] = p.hm.bot.astype(np.float32)
        arrays[f"origin{i}"] = np.array([p.hm.x0, p.hm.y0])
    arrays["summary"] = np.array(json.dumps(g.summary()))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _load(path: Path) -> AssetGeometry:
    z = np.load(path)
    d = json.loads(str(z["summary"]))
    poses = []
    for i, p in enumerate(d.pop("poses")):
        x0, y0 = z[f"origin{i}"]
        p["quat_wxyz"] = tuple(p["quat_wxyz"])
        poses.append(StablePose(**p, hm=HeightMap(float(x0), float(y0), z[f"top{i}"].astype(np.float64),
                                                  z[f"bot{i}"].astype(np.float64))))
    return AssetGeometry(**d, poses=poses)

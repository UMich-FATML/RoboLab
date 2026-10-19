# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounded placement: rest objects on a scene heightfield instead of dropping them.

The scene is a top-down heightfield over the table. Placing an object at
``(x, y, yaw)`` in one of its stable poses means lowering its bottom heightmap
until it touches the scene's top surface (``HeightField.drop``) — an exact
vertical-drop rest height for any geometry: the table, a bowl's floor, a plate's
dished interior, or another object. The drop also reports *what* it rests on
and whether the contact patch contains the COM (quasi-static stability), so
``in``/``on`` relations hold at authoring time and physics settle only has to
confirm them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from robolab.scene_gen.geometry import (
    CELL,
    CONTACT_TOL,
    AssetGeometry,
    quat_from_yaw,
    quat_mul,
    quat_to_matrix,
    support_margin,
)

TABLE_Z = 0.0
TABLE_ID = "table"
DROP_GAP = 0.003  # authored clearance above the rest height
# Resting in/on a host (curved bowl floor, small plate) must be harder to tip than on the table.
MIN_TIP_RATIO_ON_HOST = 0.15
# Spawn region for table-level objects (matches the scenegen skill's solver bounds).
TABLE_BOUNDS = (0.25, 0.85, -0.45, 0.45)
_GRID_BOUNDS = (0.10, 1.00, -0.60, 0.60)


@dataclass
class Placement:
    name: str
    geom: AssetGeometry
    pose_index: int
    x: float
    y: float
    z: float
    yaw: float
    supports: set = field(default_factory=set)
    settles: bool = False  # contained but not quasi-statically stable: physics will shift it

    @property
    def quat_wxyz(self):
        return quat_mul(quat_from_yaw(self.yaw), self.geom.poses[self.pose_index].quat_wxyz)

    @property
    def upright(self) -> bool:
        return self.geom.poses[self.pose_index].upright

    def bottom_z(self) -> float:
        hm = self.geom.poses[self.pose_index].hm
        return self.z + float(np.nanmin(hm.bot))

    def top_z(self) -> float:
        hm = self.geom.poses[self.pose_index].hm
        return self.z + float(np.nanmax(hm.top))

    def com_world(self) -> np.ndarray:
        c = np.array(self.geom.poses[self.pose_index].com)
        R = quat_to_matrix(quat_from_yaw(self.yaw))
        return R @ c + np.array([self.x, self.y, self.z])

    def to_json(self) -> dict:
        return {
            "x": round(self.x, 5), "y": round(self.y, 5), "z": round(self.z, 5), "yaw": round(self.yaw, 3),
            "pose": self.pose_index, "upright": self.upright,
            "quat_wxyz": [round(c, 6) for c in self.quat_wxyz],
            "supports": sorted(self.supports), "settles": self.settles,
        }


@dataclass
class DropResult:
    z: float
    supports: set
    stable: bool
    cells: tuple  # (ii, jj) world indices covered by the footprint
    tip_ratio: float = 0.0


class HeightField:
    """Scene top surface on a world grid, with the owner of each cell's top."""

    def __init__(self, table_bounds=TABLE_BOUNDS):
        x0, x1, y0, y1 = _GRID_BOUNDS
        self.x0, self.y0 = x0, y0
        self.nx = int(round((x1 - x0) / CELL))
        self.ny = int(round((y1 - y0) / CELL))
        self.top = np.full((self.nx, self.ny), -1.0)  # floor, far below the table
        self.owner = np.full((self.nx, self.ny), "", dtype=object)
        tx0, tx1, ty0, ty1 = table_bounds
        xs = self.x0 + (np.arange(self.nx) + 0.5) * CELL
        ys = self.y0 + (np.arange(self.ny) + 0.5) * CELL
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        on_table = (X >= tx0 - 0.1) & (X <= tx1 + 0.1) & (Y >= ty0 - 0.1) & (Y <= ty1 + 0.1)
        self.top[on_table] = TABLE_Z
        self.owner[on_table] = TABLE_ID
        self.X, self.Y = X, Y

    def _footprint(self, geom: AssetGeometry, pose_index: int, x: float, y: float, yaw: float):
        """World cells under the posed+yawed object and the object's local top/bot there."""
        hm = geom.poses[pose_index].hm
        nx, ny = hm.top.shape
        r = math.hypot(max(abs(hm.x0), abs(hm.x0 + nx * CELL)), max(abs(hm.y0), abs(hm.y0 + ny * CELL))) + CELL
        i0 = max(int((x - r - self.x0) / CELL), 0)
        i1 = min(int((x + r - self.x0) / CELL) + 1, self.nx)
        j0 = max(int((y - r - self.y0) / CELL), 0)
        j1 = min(int((y + r - self.y0) / CELL) + 1, self.ny)
        if i0 >= i1 or j0 >= j1:
            return None
        WX, WY = self.X[i0:i1, j0:j1] - x, self.Y[i0:i1, j0:j1] - y
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        lx, ly = c * WX + s * WY, -s * WX + c * WY  # world -> object local (inverse yaw)
        li = np.floor((lx - hm.x0) / CELL).astype(int)
        lj = np.floor((ly - hm.y0) / CELL).astype(int)
        valid = (li >= 0) & (li < nx) & (lj >= 0) & (lj < ny)
        li_c, lj_c = np.clip(li, 0, nx - 1), np.clip(lj, 0, ny - 1)
        bot = np.where(valid, hm.bot[li_c, lj_c], np.nan)
        top = np.where(valid, hm.top[li_c, lj_c], np.nan)
        sel = ~np.isnan(bot)
        if not sel.any():
            return None
        ii, jj = np.nonzero(sel)
        return ii + i0, jj + j0, bot[sel], top[sel]

    def drop(self, geom: AssetGeometry, pose_index: int, x: float, y: float, yaw: float) -> DropResult | None:
        fp = self._footprint(geom, pose_index, x, y, yaw)
        if fp is None:
            return None
        ii, jj, bot, _ = fp
        under = self.top[ii, jj]
        z = float(np.max(under - bot))
        gap = (bot + z) - under
        contact = gap <= CONTACT_TOL
        supports = set(self.owner[ii[contact], jj[contact]].tolist()) - {""}
        # Quasi-static stability: COM over the contact patch (grown by half a cell).
        com = np.array(geom.poses[pose_index].com)
        R = quat_to_matrix(quat_from_yaw(yaw))
        com_w = R @ com + np.array([x, y, 0.0])
        margin = support_margin(self.X[ii[contact], jj[contact]], self.Y[ii[contact], jj[contact]], com_w[:2])
        com_h = max(com[2] - float(np.min(bot)), 1e-3)
        return DropResult(z + DROP_GAP, supports, margin >= -CELL / 2, (ii, jj), margin / com_h)

    def stamp(self, p: Placement) -> None:
        fp = self._footprint(p.geom, p.pose_index, p.x, p.y, p.yaw)
        ii, jj, _, top = fp
        new = top + p.z
        upd = new > self.top[ii, jj]
        self.top[ii[upd], jj[upd]] = new[upd]
        self.owner[ii[upd], jj[upd]] = p.name

    def footprint_bounds(self, p: Placement):
        fp = self._footprint(p.geom, p.pose_index, p.x, p.y, p.yaw)
        ii, jj = fp[0], fp[1]
        return (self.X[ii, jj].min(), self.X[ii, jj].max(), self.Y[ii, jj].min(), self.Y[ii, jj].max())


def footprint_center_radius(geom: AssetGeometry, pose_index: int) -> tuple[float, float, float]:
    """Footprint center (posed frame) and bounding radius about it — for the 2D spatial solver."""
    hm = geom.poses[pose_index].hm
    X, Y = hm.cell_centers()
    m = hm.mask
    cx, cy = float(X[m].mean()), float(Y[m].mean())
    r = float(np.sqrt((X[m] - cx) ** 2 + (Y[m] - cy) ** 2).max()) + CELL
    return cx, cy, r


def _spiral(radius_x: float, radius_y: float, rng: random.Random, n_rings: int = 5):
    out = [(0.0, 0.0)]
    for k in range(1, n_rings + 1):
        f = k / n_rings
        count = 6 + 4 * k
        phase = rng.random() * 2 * math.pi
        out += [(radius_x * f * math.cos(phase + 2 * math.pi * i / count),
                 radius_y * f * math.sin(phase + 2 * math.pi * i / count)) for i in range(count)]
    return out


def place_relative(hf: HeightField, name: str, geom: AssetGeometry, pose_index: int, host: Placement,
                   relation: str, rng: random.Random, siblings: set | None = None, yaws=None) -> Placement | None:
    """Rest ``geom`` in (relation='in') or on (relation='on') the placed ``host``.

    Candidates spiral out from the host center over its extent. A candidate is
    accepted when the drop rests on the host and is quasi-statically stable (for
    'on' also hard to tip). For 'in' only, a second pass accepts contained but
    unstable spots — round fruit on a curved floor next to a sibling rolls but
    stays in — and marks them ``settles`` so settle statistics can separate them.
    'in' always requires the object's COM to sit below the rim.
    """
    hext = host.geom.poses[host.pose_index].extents
    rx, ry = hext[0] * 0.42, hext[1] * 0.42
    if yaws is None:
        yaws = [rng.uniform(0, 360) for _ in range(3)] + [0.0, 90.0]
    allowed = {host.name} | (siblings or set())
    c, s = math.cos(math.radians(host.yaw)), math.sin(math.radians(host.yaw))
    candidates = []
    for dx, dy in _spiral(rx, ry, rng, n_rings=7):
        wx, wy = host.x + c * dx - s * dy, host.y + s * dx + c * dy
        for yaw in yaws:
            d = hf.drop(geom, pose_index, wx, wy, yaw)
            if d is None or host.name not in d.supports or not d.supports <= allowed:
                continue
            p = Placement(name, geom, pose_index, wx, wy, d.z, yaw, d.supports)
            if relation == "in":
                # Inside means the mass is inside: a banana bridging a mug rim sags below
                # the rim but its COM stays above it.
                if p.com_world()[2] > host.z + host.geom.cavity["rim_z"] - 0.005:
                    continue
                if d.stable and d.supports == {host.name}:
                    return p
                candidates.append(p)
            elif d.stable and d.supports == {host.name} and d.tip_ratio >= MIN_TIP_RATIO_ON_HOST:
                return p
    if relation == "in" and candidates:
        candidates[0].settles = True
        return candidates[0]
    return None

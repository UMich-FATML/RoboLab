# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog for scene generation: object_catalog.json ⋈ semantic ontology ⋈ mesh geometry."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import yaml

from robolab.constants import ASSET_DIR, OBJECT_CATALOG_PATH, PACKAGE_DIR
from robolab.scene_gen.geometry import AssetGeometry, GeometryDB

log = logging.getLogger(__name__)
ONTOLOGY_PATH = Path(ASSET_DIR) / "objects" / "scenegen_ontology.yaml"


@dataclass
class CatalogObject:
    name: str
    usd_path: str  # repo-relative, e.g. assets/objects/ycb/banana.usd
    display: str
    category: str
    color: str
    geom: AssetGeometry

    @property
    def is_container(self) -> bool:
        return self.geom.cavity is not None

    @property
    def is_support(self) -> bool:
        return self.geom.support is not None

    @property
    def graspable(self) -> bool:
        return self.geom.graspable

    @property
    def size(self) -> str:
        m = max(self.geom.extents)
        return "small" if m < 0.08 else ("large" if m > 0.2 else "medium")


class Catalog:
    """Objects the generator may sample: present in the ontology, loadable, trivially rooted."""

    def __init__(self, ontology_path: Path = ONTOLOGY_PATH, catalog_path: str = OBJECT_CATALOG_PATH,
                 geometry: GeometryDB | None = None):
        self.excluded: dict[str, str] = {}
        raw = json.load(open(catalog_path))
        first: dict[str, dict] = {}
        for entry in raw:
            if entry["name"] in first:
                self.excluded.setdefault(entry["name"] + "#dup", "duplicate catalog name (kept first)")
                continue
            first[entry["name"]] = entry
        self._raw = first
        self.ontology = yaml.safe_load(open(ontology_path))["objects"]
        self.geometry = geometry or GeometryDB(PACKAGE_DIR)

    @cached_property
    def objects(self) -> dict[str, CatalogObject]:
        out = {}
        for name, sem in self.ontology.items():
            entry = self._raw.get(name)
            if entry is None:
                self.excluded[name] = "not in object_catalog.json"
                continue
            reason = _root_problem(entry["usd_path"])
            if reason:
                self.excluded[name] = reason
                continue
            try:
                geom = self.geometry.get(name, entry["usd_path"])
            except Exception as e:  # noqa: BLE001 — any extraction failure excludes the asset
                self.excluded[name] = f"geometry extraction failed: {e}"
                continue
            out[name] = CatalogObject(name, entry["usd_path"], sem["display"], sem["category"], sem["color"], geom)
        if self.excluded:
            log.info("scene_gen catalog excluded %d objects: %s", len(self.excluded), self.excluded)
        return out

    def select(self, *, names=None, category=None, color=None, container=None, support=None,
               graspable=None) -> list[CatalogObject]:
        res = []
        for o in self.objects.values():
            if names is not None and o.name not in names:
                continue
            if category is not None and o.category not in ([category] if isinstance(category, str) else category):
                continue
            if color is not None and o.color != color:
                continue
            if container is not None and o.is_container != container:
                continue
            if support is not None and o.is_support != support:
                continue
            if graspable is not None and o.graspable != graspable:
                continue
            res.append(o)
        return sorted(res, key=lambda o: o.name)


def _root_problem(usd_path: str) -> str | None:
    """Scenes author translate/orient/scale on the referencing prim, which replaces the
    asset root's own xformOps — so only assets with an identity root, in meters, Z-up,
    can be placed faithfully."""
    from pxr import Usd, UsdGeom

    path = Path(PACKAGE_DIR) / usd_path
    if not path.exists():
        return "usd file missing"
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception:  # noqa: BLE001
        return "usd file unreadable"
    prim = stage.GetDefaultPrim()
    if not prim:
        return "no default prim"
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != "Z":
        return "not meters/Z-up"
    # The runtime spawns RigidObjectCfg + contact sensors at /scene/<name>: the asset
    # must be one rigid body on its root, or the env fails and settle moves a child.
    from pxr import UsdPhysics

    bodies = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if [p.GetPath() for p in bodies] != [prim.GetPath()]:
        return "rigid body not (only) on the root prim"
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        v = op.Get()
        if v is None:
            continue
        name = op.GetOpName()
        vals = [v.GetReal(), *v.GetImaginary()] if hasattr(v, "GetReal") else list(v)
        ident = [1.0, 0, 0, 0] if "orient" in name else ([1.0, 1, 1] if "scale" in name else [0.0] * len(vals))
        if any(abs(a - b) > 1e-4 for a, b in zip(vals, ident)):
            return f"non-identity root xformOp {name}"
    return None

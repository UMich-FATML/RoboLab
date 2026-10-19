# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab version compatibility layer.

IsaacLab 3.0 flips every quaternion convention from ``(w, x, y, z)`` to
``(x, y, z, w)``: the ``isaaclab.utils.math`` frame-transform helpers, the
``*_quat_w`` state tensors, and the config classes' ``rot`` fields (e.g.
``AssetBaseCfg.InitialStateCfg.rot`` and the camera/frame ``OffsetCfg.rot``
defaults changed from ``(1, 0, 0, 0)`` to ``(0, 0, 0, 1)``).

RoboLab's contract is ``(w, x, y, z)`` everywhere — authored configs, the
:class:`~robolab.core.world.world_state.WorldState` API, observations, and
every recorded channel (``docs/frames.md``; DROID is wxyz). This module is
the single place that knows which convention the installed isaaclab speaks
and converts at the boundary:

- :func:`quat_to_wxyz` / :func:`quat_from_wxyz`: tensor converters used at
  every runtime boundary (observations, recorders, WorldState, predicates).
  No-op on isaaclab 2.x.
- :func:`convert_env_cfg_quats`: rewrites RoboLab's authored-wxyz config
  ``rot`` fields into the installed stack's convention. Called by
  ``create_env`` before the environment is constructed. No-op on 2.x.
- :func:`convert_scene_state_quats`: rewrites ``scene.get_state()`` pose
  tensors back to wxyz for the states/initial-state recorders. No-op on 2.x.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

import torch


@lru_cache(maxsize=1)
def isaaclab_version() -> str | None:
    """Return the installed isaaclab distribution version, or ``None``.

    Read from package metadata (isaaclab has no ``__version__`` attribute)
    so this works before the app is launched.
    """
    try:
        return _dist_version("isaaclab")
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def isaaclab_major_version() -> int | None:
    """Return the installed isaaclab's major version, or ``None`` if absent."""
    v = isaaclab_version()
    if v is None:
        return None
    try:
        return int(v.split(".", 1)[0])
    except ValueError:
        return None


@lru_cache(maxsize=1)
def quats_are_xyzw() -> bool:
    """Return ``True`` when the installed isaaclab speaks ``(x, y, z, w)``.

    IsaacLab 3.0 flipped every quaternion convention from ``(w, x, y, z)`` to
    ``(x, y, z, w)`` — math helpers, state tensors, and config ``rot`` fields.
    Absent or unparsable versions default to the 2.x convention.
    """
    major = isaaclab_major_version()
    return major is not None and major >= 3


def as_torch(value):
    """Return an isaaclab data buffer as a torch tensor.

    isaaclab 3.0 hands out warp ``ProxyArray`` wrappers for asset data
    buffers. Indexing and torch methods work through a deprecation bridge,
    but isaaclab's torch-scripted math helpers (e.g.
    ``subtract_frame_transforms`` → ``quat_inv``) reject ProxyArray inputs
    outright, so buffers must be converted before being passed to them.
    isaaclab 2.x returns torch tensors (or raw warp arrays on 2.3) directly.
    """
    if isinstance(value, torch.Tensor):
        return value
    torch_view = getattr(value, "torch", None)  # isaaclab 3.0 ProxyArray
    if torch_view is not None:
        return torch_view
    import warp as wp

    return wp.to_torch(value)  # isaaclab 2.3 raw warp arrays


def quat_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert a runtime-convention quaternion tensor to RoboLab's ``(w, x, y, z)``.

    No-op when the installed isaaclab already speaks wxyz (isaaclab 2.x);
    on isaaclab 3.0 the ``(x, y, z, w)`` columns are rolled into wxyz.
    """
    if not quats_are_xyzw():
        return quat
    return torch.cat([quat[..., 3:4], quat[..., 0:3]], dim=-1)


def quat_from_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert a ``(w, x, y, z)`` quaternion tensor to the runtime convention.

    Inverse of :func:`quat_to_wxyz`; no-op on isaaclab 2.x.
    """
    if not quats_are_xyzw():
        return quat
    return torch.cat([quat[..., 1:4], quat[..., 0:1]], dim=-1)


@lru_cache(maxsize=1)
def articulation_base() -> type:
    """The class live articulations are instances of on the installed stack.

    Same backend-factory split as :func:`contact_sensor_base`: isaaclab 3.0's
    ``Articulation`` is a factory whose constructed scene assets are backend
    implementations (e.g. ``isaaclab_physx``'s) subclassing
    ``BaseArticulation``, so ``isinstance`` against the facade matches nothing.
    """
    if (isaaclab_major_version() or 0) >= 3:
        from isaaclab.assets.articulation.base_articulation import BaseArticulation

        return BaseArticulation
    from isaaclab.assets import Articulation

    return Articulation


@lru_cache(maxsize=1)
def contact_sensor_base() -> type:
    """The class live contact sensors are instances of on the installed stack.

    isaaclab 3.0 turned ``ContactSensor`` into a physics-backend factory: the
    sensors ``InteractiveScene`` creates are backend implementations (e.g.
    ``isaaclab_physx``'s) that subclass ``BaseContactSensor``, not the
    ``isaaclab.sensors.ContactSensor`` facade that isaaclab 2.x instantiated
    directly. ``isinstance`` checks against the facade silently match nothing
    on 3.0, so match the common base there instead.
    """
    if (isaaclab_major_version() or 0) >= 3:
        from isaaclab.sensors.contact_sensor.base_contact_sensor import BaseContactSensor

        return BaseContactSensor
    from isaaclab.sensors import ContactSensor

    return ContactSensor


def contact_force_matrix(sensor) -> torch.Tensor:
    """The filtered contact-force matrix of a contact sensor, as a torch tensor.

    Shape ``(num_envs, num_sensor_bodies, num_filter_bodies, 3)`` on every
    stack. isaaclab 2.x exposes it as ``data.force_matrix_w``; 3.0 renamed the
    same PhysX buffer to ``data.normal_force_matrix_w`` and hands out a
    warp ``ProxyArray`` whose torch view is taken here.
    """
    data = sensor.data
    if (isaaclab_major_version() or 0) >= 3:
        return data.normal_force_matrix_w.torch
    return data.force_matrix_w


@lru_cache(maxsize=1)
def _quat_cfg_classes() -> tuple[type, ...]:
    """Config classes whose ``rot`` field RoboLab authors in wxyz.

    Resolved by ``isinstance`` so subclasses are covered: the articulation /
    rigid-object ``InitialStateCfg`` nest under ``AssetBaseCfg.InitialStateCfg``,
    and ``TiledCameraCfg.OffsetCfg`` is ``CameraCfg.OffsetCfg``.
    """
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors.camera import CameraCfg
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

    return (AssetBaseCfg.InitialStateCfg, CameraCfg.OffsetCfg, OffsetCfg)


def _roll_wxyz_tuple(rot) -> tuple:
    """Roll an authored ``(w, x, y, z)`` 4-sequence into ``(x, y, z, w)``."""
    vals = list(rot)
    return tuple(vals[1:] + vals[:1])


def _convert_dataclass_quats(obj, quat_cfgs: tuple[type, ...], _seen: set[int]) -> None:
    """Recursively convert authored wxyz ``rot`` fields in a cfg tree, in place."""
    if id(obj) in _seen:
        return
    _seen.add(id(obj))

    if isinstance(obj, quat_cfgs):
        rot = getattr(obj, "rot", None)
        if rot is not None and len(rot) == 4:
            obj.rot = _roll_wxyz_tuple(rot)

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            _convert_dataclass_quats(getattr(obj, field.name, None), quat_cfgs, _seen)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            _convert_dataclass_quats(item, quat_cfgs, _seen)
    elif isinstance(obj, dict):
        for value in obj.values():
            _convert_dataclass_quats(value, quat_cfgs, _seen)


def convert_env_cfg_quats(env_cfg) -> None:
    """Rewrite RoboLab's authored wxyz config quats into the installed convention.

    RoboLab authors every config quaternion in ``(w, x, y, z)`` (see the
    ``rot=`` sites in robots, variations, tasks, and scene fixtures). IsaacLab
    3.0 reads the same ``rot`` fields as ``(x, y, z, w)``, so every authored
    value must be rolled before the config reaches isaaclab — otherwise robots
    spawn flipped and cameras point the wrong way, with no error raised.

    Must run before the environment is constructed (``gym.make``); call from
    ``create_env``. No-op on isaaclab 2.x. Idempotent: the cfg is stamped on
    conversion, so handing the same cfg instance to ``create_env`` twice is
    safe (the roll is destructive and must not run twice).
    """
    if not quats_are_xyzw():
        return
    if getattr(env_cfg, "_robolab_quats_converted", False):
        return
    env_cfg._robolab_quats_converted = True
    _convert_dataclass_quats(env_cfg, _quat_cfg_classes(), set())


# ``scene.get_state()`` keys whose (N, 7) tensors carry a quaternion in
# columns 3:7 (position in 0:3).
_POSE_STATE_KEYS = ("root_pose", "segment_pose")


def convert_scene_state_quats(state: dict) -> dict:
    """Convert ``scene.get_state()`` pose tensors to ``(w, x, y, z)``, in place.

    On isaaclab 3.0 the (N, 7) ``root_pose`` / ``segment_pose`` tensors carry
    an ``(x, y, z, w)`` quaternion in columns 3:7, while RoboLab's recorded
    ``states/**`` and ``initial_state/**`` channels are wxyz (``docs/frames.md``).
    No-op on isaaclab 2.x.
    """
    if not quats_are_xyzw():
        return state

    def _walk(node: dict) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                _walk(value)
            elif key in _POSE_STATE_KEYS and isinstance(value, torch.Tensor) and value.shape[-1] == 7:
                node[key] = torch.cat([value[..., :3], quat_to_wxyz(value[..., 3:7])], dim=-1)

    _walk(state)
    return state

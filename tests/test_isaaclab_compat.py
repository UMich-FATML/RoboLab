# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the isaaclab version-compat layer.

``robolab.core.utils.isaaclab_compat`` holds the (w, x, y, z) <-> (x, y, z, w)
conversions that keep RoboLab's quaternion contract (docs/frames.md) on
isaaclab 3.x, which flipped the runtime convention. These tests run against
whichever stack is installed: on isaaclab 2.x every conversion must be a
no-op, on 3.x it must roll the components.
"""

import numpy as np
import torch

from robolab.core.utils.isaaclab_compat import (
    convert_env_cfg_quats,
    convert_scene_state_quats,
    isaaclab_version,
    quat_from_wxyz,
    quat_to_wxyz,
    quats_are_xyzw,
    sim_cfg_has_render,
)


def test_version_probe_matches_installed_convention():
    if isaaclab_version() is None:
        import pytest

        pytest.skip("isaaclab not installed")
    assert quats_are_xyzw() == (int(isaaclab_version().split(".")[0]) >= 3)


def test_sim_cfg_render_probe_matches_installed_stack():
    # SimulationCfg.render exists on isaaclab 2.x and was removed in 3.0.
    import pytest

    if isaaclab_version() is None:
        pytest.skip("isaaclab not installed")
    sim_cfg = pytest.importorskip("isaaclab.sim").SimulationCfg
    assert sim_cfg_has_render() == ("render" in sim_cfg.__dataclass_fields__)


def test_quat_converters_follow_installed_convention():
    wxyz = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    runtime = quat_from_wxyz(wxyz)
    if quats_are_xyzw():
        # (w, x, y, z) -> (x, y, z, w)
        torch.testing.assert_close(runtime, torch.tensor([[0.2, 0.3, 0.4, 0.1]]))
    else:
        torch.testing.assert_close(runtime, wxyz)
    # And back.
    torch.testing.assert_close(quat_to_wxyz(runtime), wxyz)


def test_quat_from_wxyz_feeds_isaaclab_math_correctly():
    """A RoboLab wxyz quaternion must rotate correctly through isaaclab math."""
    from isaaclab.utils.math import quat_apply

    # (w, x, y, z) yaw +90 deg maps (1, 0, 0) -> (0, 1, 0).
    yaw90 = torch.tensor([[float(np.sqrt(0.5)), 0.0, 0.0, float(np.sqrt(0.5))]])
    v = torch.tensor([[1.0, 0.0, 0.0]])
    out = quat_apply(quat_from_wxyz(yaw90), v)
    torch.testing.assert_close(out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=0.0)


def test_convert_env_cfg_quats_rolls_authored_wxyz():
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors.camera import TiledCameraCfg
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg as FrameOffsetCfg
    from isaaclab.utils import configclass

    @configclass
    class _EnvLikeCfg:
        """Cfg tree holding one instance of every quat-bearing config class."""

        asset = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/obj",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.1, 0.2, 0.3, 0.4)),
        )
        camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/cam",
            height=720,
            width=1280,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"
            ),
        )
        frame_offset = FrameOffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0))

    cfg = _EnvLikeCfg()
    convert_env_cfg_quats(cfg)

    if quats_are_xyzw():
        assert cfg.asset.init_state.rot == (0.2, 0.3, 0.4, 0.1)
        assert cfg.camera.offset.rot == (-0.195, 0.399, 0.805, -0.393)
        assert cfg.frame_offset.rot == (0.0, 0.0, 1.0, 0.0)
    else:
        assert cfg.asset.init_state.rot == (0.1, 0.2, 0.3, 0.4)
        assert cfg.camera.offset.rot == (-0.393, -0.195, 0.399, 0.805)
        assert cfg.frame_offset.rot == (0.0, 0.0, 0.0, 1.0)

    # Idempotent: a cfg handed to create_env twice must not roll twice.
    convert_env_cfg_quats(cfg)
    if quats_are_xyzw():
        assert cfg.asset.init_state.rot == (0.2, 0.3, 0.4, 0.1)
    else:
        assert cfg.asset.init_state.rot == (0.1, 0.2, 0.3, 0.4)


def test_convert_scene_state_quats():
    root_pose = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4]])
    joint_pos = torch.zeros(1, 2)
    state = {"articulation": {"robot": {"root_pose": root_pose, "joint_position": joint_pos}}}

    convert_scene_state_quats(state)

    converted = state["articulation"]["robot"]["root_pose"]
    # Position columns are untouched.
    torch.testing.assert_close(converted[0, :3], torch.tensor([1.0, 2.0, 3.0]))
    if quats_are_xyzw():
        torch.testing.assert_close(converted[0, 3:], torch.tensor([0.4, 0.1, 0.2, 0.3]))
    else:
        torch.testing.assert_close(converted[0, 3:], torch.tensor([0.1, 0.2, 0.3, 0.4]))
    # Non-pose tensors pass through.
    assert state["articulation"]["robot"]["joint_position"] is joint_pos

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera placement math of the screenshot helper (rendering itself needs a GPU)."""

import numpy as np
import pytest

from robolab.core.utils.render_utils import _look_at_xform


@pytest.mark.parametrize("eye, target", [
    ((-0.3, 0.3, 0.7), (0.5, 0.0, 0.0)),  # settle_scenes "angled" view
    ((0.5, 0.0, 1.0), (0.5, 0.0, 0.0)),  # straight down: the Z-up hint is degenerate
])
def test_camera_sits_at_eye_and_looks_at_target(eye, target):
    m = np.array(_look_at_xform(eye, target))  # USD row-vector convention: p' = p @ m
    np.testing.assert_allclose(m[3, :3], eye, atol=1e-9)
    forward = np.array([0.0, 0.0, -1.0]) @ m[:3, :3]  # USD cameras look down local -Z
    want = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    np.testing.assert_allclose(forward, want, atol=1e-9)
    if abs(want[2]) < 0.999:  # image "up" stays on the world-up side
        assert (np.array([0.0, 1.0, 0.0]) @ m[:3, :3])[2] > 0

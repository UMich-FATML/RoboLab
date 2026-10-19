# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Pytest configuration for RoboLab install-verification tests.

Boots Isaac Sim once at conftest load so test files can freely import
isaaclab/robolab modules at their top level. Auto-accepts the Omniverse
EULA so a fresh install works headless without any prompts.

Set ``ROBOLAB_TEST_DISABLE_CAMERAS=1`` to boot without cameras: on clusters
whose GPU driver cannot run the RTX renderer (IsaacSim 5.x needs an R580
driver; R590 segfaults in librtx.scenedb), this still runs every unit test,
while the two env-constructing tests skip (their scenes contain cameras).
"""

import os

# Accept the Omniverse EULA non-interactively. Must be set before any
# isaaclab import. setdefault → user can still override.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

# Read before the app is launched; tests import the same constant for skips.
CAMERAS_DISABLED = os.environ.get("ROBOLAB_TEST_DISABLE_CAMERAS", "0") == "1"

import cv2  # noqa: E402, F401  must be imported before isaaclab

from isaaclab.app import AppLauncher  # noqa: E402

# Launch Isaac Sim once for the whole pytest session. Carb's logger is told to
# only emit warnings or worse so install-verification output isn't drowned in
# Isaac Sim's [Info] startup/shutdown chatter.
_launcher = AppLauncher(
    headless=True,
    enable_cameras=not CAMERAS_DISABLED,
    carb_settings={
        "/log/level": "warn",
        "/log/outputStreamLevel": "warn",
        "/log/fileLogLevel": "warn",
    },
)
simulation_app = _launcher.app


def pytest_addoption(parser):
    parser.addoption(
        "--task",
        default="BananaInBowlTask",
        help="Task class name for test_run_empty (default: BananaInBowlTask). "
             "The test runs the first registered env matching this task.",
    )
    parser.addoption(
        "--env-name",
        default=None,
        help="Full registered env name for test_run_empty (e.g. BananaInBowlTaskHomeOffice). "
             "If set, overrides --task.",
    )


import pytest


@pytest.fixture
def task_arg(request):
    return request.config.getoption("--task")


@pytest.fixture
def env_name_arg(request):
    return request.config.getoption("--env-name")


def pytest_sessionfinish(session, exitstatus):
    """Close the Isaac Sim app cleanly so the pytest process exits 0."""
    try:
        simulation_app.close()
    except Exception:
        pass

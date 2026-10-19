# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests (no IsaacSim boot, no GPU). Needs pxr from any IsaacSim venv.

Unlike tests/conftest.py this does not launch the simulator, so the scene
generator (robolab/scene_gen/) can be tested on a login node:

    OMNI_KIT_ACCEPT_EULA=Y .venv-60/bin/pytest tests_nosim -q
"""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def catalog():
    from robolab.scene_gen.catalog import Catalog

    c = Catalog()
    _ = c.objects
    return c

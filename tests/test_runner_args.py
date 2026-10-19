# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Guard the eval runner's argparse wiring against AppLauncher collisions.

The per-policy runners build their parser as::

    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)

IsaacLab 2.x's ``add_app_launcher_args`` raises ``ValueError`` if the parser
already declares a field it owns (2.x owns e.g. ``rendering_mode``; 3.0
dropped that arg). This never shows up in the other tests because they drive
``AppLauncher`` directly rather than through the runner's parser — so the
no-collision property is asserted explicitly here, against whichever
isaaclab stack is installed.
"""

import argparse

from isaaclab.app import AppLauncher

from robolab.eval.runner import add_common_eval_args

# Dests RoboLab's eval runner owns and consumes in run_evaluation().
_ROBOLAB_DESTS = {"renderer", "rendering_type", "enable_gt_state"}


def _launcher_dests() -> set[str]:
    """Dests AppLauncher declares on the installed isaaclab."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--placeholder", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return {action.dest for action in parser._actions} - {"placeholder", "help"}


def test_common_eval_args_do_not_collide_with_app_launcher():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    # Must not raise: whichever fields AppLauncher owns on the installed
    # isaaclab, add_common_eval_args must not declare them.
    AppLauncher.add_app_launcher_args(parser)

    dests = {action.dest for action in parser._actions}
    # run_evaluation() consumes args.renderer and args.rendering_type, so both
    # of ours must be present alongside AppLauncher's fields.
    for dest in _ROBOLAB_DESTS:
        assert dest in dests, f"runner-owned dest '{dest}' missing from the combined parser"

    launcher_dests = _launcher_dests()
    # None of our dests may be shadowed or redeclared by AppLauncher.
    assert not launcher_dests & _ROBOLAB_DESTS

    # isaaclab-owned dests the runner relies on must coexist where the
    # installed isaaclab declares them: 2.x's AppLauncher owns
    # `rendering_mode` (which is why RoboLab's flag is named
    # `--rendering-type`); 3.0 removed the argument entirely.
    if "rendering_mode" in launcher_dests:
        assert "rendering_mode" in dests

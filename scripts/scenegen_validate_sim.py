# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Validate generated tasks inside IsaacSim (GPU). See docs/scene_gen_pipeline.md.

    python scripts/scenegen_validate_sim.py --junit-xml junit_scenegen.xml            # verify_task_valid, all
    python scripts/scenegen_validate_sim.py --episodes 3 --junit-xml junit_scenegen.xml  # + empty episodes

Runs the tasks under robolab/tasks/generated/ through the same checks as
tests/test_tasks_valid.py, then (optionally) registers that folder and runs a
no-policy episode for K tasks. Writes a JUnit XML because IsaacSim can
hard-exit before printing a summary (docs/running.md).
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--tasks-subfolder", default="generated")
parser.add_argument("--episodes", type=int, default=0)
parser.add_argument("--num-steps", type=int, default=5)
parser.add_argument("--junit-xml", type=Path, default=None)
parser.add_argument("--no-cameras", action="store_true")
args = parser.parse_args()

import cv2  # noqa: E402, F401  (must precede isaaclab, as in tests/conftest.py)
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=not args.no_cameras,
                  carb_settings={"/log/level": "warn", "/log/outputStreamLevel": "warn"}).app

from robolab.constants import TASK_DIR  # noqa: E402
from robolab.core.task.task import verify_task_valid  # noqa: E402
from robolab.core.task.task_utils import find_task_files, load_task_from_file  # noqa: E402

results: list[tuple[str, str, str | None, float]] = []  # (suite, name, error, seconds)


def flush_junit() -> None:
    if not args.junit_xml:
        return
    fails = sum(1 for r in results if r[2])
    out = [f'<?xml version="1.0" encoding="utf-8"?>\n<testsuites><testsuite name="scenegen" tests="{len(results)}" '
           f'failures="{fails}" skipped="0">']
    for suite, name, err, sec in results:
        out.append(f'<testcase classname="{suite}" name="{escape(name)}" time="{sec:.2f}">')
        if err:
            out.append(f'<failure message="{escape(err.splitlines()[0][:200])}">{escape(err)}</failure>')
        out.append("</testcase>")
    out.append("</testsuite></testsuites>\n")
    args.junit_xml.write_text("\n".join(out))


files = sorted(find_task_files(os.path.join(TASK_DIR, args.tasks_subfolder)))
classes = {}
for f in files:
    t0 = time.time()
    try:
        cls = load_task_from_file(f, allow_multiple=False)
        ok, err = verify_task_valid(cls)
        classes[f] = cls
        results.append(("verify_task_valid", Path(f).stem, None if ok else str(err), time.time() - t0))
    except Exception:
        results.append(("verify_task_valid", Path(f).stem, traceback.format_exc(), time.time() - t0))
    flush_junit()
print(f"verify_task_valid: {sum(1 for r in results if not r[2])}/{len(results)} passed", flush=True)

if args.episodes:
    # Same loop as tests/test_run_empty.py; additionally the success term must
    # not fire right after reset (a trivially satisfied goal is a generator bug).
    import torch  # noqa: E402
    from robolab.core.environments.factory import get_envs  # noqa: E402
    from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

    valid = [f for f in files if f in classes]
    picks = valid[:: max(1, len(valid) // args.episodes)][: args.episodes]
    names = [classes[f].__name__ for f in picks]
    # By file path: class-name lookup only searches DEFAULT_TASK_SUBFOLDERS (benchmark/).
    auto_register_droid_envs(task_dirs=[args.tasks_subfolder], task=list(picks))
    for name in names:
        t0 = time.time()
        try:
            env_name = get_envs(task=name)[0]
            env, env_cfg = create_env(env_name, num_envs=1, use_fabric=True)
            try:
                env.reset()
                actions = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
                for step in range(args.num_steps):
                    _, _, term, trunc, _ = env.step(actions)
                    if bool(term.any()):
                        raise AssertionError(f"episode terminated at step {step} with no policy — goal trivially true?")
                end_episode(env)
            finally:
                env.close()
            results.append(("empty_episode", name, None, time.time() - t0))
        except Exception:
            results.append(("empty_episode", name, traceback.format_exc(), time.time() - t0))
        flush_junit()
        print(f"empty episode {name}: {'ok' if not results[-1][2] else 'FAILED'}", flush=True)

flush_junit()
failed = [r for r in results if r[2]]
for r in failed:
    print(f"FAIL {r[0]} {r[1]}: {r[2].splitlines()[-1] if r[2] else ''}")
print(f"scenegen sim validation: {len(results) - len(failed)}/{len(results)} passed", flush=True)
# app.close() hard-exits with status 0 on isaaclab 3.0 (docs/running.md), which would
# hide failures from the batch job; the JUnit XML is already flushed, so exit directly.
sys.stdout.flush()
os._exit(1 if failed else 0)

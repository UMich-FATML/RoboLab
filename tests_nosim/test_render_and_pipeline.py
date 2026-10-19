# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import random
import shutil
from pathlib import Path

import pytest

from robolab.constants import SCENE_DIR, TASK_DIR
from robolab.scene_gen import pipeline
from robolab.scene_gen.quest import FAMILIES, NoGoal, sample_goal
from robolab.scene_gen.render_task import render_task_source, termination_objects
from robolab.scene_gen.scene_builder import Rejected, build_scene
from robolab.scene_gen.templates import load_templates


def test_every_family_renders_consistently(catalog):
    seen = set()
    for t in load_templates():
        for seed in range(12):
            try:
                scene = build_scene(t, catalog, seed)
            except Rejected:
                continue
            for fam in t.goals:
                if fam in seen:
                    continue
                try:
                    g = sample_goal(scene, random.Random(seed), families=[fam])
                except NoGoal:
                    continue
                src = render_task_source(scene, g)
                ast.parse(src)
                contact = set(sorted(scene.objects) + ["table"])
                assert termination_objects(src) <= contact
                seen.add(fam)
    assert seen == set(FAMILIES), f"families never rendered: {set(FAMILIES) - seen}"


def test_generate_and_validate_cpu_roundtrip():
    # Under assets/scenes/ so payload depth matches real output; '_' dirs are skipped by find_scene_file.
    scene_out = Path(SCENE_DIR) / "_pytest_scenegen"
    task_out = Path(TASK_DIR) / "_pytest_scenegen"
    for d in (scene_out, task_out):
        shutil.rmtree(d, ignore_errors=True)
    try:
        stats = pipeline.generate(3, seed=99, scene_out=scene_out, task_out=task_out, log=lambda *_: None)
        assert stats["live"] == 3
        assert pipeline.validate_cpu(scene_out, log=lambda *_: None) == []
        again = pipeline.generate(3, seed=99, scene_out=scene_out, task_out=task_out, log=lambda *_: None)
        assert again["attempts"] == 0  # idempotent: nothing to top up
    finally:
        for d in (scene_out, task_out):
            shutil.rmtree(d, ignore_errors=True)

#!/usr/bin/env python3
"""Structural lint for RoboLab's hard rules (AGENTS.md → Hard rules).

Checks the rules a grep can judge, so they are enforced rather than trusted
(docs/design.md, core belief 4):

  Rule 1  Only robolab/core/utils/isaaclab_compat.py branches on the installed
          isaaclab version. Elsewhere, call a capability helper from the compat
          layer (e.g. ``quats_are_xyzw()``, ``sim_cfg_has_render()``) instead of
          probing the version yourself.
  Rule 4  The slurm/ tree is local-only: no tracked file lives under it and no
          markdown links to it or names it as a backticked path.

Stdlib only — runs in the pre-commit hook (git config core.hooksPath .githooks).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPAT = "robolab/core/utils/isaaclab_compat.py"

# Code roots held to rule 1. tests/ is exempt: it verifies the compat layer
# against the installed version, which requires reading that version.
RULE1_ROOTS = ("robolab", "policies", "scripts", "examples", "dashboard", "analysis", "assets")
RULE1_RE = re.compile(
    r"\b(?:isaaclab_version|isaaclab_major_version)\s*\("
    r"|\bversion\(\s*[\"']isaaclab[\"']"
    r"|\bisaaclab\.__version__"
)
SLURM_LINK_RE = re.compile(r"\]\((?:\.\./|\./)*slurm(?:/|\))")
SLURM_TICK_RE = re.compile(r"`slurm/[^`]*`")

failures: list[str] = []


def tracked(*pathspecs: str) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", *pathspecs],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [p for p in out.split("\0") if p]


def check_version_branching() -> None:
    for rel in tracked(*(f"{root}/*.py" for root in RULE1_ROOTS)):
        if rel == COMPAT:
            continue
        path = REPO / rel
        if not path.exists():  # deleted in the working tree, not yet staged
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if RULE1_RE.search(line):
                failures.append(
                    f"{rel}:{lineno}: probes the isaaclab version outside {COMPAT} — "
                    f"add a capability helper there (see quats_are_xyzw / "
                    f"sim_cfg_has_render) and call it instead (AGENTS.md hard rule 1)"
                )


def check_slurm_local_only() -> None:
    for rel in tracked("slurm"):
        failures.append(
            f"{rel}: slurm/ is local-only and must not be committed — "
            f"`git rm --cached` it; put reusable cluster knowledge in "
            f"docs/running.md (AGENTS.md hard rule 4)"
        )
    for rel in tracked("*.md"):
        path = REPO / rel
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if SLURM_LINK_RE.search(line) or SLURM_TICK_RE.search(line):
                failures.append(
                    f"{rel}:{lineno}: links or backticks the local-only slurm/ tree, "
                    f"which does not exist in a fresh clone — describe it in plain "
                    f"words instead (AGENTS.md hard rule 4)"
                )


def main() -> int:
    check_version_branching()
    check_slurm_local_only()
    if failures:
        print(f"check_rules: {len(failures)} hard-rule violation(s):", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    print("check_rules: OK (hard rules 1 and 4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Docs & map lint for RoboLab.

Enforces the harness rules that keep the repo legible for agents
(docs/design.md, core belief 4: "The map is checked, not trusted"):

  1. Relative markdown links resolve to existing files (no dead pointers).
  2. Every docs/*.md is indexed from docs/README.md (no orphan docs).
  3. AGENTS.md exists, stays a map (<= 150 lines) and CLAUDE.md defers to it.
  4. Path-like tokens in AGENTS.md exist on disk (the map describes reality).
  5. Every plan has a Status line and is listed in docs/plans/README.md.

Stdlib only — runs anywhere, including the pre-commit hook
(git config core.hooksPath .githooks, see AGENTS.md → Toolchain).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAX_AGENTS_LINES = 150

LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
# Path-like backticked tokens in AGENTS.md: dirs/files under known roots.
AGENTS_PATH_RE = re.compile(
    r"`((?:docs|robolab|policies|tests|scripts|examples|dashboard|analysis|slurm)"
    r"(?:/[\w.\-]+)*/?)`"
)

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)


def md_files() -> list[Path]:
    files = [p for p in REPO.glob("*.md")]
    files += sorted(REPO.glob("docs/**/*.md"))
    return files


def check_links(files: list[Path]) -> None:
    """Rule 1: relative links must resolve (external URLs and pure #anchors
    are out of scope)."""
    for md in files:
        text = md.read_text(encoding="utf-8")
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part = target.split("#", 1)[0]
            if not path_part:  # same-file anchor
                continue
            resolved = (md.parent / path_part).resolve()
            if not resolved.exists():
                fail(
                    f"{md.relative_to(REPO)}: dead link '{target}' — fix the "
                    f"link or add the file. Docs are indexed in docs/README.md "
                    f"(rule: docs/design.md core belief 4)"
                )


def check_orphans() -> None:
    """Rule 2: docs/*.md (except the index itself) must be linked from
    docs/README.md, so the index covers the territory."""
    index = REPO / "docs" / "README.md"
    index_links = set(LINK_RE.findall(index.read_text(encoding="utf-8")))
    for md in sorted((REPO / "docs").glob("*.md")):
        if md.name == "README.md":
            continue
        if md.name not in index_links:
            fail(
                f"docs/{md.name} is not linked from docs/README.md — add it "
                f"under the right section so agents can find it "
                f"(rule: docs/design.md core belief 4)"
            )


def check_agents() -> None:
    """Rules 3+4: AGENTS.md is a short map; CLAUDE.md defers to it; the
    paths it names exist."""
    agents = REPO / "AGENTS.md"
    if not agents.exists():
        fail("AGENTS.md is missing — create the map (see harness-engineering: "
             "AGENTS.md is a map, not a manual)")
        return
    lines = agents.read_text(encoding="utf-8").splitlines()
    if len(lines) > MAX_AGENTS_LINES:
        fail(
            f"AGENTS.md is {len(lines)} lines (> {MAX_AGENTS_LINES}) — move "
            f"detail into the owning doc under docs/ and leave a pointer "
            f"(rule: AGENTS.md must stay a map; docs/design.md core belief 4)"
        )
    for token in AGENTS_PATH_RE.findall(agents.read_text(encoding="utf-8")):
        if not (REPO / token.rstrip("/")).exists():
            fail(
                f"AGENTS.md names '{token}' but it does not exist — fix the "
                f"path (the map must describe the real tree; "
                f"rule: docs/design.md core belief 4)"
            )
    claude = REPO / "CLAUDE.md"
    if claude.exists():
        content = claude.read_text(encoding="utf-8").strip()
        if "@AGENTS.md" not in content:
            fail("CLAUDE.md must defer to AGENTS.md — make it contain just "
                 "'@AGENTS.md' (rule: one map for all harnesses)")


def check_plans() -> None:
    """Rule 5: plans carry a Status and are listed in docs/plans/README.md."""
    plans_dir = REPO / "docs" / "plans"
    index = plans_dir / "README.md"
    if not plans_dir.exists():
        return  # plans/ is optional for the lint; AGENTS.md points when used
    index_links = set(LINK_RE.findall(index.read_text(encoding="utf-8")))
    for md in sorted(plans_dir.glob("*.md")):
        if md.name == "README.md":
            continue
        rel = md.name
        text = md.read_text(encoding="utf-8")
        if "**Status" not in text:
            fail(
                f"docs/plans/{rel} has no '**Status:' line — add one "
                f"(active | completed | blocked) at the top "
                f"(rule: docs/plans/README.md → Convention)"
            )
        if rel not in index_links:
            fail(
                f"docs/plans/{rel} is not listed in docs/plans/README.md — "
                f"add it under Active or Completed "
                f"(rule: docs/plans/README.md → Convention)"
            )


def main() -> int:
    files = md_files()
    check_links(files)
    check_orphans()
    check_agents()
    check_plans()
    if failures:
        print(f"check_docs: {len(failures)} problem(s) — the docs map is "
              f"broken:", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    print(f"check_docs: OK ({len(files)} markdown files, links and index "
          f"consistent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

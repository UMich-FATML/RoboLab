# Running & Verifying

How to run the suite and verify changes. The full suite needs a GPU with a
driver supported by the *installed* IsaacSim stack; nothing here needs a
display (all runs are headless).

## Per-stack venvs

The three IsaacSim stacks are mutually exclusive extras (see the README
install section). Keep one venv per stack — and note `isaac60` needs its venv
created with `uv venv --python 3.12` *before* `uv sync`, because IsaacSim 6.x
ships cp312-only wheels:

```bash
UV_PROJECT_ENVIRONMENT=.venv    uv sync --extra isaac50 --extra test
UV_PROJECT_ENVIRONMENT=.venv-51 uv sync --extra isaac51 --extra test
UV_PROJECT_ENVIRONMENT=.venv-60 uv venv --python 3.12 .venv-60 && \
UV_PROJECT_ENVIRONMENT=.venv-60 uv sync --extra isaac60 --extra test
```

## Before any GPU run

The lints and the CPU-only tests need no GPU and no IsaacSim boot; run them
first (the pre-commit hook runs the three lints):

```bash
uvx ruff check . && python3 scripts/check_docs.py && python3 scripts/check_rules.py
UV_PROJECT_ENVIRONMENT=.venv-60 uv run --extra test pytest tests_nosim/
```

`tests_nosim/` still needs a stack venv (it reads USD through `pxr`), but runs
on a login node in about 20 s.

## The suite

```bash
UV_PROJECT_ENVIRONMENT=.venv-60 uv run --extra test pytest tests/ --junit-xml=junit.xml
```

Always pass `--junit-xml`. IsaacSim's kit boot hard-exits the process on some
failures before pytest can print its failure summary — the XML report is then
the *only* place the failure lands. Read failures from the XML, not stdout:

```bash
python - <<'EOF'
import xml.etree.ElementTree as ET
s = ET.parse("junit.xml").getroot().find("testsuite") or ET.parse("junit.xml").getroot()
print("tests:", s.get("tests"), "failures:", s.get("failures"), "skipped:", s.get("skipped"))
for tc in s.iter("testcase"):
    if tc.find("failure") is not None:
        print("--- FAILED:", tc.get("name")); print((tc.find("failure").text or "")[-1500:])
EOF
```

## Camera-less smoke runs

Some GPU driver generations boot IsaacSim headless but fail when the RTX
renderer initializes (camera-enabled runs then crash at kit boot with a device
error, node-dependently). For such clusters, or for fast iteration that doesn't
need cameras:

```bash
ROBOLAB_TEST_DISABLE_CAMERAS=1 UV_PROJECT_ENVIRONMENT=.venv-51 \
  uv run --extra test pytest tests/ --junit-xml=junit.xml
```

The two env-constructing tests (`test_run_empty`, `test_frame_contract`) skip
in this mode; everything else still runs — registration, task definitions,
config conversion, the compat layer.

## Batch clusters (Slurm)

- Batch scripts live in a local slurm/ directory and are **local-only**
  (git-excluded, [AGENTS.md](../AGENTS.md) rule 4, enforced by
  `scripts/check_rules.py`). Keep them there; put reusable knowledge
  in this file instead.
- The suite needs a real GPU allocation — run it as a batch job, not on the
  login node. Submit with the project's venv path pinned via
  `UV_PROJECT_ENVIRONMENT`.
- Camera-enabled runs of an RTX-sensitive stack can pass on some nodes and
  crash at boot on others in the same cluster. When you see
  `ERROR_DEVICE_LOST`-style crashes at viewport/renderer init, pin the job to
  a node where the stack is known to boot (`#SBATCH --nodelist=…`) rather than
  debugging the code — it is a driver/node property, not a repo bug.
- `set -o pipefail` + `nvidia-smi | head` in a batch script dies on SIGPIPE;
  guard diagnostics with `|| cat >/dev/null`.
- When iterating on boot-path failures, a crash dump under the venv's
  `…/isaacsim/kit/data/…` tree names the failing module — cheaper than
  bisecting.

## What "verified" means

- Full suite on the stack you changed, `--junit-xml` read from the XML.
- If you touched the compat layer or anything on a 2.x path: full suite on an
  isaac 5.x stack too — non-regression is a merge condition
  ([design](design.md), core belief 3).
- Camera-less runs count only when cameras were impossible on the cluster
  (say so in the report); otherwise run the camera path.

# Plans

Execution plans are versioned artifacts: complex work gets a checked-in plan
with progress and decisions, so the reasoning survives the session that made
it. Simple changes don't need one — the commit body is enough.

## Convention

- One file per plan: `YYYY-MM-<slug>.md`.
- A `Status` line at the top: `active`, `completed`, or `blocked`.
- Update progress in place while working; when done, record the outcome
  (commits, verification numbers) and mark `completed`. Don't delete plans —
  they are the record of *how* the repo got here.

## Active plans

(none)

## Completed plans

- [2026-10 — Batch scene/task generation pipeline](2026-10-scene-gen-pipeline.md)
- [2026-10 — IsaacSim 6.1 / IsaacLab 3.0 support](2026-10-isaac60-support.md)

## Tech debt

Ranked by expected bite. Each item names the condition that triggers paying
it.

1. **isaaclab 3.0 is a release candidate** (`3.0.0rc1`). On the final release:
   re-pin the `isaac60` extra, re-run the full suite on both stacks, and
   re-check `isaaclab_compat.py` against the final API (the rc's factory
   pattern and ProxyArray semantics are the risky parts). Update the README's
   experimental note.
2. **ProxyArray deprecation bridge.** `as_torch()` already unwraps at the
   scripted-math boundaries, but plain indexing through `ProxyArray` (used in
   several data paths) only works via a deprecated bridge. Migrate remaining
   indexing sites to explicit `.torch` when 3.0 finalizes the API.
3. **Camera-enabled isaac 5.x runs are impossible on R590-driver clusters**
   (RTX renderer unsupported). Any change touching cameras or render settings
   can only be fully verified on a cluster with an older driver, or on the
   `isaac60` stack. Note this in verification reports rather than skipping
   silently.
4. **`slurm/` batch scripts are local-only.** The verification harness a
   cluster actually runs is not reviewable in-repo. If a second person starts
   running the suite on a different cluster, consider committing a
   cluster-generic template (without site names) here.
5. **Generated tasks are only logically solvable.** `scripts/scenegen.py` proves
   plans symbolically; nobody executes them. Before using generated tasks for
   benchmark claims, execute reference plans with a scripted IK pick-place and
   reject the ones that fail (see [batch generation](../scene_gen_pipeline.md)).
6. **`openpi-client` upstream pins `numpy<2`.** If upstream un-pins it, it can
   rejoin the `test`/`all` extras and `test_volo_interop.py`'s
   `importorskip` becomes unnecessary (see [design](../design.md)).

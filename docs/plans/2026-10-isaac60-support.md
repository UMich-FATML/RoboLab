# IsaacSim 6.1 / IsaacLab 3.0 Support

**Status: completed** (2026-10, branch `isaac60-support`, commits
`1881885..604a1fe`)

## Goal

Run RoboLab unmodified-in-contract on IsaacSim 6.1 / IsaacLab 3.0.0rc1
(Python 3.12), as a new installable stack alongside isaac 5.0/5.1, with the
full suite green on the new stack and non-regression proven on 2.x.

## Approach

All IsaacLab-version handling behind one new boundary module,
`robolab/core/utils/isaaclab_compat.py`; RoboLab's public contract (wxyz
quaternions per `docs/frames.md`, CLI flags, recorded data) unchanged.

## What 3.0 broke (discovered incrementally, verified on GPU)

| Break | Discovery | Fix |
|---|---|---|
| Quaternion order flips wxyz→xyzw everywhere (math helpers, state tensors, config `rot` fields) | design study of 3.0 sources | converters + recursive config walk + state-dict converter; applied at every boundary |
| `SimulationCfg.render` removed; no `rendering_modes/*.kit` presets shipped | first boot | vendored presets, carb settings written directly before `gym.make` |
| `ContactSensor`/`Articulation` become backend factories — `isinstance` matches nothing, sensor dict comes up empty | verify run: `available sensors: dict_keys([])` | `contact_sensor_base()`/`articulation_base()` resolve real base classes |
| `force_matrix_w` renamed `normal_force_matrix_w` | follow-on failure | `contact_force_matrix()` |
| Data buffers are warp `ProxyArray`; torch-*scripted* math rejects them (plain torch ops work via a deprecated bridge) | verify run: `quat_inv() … found type 'ProxyArray'` | `as_torch()` at the scripted-math call sites |
| `add_app_launcher_args` drops `rendering_mode` | test design | runner-args test probes the installed parser |

## Outcome

- `isaac60` extra (`isaaclab[isaacsim]==3.0.0rc1`, python 3.12 marker) in uv
  conflict group with the 5.x stacks — one lockfile, three venvs.
- Verification (H200 cluster): isaac60 full suite **227/227** (camera-enabled,
  includes recorded-episode frame contract); isaac51 camera-less
  non-regression **216 pass / 3 by-design skips / 0 fail**.
- `openpi-client` dropped from extras (numpy pin conflict) →
  `test_volo_interop` `importorskip`; Docker image installs it `--no-deps`.

## Follow-ups

Tracked in [tech debt](README.md): re-pin on 3.0 final; ProxyArray bridge
migration; R590 camera caveat.

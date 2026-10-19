# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendored IsaacLab realtime rendering-quality presets.

IsaacLab 2.x applies the realtime quality preset (``performance`` /
``balanced`` / ``quality``) from the ``apps/rendering_modes/*.kit`` files
shipped inside the isaaclab wheel. IsaacLab 3.0 removed those preset files
(render settings moved to per-camera ``IsaacRtxRendererCfg.global_settings``),
so RoboLab vendors the carb settings here to keep the ``--rendering-type``
CLI contract identical on every stack.

Values are taken from isaaclab 2.3.2.post1, the last stack that ships them.
The isaaclab 2.2.0 files are identical except ``rtx.indirectDiffuse.enabled``,
which 2.2 left enabled in the performance/balanced presets.

Keys are native carb setting paths. ``robolab.core.environments.runtime``
writes them directly through the carb settings interface on isaaclab 3.0;
on isaaclab 2.x IsaacLab's own ``RenderCfg.rendering_mode`` machinery loads
the same values from the wheel and this module is unused.
"""

RENDERING_MODE_PRESETS: dict[str, dict[str, bool | int | float]] = {
    "performance": {
        "/rtx/translucency/enabled": False,
        "/rtx/reflections/enabled": False,
        "/rtx/reflections/denoiser/enabled": False,
        "/rtx/directLighting/sampledLighting/denoisingTechnique": 0,
        "/rtx/directLighting/sampledLighting/enabled": False,
        "/rtx/sceneDb/ambientLightIntensity": 1.0,
        "/rtx/shadows/enabled": True,
        "/rtx/indirectDiffuse/enabled": False,
        "/rtx/indirectDiffuse/denoiser/enabled": False,
        "/rtx/domeLight/upperLowerStrategy": 3,
        "/rtx/ambientOcclusion/enabled": False,
        "/rtx/ambientOcclusion/denoiserMode": 1,
        "/rtx/raytracing/subpixel/mode": 0,
        "/rtx/raytracing/cached/enabled": False,
        # DLSS frame gen does not yet support tiled camera well
        "/rtx-transient/dlssg/enabled": False,
        "/rtx-transient/dldenoiser/enabled": False,
        # DLSS model: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
        "/rtx/post/dlss/execMode": 0,
        # Avoids replicator warning
        "/rtx/pathtracing/maxSamplesPerLaunch": 1000000,
        # Avoids silent trimming of tiles
        "/rtx/viewTile/limit": 1000000,
    },
    "balanced": {
        "/rtx/translucency/enabled": False,
        "/rtx/reflections/enabled": False,
        "/rtx/reflections/denoiser/enabled": True,
        "/rtx/directLighting/sampledLighting/enabled": True,
        "/rtx/sceneDb/ambientLightIntensity": 1.0,
        "/rtx/shadows/enabled": True,
        "/rtx/indirectDiffuse/enabled": False,
        "/rtx/indirectDiffuse/denoiser/enabled": True,
        "/rtx/ambientOcclusion/enabled": False,
        "/rtx/ambientOcclusion/denoiserMode": 1,
        "/rtx/raytracing/subpixel/mode": 0,
        "/rtx/raytracing/cached/enabled": True,
        # DLSS frame gen does not yet support tiled camera well
        "/rtx-transient/dlssg/enabled": False,
        "/rtx-transient/dldenoiser/enabled": True,
        # DLSS model: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
        "/rtx/post/dlss/execMode": 1,
        # Avoids replicator warning
        "/rtx/pathtracing/maxSamplesPerLaunch": 1000000,
        # Avoids silent trimming of tiles
        "/rtx/viewTile/limit": 1000000,
    },
    "quality": {
        "/rtx/translucency/enabled": True,
        "/rtx/reflections/enabled": True,
        "/rtx/reflections/denoiser/enabled": True,
        "/rtx/directLighting/sampledLighting/enabled": True,
        "/rtx/sceneDb/ambientLightIntensity": 1.0,
        "/rtx/shadows/enabled": True,
        "/rtx/indirectDiffuse/enabled": True,
        "/rtx/indirectDiffuse/denoiser/enabled": True,
        "/rtx/ambientOcclusion/enabled": True,
        "/rtx/ambientOcclusion/denoiserMode": 0,
        "/rtx/raytracing/subpixel/mode": 1,
        "/rtx/raytracing/cached/enabled": True,
        # DLSS frame gen does not yet support tiled camera well
        "/rtx-transient/dlssg/enabled": False,
        "/rtx-transient/dldenoiser/enabled": True,
        # DLSS model: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
        "/rtx/post/dlss/execMode": 2,
        # Avoids replicator warning
        "/rtx/pathtracing/maxSamplesPerLaunch": 1000000,
        # Avoids silent trimming of tiles
        "/rtx/viewTile/limit": 1000000,
    },
}

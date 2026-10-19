# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline screenshots of USD stages (scene/object thumbnails, settle previews).

Built only on APIs that exist unchanged on every supported IsaacSim stack:
``pxr`` for the helper camera/lights/ground, ``omni.usd`` to open the stage,
``omni.timeline`` + ``app.update()`` to step, and ``omni.replicator.core``'s
``rgb`` annotator to capture. The previous implementation used
``omni.isaac.core`` (deprecated-only on 5.1) and ``isaacsim.core.api`` /
``isaacsim.sensors.camera`` (deprecated-only on 6.1), so it broke on both.
Nothing here branches on the installed version (AGENTS.md rule 1).
"""

import os

import numpy as np
from PIL import Image

HELPER_ROOT = "/RenderHelpers"  # authored on the session layer, never saved into the scene


def _enable_extension(name: str) -> None:
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled(name):
        manager.set_extension_enabled_immediate(name, True)


def _look_at_xform(eye, target):
    """Camera-to-world matrix for a USD camera (looks down -Z, +Y up) at ``eye`` facing ``target``."""
    from pxr import Gf

    eye, target = Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target))
    forward = (target - eye).GetNormalized()
    up = Gf.Vec3d(0, 0, 1)  # RoboLab stages are Z-up
    if abs(Gf.Dot(forward, up)) > 0.999:  # looking straight down/up: any horizontal up works
        up = Gf.Vec3d(1, 0, 0)
    return Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse()


def _author_helpers(stage, resolution, camera_position, camera_target, focal_length,
                    horizontal_aperture, add_lighting, add_ground, ground_position):
    from pxr import Gf, Usd, UsdGeom, UsdLux

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Scope.Define(stage, HELPER_ROOT)
        camera = UsdGeom.Camera.Define(stage, f"{HELPER_ROOT}/camera")
        width, height = resolution
        camera.CreateFocalLengthAttr(float(focal_length))
        camera.CreateHorizontalApertureAttr(float(horizontal_aperture))
        camera.CreateVerticalApertureAttr(float(horizontal_aperture) * height / width)
        # The USD default near plane is 1 scene unit (1 m): it would clip a tabletop view.
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1.0e4))
        UsdGeom.Xformable(camera).MakeMatrixXform().Set(_look_at_xform(camera_position, camera_target))

        if add_lighting:
            distant = UsdLux.DistantLight.Define(stage, f"{HELPER_ROOT}/distant_light")
            distant.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            distant.CreateEnableColorTemperatureAttr(True)
            distant.CreateColorTemperatureAttr(7250.0)
            distant.CreateIntensityAttr(1.0)
            distant.CreateExposureAttr(10.0)
            distant.CreateAngleAttr(30.0)
            dome = UsdLux.DomeLight.Define(stage, f"{HELPER_ROOT}/dome_light")
            dome.CreateIntensityAttr(1.0)
            dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            dome.CreateEnableColorTemperatureAttr(True)
            dome.CreateColorTemperatureAttr(6150.0)
            dome.CreateExposureAttr(9.0)
            dome.CreateTextureFormatAttr("latlong")

        if add_ground:
            from pxr import UsdPhysics

            ground = UsdGeom.Mesh.Define(stage, f"{HELPER_ROOT}/ground")
            s, z = 25.0, float(ground_position)
            ground.CreatePointsAttr([(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)])
            ground.CreateFaceVertexCountsAttr([4])
            ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            ground.CreateNormalsAttr([(0, 0, 1)] * 4)
            ground.CreateDisplayColorAttr([(0.5, 0.5, 0.5)])
            UsdPhysics.CollisionAPI.Apply(ground.GetPrim())  # objects settle onto it, as before
    return f"{HELPER_ROOT}/camera"


def render_stage_frame(app,
                        usd_path,
                        output_dir=None,
                        skip_frames=100,
                        resolution=(640, 480),
                        add_lighting=True,
                        add_ground=False,
                        camera_position=None,
                        camera_target=None,
                        focal_length=None,
                        horizontal_aperture=None,
                        ground_position=-0.2,
                        simulate=True,
                ):
    """Render a frame from a USD stage file.

    Args:
        app: Isaac Sim SimulationApp instance
        usd_path: Path to the USD file to render
        output_dir: Directory to save the rendered image (default: current directory)
        skip_frames: Number of frames to step before the capture step (physics
            settling, shader compilation and RTX accumulation)
        resolution: Tuple of (width, height) for the rendered image
        add_lighting: Whether to add default scene lighting
        add_ground: Whether to add a ground plane (with collision)
        camera_position: Camera position as (x, y, z). If None, uses (2.5, 0, 1.5)
        camera_target: Camera target as (x, y, z). If None, uses (0.5, 0.0, 0.0)
        focal_length: Camera focal length. If None, uses 24.0 (IsaacSim PinholeCameraCfg default)
        horizontal_aperture: Camera horizontal aperture. If None, uses 20.955
        ground_position: Z-position of the ground plane
        simulate: Step physics while warming up (objects come to rest, as the
            previous World.step-based implementation did). False renders the
            authored poses exactly.

    Returns:
        str: Path to the saved image file

    Note:
        The camera, lights and ground are authored on the stage's session layer
        under ``/RenderHelpers`` with fixed intrinsics, so framing is consistent
        across files regardless of their embedded camera settings, and the scene
        file itself is never modified.
    """
    import omni.timeline
    import omni.usd

    _enable_extension("omni.replicator.core")
    import omni.replicator.core as rep

    if camera_position is None or camera_target is None:
        camera_position, camera_target = (2.5, 0, 1.5), (0.5, 0.0, 0.0)
    focal_length = 24.0 if focal_length is None else focal_length
    horizontal_aperture = 20.955 if horizontal_aperture is None else horizontal_aperture

    # Kit does not resolve relative paths against Python's cwd: a relative path
    # "opens" an empty stage and renders a blank frame. Always pass it absolute.
    usd_path = os.path.abspath(str(usd_path))
    context = omni.usd.get_context()
    if not context.open_stage(usd_path):
        raise RuntimeError(f"render_stage_frame: could not open {usd_path}")
    stage = context.get_stage()
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.GetChildren():
        context.close_stage()
        raise RuntimeError(f"render_stage_frame: {usd_path} opened as an empty stage")
    camera_path = _author_helpers(stage, resolution, camera_position, camera_target, focal_length,
                                  horizontal_aperture, add_lighting, add_ground, ground_position)

    render_product = rep.create.render_product(camera_path, tuple(resolution))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])
    timeline = omni.timeline.get_timeline_interface()
    try:
        if simulate:
            timeline.play()
        for _ in range(skip_frames):
            app.update()
        # Annotators only capture while the timeline plays or on an explicit step;
        # delta_time=0 captures without advancing physics, subframes let materials settle.
        rep.orchestrator.step(rt_subframes=8, delta_time=0.0)
        rgba = np.asarray(annotator.get_data())
    finally:
        timeline.stop()
        annotator.detach()
        render_product.destroy()
    if rgba.ndim != 3 or rgba.shape[0] == 0:
        context.close_stage()
        raise RuntimeError(f"render_stage_frame: renderer returned no image for {usd_path} (got shape {rgba.shape})")

    output_dir = output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(str(usd_path)))[0]}.png")
    Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(output_path)
    print(f"Image saved to: {output_path}")
    context.close_stage()
    return output_path

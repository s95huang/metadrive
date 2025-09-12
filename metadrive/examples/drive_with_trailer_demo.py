#!/usr/bin/env python
"""
Demo: drive with a kinematic, collidable trailer and visualize in RGB/Depth/Semantic/LiDAR and top-down.

Controls:
  - W/A/S/D to drive
  - Q to toggle perspective (third-person <-> top-down)
  - R to reset
  - ESC to quit
"""
import argparse
import os

from metadrive import MetaDriveEnv
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera


def main(onscreen=True):
    res = (512, 384)
    env = MetaDriveEnv(
        dict(
            use_render=onscreen,
            manual_control=True,
            show_logo=False,
            show_fps=False,
            # Make the trailer clearly visible in third-person view
            camera_dist=12.0,
            camera_height=3.0,
            num_scenarios=1,
            start_seed=1010,
            traffic_density=0.1,
            # Add trailers to some traffic vehicles as well
            traffic_trailer_kinematic=dict(
                probability=0.5,
                config=dict(
                    enabled=True,
                    length=3.0,
                    width=1.6,
                    height=1.2,
                    origin_to_hitch=[1.5, 0.0, 0.45],
                    hitch_offset_on_tractor=[-2.3, 0.0, 0.45],
                ),
            ),
            image_observation=False,
            # Interface supports up to 3 panels; keep it within limits
            interface_panel=["rgb_camera", "semantic", "dashboard"],
            sensors={
                "rgb_camera": (RGBCamera, *res),
                "depth_camera": (DepthCamera, *res),
                "semantic": (SemanticCamera, *res),
            },
            vehicle_config=dict(
                # visualization toggles
                show_navi_mark=False,
                show_line_to_navi_mark=False,
                show_lidar=True,
                image_source="main_camera",
                # trailer settings
                trailer_kinematic=dict(
                    enabled=True,
                    length=3.2,
                    width=1.7,
                    height=1.3,
                    origin_to_hitch=[1.6, 0.0, 0.45],
                    hitch_offset_on_tractor=[-2.4, 0.0, 0.45],
                    # path can be provided if you have a trailer asset under models/
                    # path=["trailer/vehicle.gltf", [1, 1, 1], [0, 0, 0], [0, 0, 0]],
                ),
            ),
        )
    )

    try:
        env.reset()
        # Simple idle loop; manual control handles driving.
        for _ in range(100000):
            _, _, tm, tc, _ = env.step([0.0, 0.0])
            if tm or tc:
                env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offscreen", action="store_true", help="Run without an onscreen window")
    args = parser.parse_args()
    # Hide terrain for CI speed if running in test contexts
    if os.getenv("METADRIVE_TEST_EXAMPLE"):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    main(onscreen=not args.offscreen)

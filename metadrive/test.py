# import matplotlib.pyplot as plt
# from metadrive import (
#     MultiAgentMetaDrive,
#     MultiAgentTollgateEnv,
#     MultiAgentBottleneckEnv,
#     MultiAgentIntersectionEnv,
#     MultiAgentRoundaboutEnv,
#     MultiAgentParkingLotEnv
# )

# env_classes = dict(
#     roundabout=MultiAgentRoundaboutEnv,
#     intersection=MultiAgentIntersectionEnv,
#     tollgate=MultiAgentTollgateEnv,
#     bottleneck=MultiAgentBottleneckEnv,
#     parkinglot=MultiAgentParkingLotEnv,
#     pgma=MultiAgentMetaDrive
# )

# fig, axs = plt.subplots(2, 3, figsize=(10, 6.5), dpi=200)
# plt.tight_layout(pad=-3)

# for i in range(2):
#     for j in range(3):
#         env = list(env_classes.values())[i*3+j]({"log_level":50})
#         env.reset(seed=0)
#         m = env.render(mode="topdown", 
#                        # get the overview of the scene
#                        film_size = (1000, 1000),
#                        screen_size = (1000, 1000),
#                        # set camer to map center
#                        camera_position=env.current_map.get_center_point(), 
#                        # auto determine the number of pixels for 1 meter 
#                        scaling=None,
#                        # do not pop window
#                        window=False)
#         ax = axs[i][j]
#         ax.imshow(m, cmap="bone")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         env.close()
# plt.show()

import numpy as np
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Locate NuScenes data
nuscenes_data = AssetLoader.file_path(AssetLoader.asset_path, "nuscenes", unix_style=False)

def run_real_env(reactive):
    env = ScenarioEnv(
        {
            "reactive_traffic": reactive,
            "data_directory": nuscenes_data,
            "start_scenario_index": 6,   # use scenario #6
            "num_scenarios": 1,
            "crash_vehicle_done": True,
            "log_level": 50,
        }
    )
    try:
        o, _ = env.reset(seed=6)
        for i in range(1, 150):
            o, r, tm, tc, info = env.step([.0, -1])
            env.render(
                mode="top_down",
                window=False,
                screen_record=True,
                camera_position=(0, 0),
                screen_size=(500, 400)
            )
        frames = env.top_down_renderer.screen_frames
    finally:
        env.close()
    return frames

# Run both versions
f_1 = run_real_env(False)
f_2 = run_real_env(True)

# Concatenate frames side by side
frames = [cv2.hconcat([f_1[i], f_2[i]]) for i in range(len(f_1))]

# Save GIF with matplotlib
fig = plt.figure()
im = plt.imshow(frames[0])

def update(frame):
    im.set_array(frame)
    return [im]

ani = animation.FuncAnimation(
    fig, update, frames=frames, interval=50, blit=True
)

ani.save("comparison.gif", writer="pillow", fps=20)
plt.close(fig)

print("GIF saved as comparison.gif")

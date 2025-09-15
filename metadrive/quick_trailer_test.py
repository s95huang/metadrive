#!/usr/bin/env python
"""Quick test to check if trailer spawn collision issue is fixed"""

from metadrive import MetaDriveEnv
import time

def test_trailer_fix():
    env = MetaDriveEnv(
        dict(
            use_render=False,
            manual_control=False,
            force_destroy=True,
            num_scenarios=1,
            start_seed=1010,
            traffic_density=0.2,
            traffic_trailer_kinematic=dict(
                probability=0.5,
                config=dict(enabled=True, length=3.0, width=1.6, height=1.2),
            ),
            vehicle_config=dict(
                trailer_kinematic=dict(enabled=True, length=3.2, width=1.7, height=1.3),
            ),
        )
    )

    try:
        print("Testing trailer spawn fix...")
        start_time = time.time()
        env.reset()
        
        # Run for a few steps to see if episode lasts longer
        for i in range(20):
            obs, reward, term, trunc, info = env.step([0.1, 0.0])
            if term or trunc:
                print(f"Episode ended at step {i+1}")
                break
        else:
            print(f"Episode lasted at least 20 steps - spawn fix likely working!")
        
        elapsed = time.time() - start_time
        print(f"Test completed in {elapsed:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    test_trailer_fix()
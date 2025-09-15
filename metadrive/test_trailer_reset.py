#!/usr/bin/env python
"""
Quick test to check trailer reset behavior
"""
from metadrive import MetaDriveEnv

def test_reset():
    env = MetaDriveEnv(
        dict(
            use_render=False,
            manual_control=False,
            force_destroy=True,
            num_scenarios=1,
            start_seed=1010,
            traffic_density=0.1,
            # Add trailers to some traffic vehicles
            traffic_trailer_kinematic=dict(
                probability=0.5,
                config=dict(
                    enabled=True,
                    length=3.0,
                    width=1.6,
                    height=1.2,
                ),
            ),
            vehicle_config=dict(
                # enable trailer on ego vehicle
                trailer_kinematic=dict(
                    enabled=True,
                    length=3.2,
                    width=1.7,
                    height=1.3,
                ),
            ),
        )
    )

    try:
        print("First reset...")
        env.reset()
        print("First reset successful")
        
        # Simulate a few steps
        for i in range(5):
            _, _, tm, tc, _ = env.step([0.0, 0.0])
            if tm or tc:
                break
        
        print("Second reset...")
        env.reset()
        print("Second reset successful")
        
        print("All tests passed!")
    except Exception as e:
        print(f"Reset failed: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    test_reset()
#!/usr/bin/env python
"""Test spawn collision fix for trailers"""

from metadrive import MetaDriveEnv

def test_spawn():
    env = MetaDriveEnv(
        dict(
            use_render=False,
            manual_control=False,
            force_destroy=True,
            num_scenarios=1,
            start_seed=1010,
            traffic_density=0.3,  # Higher density to test collision
            traffic_trailer_kinematic=dict(
                probability=0.8,
                config=dict(
                    enabled=True,
                    length=3.0,
                    width=1.6,
                    height=1.2,
                ),
            ),
            vehicle_config=dict(
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
        print("Testing spawn collision fix...")
        env.reset()
        
        # Take a few steps to see if episodes last longer
        episode_lengths = []
        for episode in range(5):
            obs = env.reset()
            steps = 0
            for step in range(100):  # Max 100 steps per episode
                action = [0.1, 0.0]  # Slow forward movement
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
            
            episode_lengths.append(steps)
            print(f"Episode {episode + 1}: {steps} steps")
        
        avg_length = sum(episode_lengths) / len(episode_lengths)
        print(f"Average episode length: {avg_length:.1f} steps")
        
        if avg_length > 30:
            print("✓ Spawn collision fix appears to be working!")
        else:
            print("✗ Episodes still ending too quickly - may need further adjustment")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    test_spawn()
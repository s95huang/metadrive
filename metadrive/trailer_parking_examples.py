#!/usr/bin/env python
"""
Collection of trailer parking examples and configurations for MetaDrive.

This file provides ready-to-use examples for different parking scenarios:
1. Perpendicular parking (backing into a space)
2. Parallel parking (alongside other vehicles)
3. Loading dock parking (precise positioning)
4. Multi-space parking lot (choosing between spaces)
"""

import numpy as np
import math
from simple_trailer_parking import SimpleTrailerParkingEnv


def perpendicular_parking_config():
    """Configuration for perpendicular parking (90-degree backing)"""
    return {
        "manual_control": True,
        "use_render": True,
        "parking_space": {
            "position": [45, 8],     # Right side of road
            "heading": 270,          # Facing left (backing in from road)
            "length": 12.0,
            "width": 3.5,
        },
        "success_distance": 2.0,
        "success_heading": 15.0,
        "vehicle_config": {
            "enable_reverse": True,
            "show_dest_mark": True,
            "trailer_kinematic": {
                "enabled": True,
                "length": 6.0,       # Medium trailer
                "width": 2.2,
                "height": 2.0,
                "origin_to_hitch": [3.0, 0.0, 0.8],
                "hitch_offset_on_tractor": [-2.5, 0.0, 0.8],
            }
        },
        "camera_dist": 25,
        "camera_height": 10,
    }


def parallel_parking_config():
    """Configuration for parallel parking (alongside road)"""
    return {
        "manual_control": True,
        "use_render": True,
        "parking_space": {
            "position": [60, 2],     # Alongside road
            "heading": 90,           # Same direction as road
            "length": 15.0,          # Longer space needed for parallel
            "width": 3.0,
        },
        "success_distance": 1.5,     # Tighter tolerance for parallel
        "success_heading": 10.0,     # Must be well aligned
        "vehicle_config": {
            "enable_reverse": True,
            "show_dest_mark": True,
            "trailer_kinematic": {
                "enabled": True,
                "length": 8.0,       # Long trailer for challenge
                "width": 2.2,
                "height": 2.5,
                "origin_to_hitch": [4.0, 0.0, 0.8],
                "hitch_offset_on_tractor": [-3.0, 0.0, 0.8],
            }
        },
        "camera_dist": 30,
        "camera_height": 12,
    }


def loading_dock_config():
    """Configuration for loading dock parking (high precision)"""
    return {
        "manual_control": True,
        "use_render": True,
        "parking_space": {
            "position": [50, -6],    # Loading dock position
            "heading": 0,            # Straight backing
            "length": 10.0,
            "width": 3.0,
        },
        "success_distance": 1.0,     # Very tight tolerance
        "success_heading": 5.0,      # Must be very straight
        "vehicle_config": {
            "enable_reverse": True,
            "show_dest_mark": True,
            "trailer_kinematic": {
                "enabled": True,
                "length": 12.0,      # Extra long trailer
                "width": 2.5,
                "height": 3.0,
                "origin_to_hitch": [6.0, 0.0, 1.0],
                "hitch_offset_on_tractor": [-3.5, 0.0, 1.0],
            }
        },
        "camera_dist": 35,
        "camera_height": 15,
        "horizon": 3000,  # More time for precision parking
    }


def easy_training_config():
    """Easy configuration for learning/training algorithms"""
    return {
        "manual_control": False,    # Good for AI training
        "use_render": True,
        "parking_space": {
            "position": [40, 6],
            "heading": 270,
            "length": 15.0,         # Extra large space
            "width": 4.5,
        },
        "success_distance": 3.0,    # Generous tolerance
        "success_heading": 30.0,
        "vehicle_config": {
            "enable_reverse": True,
            "show_dest_mark": True,
            "max_speed": 15,        # Slower for easier control
            "trailer_kinematic": {
                "enabled": True,
                "length": 4.0,      # Short trailer
                "width": 2.0,
                "height": 1.8,
                "origin_to_hitch": [2.0, 0.0, 0.7],
                "hitch_offset_on_tractor": [-2.0, 0.0, 0.7],
            }
        },
        "horizon": 2000,
    }


class MultiScenarioTrailerParking:
    """Manager for running different parking scenarios"""
    
    SCENARIOS = {
        "perpendicular": {
            "name": "Perpendicular Parking",
            "description": "Back the truck-trailer into a perpendicular parking space",
            "config": perpendicular_parking_config,
            "difficulty": "Medium"
        },
        "parallel": {
            "name": "Parallel Parking", 
            "description": "Park the truck-trailer parallel to the road",
            "config": parallel_parking_config,
            "difficulty": "Hard"
        },
        "loading_dock": {
            "name": "Loading Dock",
            "description": "Precisely position trailer at loading dock",
            "config": loading_dock_config,
            "difficulty": "Expert"
        },
        "easy_training": {
            "name": "Easy Training",
            "description": "Large space for learning basic maneuvers", 
            "config": easy_training_config,
            "difficulty": "Easy"
        }
    }
    
    def __init__(self, scenario_name: str = "perpendicular"):
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(self.SCENARIOS.keys())}")
        
        self.scenario = self.SCENARIOS[scenario_name]
        self.env = None
        
    def run_scenario(self):
        """Run the selected parking scenario"""
        scenario_info = self.scenario
        config = scenario_info["config"]()
        
        print(f"\n🚛 {scenario_info['name']} Challenge")
        print("=" * 50)
        print(f"Description: {scenario_info['description']}")
        print(f"Difficulty: {scenario_info['difficulty']}")
        print(f"Target position: {config['parking_space']['position']}")
        print(f"Target heading: {config['parking_space']['heading']}°")
        print(f"Success criteria:")
        print(f"  - Distance tolerance: {config['success_distance']}m")
        print(f"  - Heading tolerance: {config['success_heading']}°")
        
        if config["manual_control"]:
            print("\nControls: W/A/S/D to drive, R to reset, ESC to quit")
        else:
            print("\nRunning in automatic mode...")
        
        self.env = SimpleTrailerParkingEnv(config)
        
        try:
            obs = self.env.reset()
            episode = 1
            
            while True:
                print(f"\n--- Episode {episode} ---")
                step_count = 0
                best_performance = {"distance": float('inf'), "heading": 180.0}
                
                while True:
                    if config["manual_control"]:
                        action = [0, 0]  # Manual control
                    else:
                        # Simple AI for demo
                        action = self._get_ai_action()
                    
                    obs, reward, done, truncated, info = self.env.step(action)
                    step_count += 1
                    
                    # Track best performance this episode
                    current_dist = info.get("distance_to_parking", float('inf'))
                    current_heading = info.get("heading_error", 180.0)
                    
                    if current_dist < best_performance["distance"]:
                        best_performance["distance"] = current_dist
                        best_performance["heading"] = current_heading
                    
                    # Progress updates
                    if step_count % 100 == 0:
                        progress = info.get("parking_progress_percent", 0)
                        print(f"Step {step_count}: Progress={progress:.1f}%, "
                              f"Distance={current_dist:.2f}m, "
                              f"Heading error={current_heading:.1f}°")
                    
                    if done or truncated:
                        self._report_episode_results(episode, step_count, info, best_performance)
                        break
                
                # Reset for next episode
                obs = self.env.reset()
                episode += 1
                
                # For non-manual mode, limit episodes
                if not config["manual_control"] and episode > 5:
                    break
                    
        except KeyboardInterrupt:
            print(f"\n{scenario_info['name']} scenario interrupted by user")
        finally:
            if self.env:
                self.env.close()
    
    def _get_ai_action(self):
        """Simple AI for automated testing"""
        distance = self.env._get_distance_to_parking()
        heading_error = self.env._get_heading_error()
        
        # Simple approach strategy
        if distance > 15:
            # Far away - drive forward
            return [0.4, 0]
        elif distance > 5:
            # Medium distance - approach with steering correction
            steering = np.clip(heading_error / 45.0, -0.8, 0.8)
            return [0.2, steering * 0.3]
        else:
            # Close - precision maneuvering
            steering = np.clip(heading_error / 30.0, -1.0, 1.0)
            return [0.1, steering * 0.5]
    
    def _report_episode_results(self, episode, steps, info, best_performance):
        """Report episode results"""
        print(f"\n📊 Episode {episode} Results ({steps} steps):")
        
        if info.get("parking_success"):
            print("✅ SUCCESS!")
            print(f"  Final distance: {info['distance_to_parking']:.2f}m")
            print(f"  Final heading error: {info['heading_error']:.1f}°")
            print(f"  Efficiency: {100 - (steps/15):.1f}% (fewer steps is better)")
        else:
            print("❌ FAILED")
            print(f"  Best distance achieved: {best_performance['distance']:.2f}m")
            print(f"  Best heading: {best_performance['heading']:.1f}° error")
            
            # Specific failure reasons
            if not info.get('distance_criterion_met', True):
                required = info.get('required_distance', 'unknown')
                actual = info.get('distance_to_parking', 'unknown') 
                print(f"  ❌ Distance: {actual:.2f}m > {required}m required")
            if not info.get('heading_criterion_met', True):
                required = info.get('required_heading_accuracy', 'unknown')
                actual = info.get('heading_error', 'unknown')
                print(f"  ❌ Heading: {actual:.1f}° > {required}° required")
    
    @classmethod
    def list_scenarios(cls):
        """List all available scenarios"""
        print("\n🚛 Available Trailer Parking Scenarios:")
        print("=" * 45)
        
        for key, scenario in cls.SCENARIOS.items():
            print(f"\n{key}:")
            print(f"  Name: {scenario['name']}")
            print(f"  Description: {scenario['description']}")
            print(f"  Difficulty: {scenario['difficulty']}")


def main():
    """Main function to run parking scenarios"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python trailer_parking_examples.py <scenario>")
        MultiScenarioTrailerParking.list_scenarios()
        print("\nExample: python trailer_parking_examples.py perpendicular")
        return
    
    scenario_name = sys.argv[1]
    
    if scenario_name == "list":
        MultiScenarioTrailerParking.list_scenarios()
        return
    
    try:
        parking_test = MultiScenarioTrailerParking(scenario_name)
        parking_test.run_scenario()
    except ValueError as e:
        print(f"Error: {e}")
        MultiScenarioTrailerParking.list_scenarios()


if __name__ == "__main__":
    main()
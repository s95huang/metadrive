#!/usr/bin/env python
"""
Visual Trailer Parking Lot using MetaDrive's existing parking lot components.

This uses MetaDrive's built-in parking lot map and extends it for trailer parking.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
import random

from metadrive.envs.marl_envs.marl_parking_lot import MultiAgentParkingLotEnv
from metadrive import MetaDriveEnv


class VisualTrailerParkingEnv(MetaDriveEnv):
    """
    Trailer parking environment using MetaDrive's existing parking lot infrastructure.
    
    This creates a visual parking lot with proper spaces and markers.
    """
    
    @classmethod
    def default_config(cls) -> Dict:
        config = super().default_config()
        config.update({
            # Use parking lot map
            "map_config": {
                "type": "parking_lot",
                "parking_space_num": 12,  # Number of parking spaces
                "exit_length": 50,
            },
            
            # Environment settings
            "num_scenarios": 1,
            "traffic_density": 0.0,
            "start_seed": 1000,
            
            # Vehicle with trailer
            "vehicle_config": {
                "enable_reverse": True,
                "show_dest_mark": True,
                "show_line_to_dest": True,
                "trailer_kinematic": {
                    "enabled": True,
                    "length": 8.0,
                    "width": 2.4,
                    "height": 2.5,
                    "origin_to_hitch": [4.0, 0.0, 1.0],
                    "hitch_offset_on_tractor": [-3.2, 0.0, 1.0],
                }
            },
            
            # Success criteria
            "target_parking_space": 1,  # Which space to target (0-11)
            "position_tolerance": 2.5,
            "heading_tolerance": 20.0,
            "success_hold_time": 2.0,
            
            # Visual settings
            "use_render": True,
            "manual_control": True,
            "camera_height": 25,
            "camera_dist": 40,
            "show_fps": True,
            "show_logo": False,
            
            # Episode settings
            "horizon": 3000,
        })
        return config
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Parking state
        self.target_space_id = self.config["target_parking_space"]
        self.success_hold_timer = 0.0
        self.initial_distance = 0.0
        self.best_distance = float('inf')
        
        # Define parking space positions (approximate)
        # These are typical positions in MetaDrive's parking lot
        self.parking_spaces = self._define_parking_spaces()
    
    def _define_parking_spaces(self) -> List[Dict]:
        """Define approximate parking space positions and properties"""
        spaces = []
        
        # MetaDrive parking lot has spaces arranged in rows
        # These are approximate positions based on the standard layout
        base_positions = [
            # Right side spaces (reverse parking)
            {"id": 0, "pos": [75, 15], "heading": 270, "type": "reverse"},
            {"id": 1, "pos": [85, 15], "heading": 270, "type": "reverse"},  
            {"id": 2, "pos": [95, 15], "heading": 270, "type": "reverse"},
            {"id": 3, "pos": [105, 15], "heading": 270, "type": "reverse"},
            {"id": 4, "pos": [115, 15], "heading": 270, "type": "reverse"},
            {"id": 5, "pos": [125, 15], "heading": 270, "type": "reverse"},
            
            # Left side spaces (forward parking)
            {"id": 6, "pos": [75, -15], "heading": 90, "type": "forward"},
            {"id": 7, "pos": [85, -15], "heading": 90, "type": "forward"},
            {"id": 8, "pos": [95, -15], "heading": 90, "type": "forward"},
            {"id": 9, "pos": [105, -15], "heading": 90, "type": "forward"},
            {"id": 10, "pos": [115, -15], "heading": 90, "type": "forward"},
            {"id": 11, "pos": [125, -15], "heading": 90, "type": "forward"},
        ]
        
        for space_info in base_positions:
            space = {
                "id": space_info["id"],
                "position": space_info["pos"] + [0.3],  # Add z coordinate
                "heading": space_info["heading"],
                "type": space_info["type"],
                "length": 12.0,
                "width": 3.5,
                "difficulty": self._assign_difficulty(space_info["id"])
            }
            spaces.append(space)
        
        return spaces
    
    def _assign_difficulty(self, space_id: int) -> str:
        """Assign difficulty based on space position"""
        if space_id in [0, 5, 6, 11]:  # End spaces
            return "easy"
        elif space_id in [1, 4, 7, 10]:  # Near end
            return "medium"  
        else:  # Middle spaces
            return "hard"
    
    def reset(self, seed: Optional[int] = None):
        """Reset environment"""
        obs = super().reset(seed=seed)
        
        # Select target space (can be random or specified)
        if hasattr(self.config, "random_target") and self.config["random_target"]:
            self.target_space_id = random.randint(0, len(self.parking_spaces) - 1)
        
        self.target_space = self.parking_spaces[self.target_space_id]
        
        # Position vehicle at parking lot entrance
        self._position_vehicle_at_entrance()
        
        # Initialize metrics
        self.initial_distance = self._get_distance_to_target()
        self.best_distance = self.initial_distance
        self.success_hold_timer = 0.0
        
        self._print_episode_info()
        
        return obs
    
    def _position_vehicle_at_entrance(self):
        """Position vehicle at parking lot entrance"""
        # Position vehicle at entrance
        entrance_x = 20.0
        entrance_y = 0.0
        entrance_heading = 0.0
        
        self.vehicle.set_position([entrance_x, entrance_y, 0.3])
        self.vehicle.set_heading_theta(math.radians(entrance_heading))
    
    def step(self, action):
        """Step with parking lot logic"""
        obs, reward, done, truncated, info = super().step(action)
        
        # Calculate parking reward
        parking_reward = self._calculate_parking_reward()
        reward = parking_reward
        
        # Check parking success
        success, success_info = self._check_parking_success()
        
        if success:
            done = True
            reward += 100.0
            info["parking_success"] = True
            print(f"🎉 PARKING SUCCESS in space {self.target_space['id']}!")
        
        # Add parking metrics
        info.update(success_info)
        info.update(self._get_parking_metrics())
        
        return obs, reward, done, truncated, info
    
    def _calculate_parking_reward(self) -> float:
        """Calculate parking progress reward"""
        reward = 0.0
        
        # Distance reward
        current_distance = self._get_distance_to_target()
        if current_distance < self.best_distance:
            progress = (self.best_distance - current_distance) / max(self.initial_distance, 1.0)
            reward += progress * 8.0
            self.best_distance = current_distance
        
        # Heading alignment reward
        heading_error = self._get_heading_error()
        heading_reward = max(0, 1.0 - heading_error / 180.0)
        reward += heading_reward * 3.0
        
        # Proximity bonus
        if current_distance < 8.0:
            proximity_bonus = (8.0 - current_distance) / 8.0
            reward += proximity_bonus * 4.0
        
        # Success criteria progress
        position_progress = max(0, 1.0 - current_distance / self.config["position_tolerance"])
        heading_progress = max(0, 1.0 - heading_error / self.config["heading_tolerance"])
        success_progress = (position_progress + heading_progress) / 2.0
        reward += success_progress * 5.0
        
        # Time penalty
        reward -= 0.02
        
        # Collision penalty
        if self.vehicle.crash_vehicle or self.vehicle.crash_object:
            reward -= 3.0
        
        return reward
    
    def _check_parking_success(self) -> Tuple[bool, Dict]:
        """Check parking success with hold time"""
        distance = self._get_distance_to_target()
        heading_error = self._get_heading_error()
        
        position_ok = distance <= self.config["position_tolerance"]
        heading_ok = heading_error <= self.config["heading_tolerance"]
        criteria_met = position_ok and heading_ok
        
        if criteria_met:
            self.success_hold_timer += 1.0/60.0  # Assuming 60 FPS
        else:
            self.success_hold_timer = 0.0
        
        success = self.success_hold_timer >= self.config["success_hold_time"]
        
        info = {
            "distance_to_target": distance,
            "heading_error": heading_error,
            "position_criteria_met": position_ok,
            "heading_criteria_met": heading_ok,
            "success_hold_progress": min(1.0, self.success_hold_timer / self.config["success_hold_time"]),
            "target_space_id": self.target_space["id"],
            "target_space_type": self.target_space["type"],
        }
        
        return success, info
    
    def _get_distance_to_target(self) -> float:
        """Distance to target parking space"""
        vehicle_pos = np.array(self.vehicle.position[:2])
        target_pos = np.array(self.target_space["position"][:2])
        return np.linalg.norm(target_pos - vehicle_pos)
    
    def _get_heading_error(self) -> float:
        """Heading error from target (degrees)"""
        vehicle_heading = math.degrees(self.vehicle.heading_theta)
        target_heading = self.target_space["heading"]
        
        # Normalize angle difference
        error = target_heading - vehicle_heading
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
            
        return abs(error)
    
    def _get_parking_metrics(self) -> Dict:
        """Get parking performance metrics"""
        return {
            "current_distance": self._get_distance_to_target(),
            "current_heading_error": self._get_heading_error(),
            "best_distance_achieved": self.best_distance,
            "initial_distance": self.initial_distance,
            "progress_percent": max(0, (self.initial_distance - self.best_distance) / max(self.initial_distance, 1.0) * 100),
            "target_space_info": self.target_space,
        }
    
    def _print_episode_info(self):
        """Print episode information"""
        print(f"\n🅿️  Visual Trailer Parking Lot")
        print("=" * 50)
        print(f"Target Space: #{self.target_space['id']}")
        print(f"Parking Type: {self.target_space['type'].upper()}")
        print(f"Difficulty: {self.target_space['difficulty'].upper()}")
        print(f"Position: ({self.target_space['position'][0]:.1f}, {self.target_space['position'][1]:.1f})")
        print(f"Target Heading: {self.target_space['heading']}°")
        
        print(f"\nControls:")
        print(f"  W/S - Forward/Reverse (REVERSE ENABLED!)")
        print(f"  A/D - Steer left/right")
        print(f"  R - Reset episode")
        print(f"  ESC - Quit")
        
        # Show all available spaces
        reverse_spaces = [s for s in self.parking_spaces if s["type"] == "reverse"]
        forward_spaces = [s for s in self.parking_spaces if s["type"] == "forward"]
        
        print(f"\nParking Lot Layout:")
        print(f"  Reverse spaces (back in): {[s['id'] for s in reverse_spaces]}")
        print(f"  Forward spaces (drive in): {[s['id'] for s in forward_spaces]}")
        print(f"  Total spaces: {len(self.parking_spaces)}")
    
    def set_target_space(self, space_id: int) -> bool:
        """Set specific target space"""
        if 0 <= space_id < len(self.parking_spaces):
            self.target_space_id = space_id
            self.target_space = self.parking_spaces[space_id]
            print(f"Target set to space #{space_id} ({self.target_space['type']})")
            return True
        return False
    
    def get_all_spaces(self) -> List[Dict]:
        """Get information about all parking spaces"""
        return self.parking_spaces.copy()


def demo_visual_parking_lot():
    """Demo the visual parking lot environment"""
    
    # Configuration for different scenarios
    configs = {
        "reverse_parking": {
            "target_parking_space": 2,  # Reverse space
            "position_tolerance": 2.5,
            "heading_tolerance": 15.0,
        },
        "forward_parking": {
            "target_parking_space": 8,  # Forward space  
            "position_tolerance": 2.0,
            "heading_tolerance": 20.0,
        },
        "random_challenge": {
            "random_target": True,  # Random space selection
            "position_tolerance": 2.0,
            "heading_tolerance": 15.0,
        }
    }
    
    # Choose configuration
    config_name = "reverse_parking"  # Change this to try different scenarios
    config = configs[config_name]
    
    env = VisualTrailerParkingEnv(config)
    
    try:
        print(f"🚛 Visual Trailer Parking - {config_name.replace('_', ' ').title()}")
        print("This uses MetaDrive's built-in parking lot with trailer challenges!")
        
        obs = env.reset()
        
        # Show available spaces
        all_spaces = env.get_all_spaces()
        print(f"\nAll available spaces:")
        for space in all_spaces:
            print(f"  Space #{space['id']}: {space['type']} at ({space['position'][0]:.0f}, {space['position'][1]:.0f}) - {space['difficulty']}")
        
        step_count = 0
        while True:
            obs, reward, done, truncated, info = env.step([0, 0])  # Manual control
            step_count += 1
            
            # Progress updates
            if step_count % 120 == 0:
                metrics = info
                print(f"Step {step_count}: Distance={metrics.get('current_distance', 0):.2f}m, "
                      f"Heading error={metrics.get('current_heading_error', 0):.1f}°, "
                      f"Progress={metrics.get('progress_percent', 0):.1f}%")
            
            if done or truncated:
                if info.get("parking_success"):
                    space_info = info["target_space_info"]
                    print(f"✅ SUCCESS! Parked in space #{space_info['id']} ({space_info['type']})")
                    print(f"Final distance: {info['distance_to_target']:.2f}m")
                    print(f"Final heading error: {info['heading_error']:.1f}°")
                else:
                    print(f"❌ Episode ended without success")
                
                # Ask user if they want to try different space
                print(f"\nWould you like to:")
                print(f"  1. Try the same space again")
                print(f"  2. Try a different space")
                print(f"  3. Random space")
                print(f"  4. Quit")
                
                try:
                    # For demo, just reset to same space
                    obs = env.reset()
                    step_count = 0
                except KeyboardInterrupt:
                    break
                
    except KeyboardInterrupt:
        print("\nDemo ended by user")
    finally:
        env.close()


# Utility functions for different parking challenges
def challenge_reverse_parking():
    """Challenge: Reverse parking (backing into space)"""
    config = {"target_parking_space": random.choice([0, 1, 2, 3, 4, 5])}  # Reverse spaces
    env = VisualTrailerParkingEnv(config)
    return env

def challenge_forward_parking():
    """Challenge: Forward parking (driving into space)"""
    config = {"target_parking_space": random.choice([6, 7, 8, 9, 10, 11])}  # Forward spaces
    env = VisualTrailerParkingEnv(config)
    return env

def challenge_tight_parking():
    """Challenge: Tight parking spaces"""
    config = {
        "target_parking_space": random.choice([2, 3, 8, 9]),  # Middle spaces (harder)
        "position_tolerance": 1.5,  # Tighter tolerance
        "heading_tolerance": 10.0,
    }
    env = VisualTrailerParkingEnv(config)
    return env


if __name__ == "__main__":
    demo_visual_parking_lot()
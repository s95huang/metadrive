#!/usr/bin/env python
"""
Custom MetaDrive environment for truck trailer parking testing.

This environment provides:
- Large parking spaces suitable for trucks with trailers  
- Multiple parking challenges (parallel, perpendicular, reverse parking)
- Success/failure detection based on final position and orientation
- Customizable difficulty levels
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import math

from metadrive import MetaDriveEnv
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import Straight
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.road_network import Road
from metadrive.constants import TerminationState
from metadrive.utils.coordinates_shift import panda_vector
from metadrive.obs.observation_base import ObservationBase


class TrailerParkingObservation(ObservationBase):
    """Custom observation for trailer parking that includes parking space info"""
    
    def __init__(self, config):
        super().__init__(config)
        
    @property
    def observation_space(self):
        from gym.spaces import Box
        # Basic vehicle state + parking target info
        return Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    
    def observe(self, vehicle):
        # Vehicle state
        pos = vehicle.position
        heading = vehicle.heading_theta
        velocity = vehicle.velocity
        
        # Get parking target from environment
        env = self.engine.current_env
        if hasattr(env, 'target_parking_space'):
            target_pos = env.target_parking_space['position']
            target_heading = env.target_parking_space['heading']
            
            # Relative position to target
            rel_x = target_pos[0] - pos[0]
            rel_y = target_pos[1] - pos[1]
            rel_heading = target_heading - heading
            
            # Distance to target
            distance = np.linalg.norm([rel_x, rel_y])
        else:
            rel_x = rel_y = rel_heading = distance = 0.0
        
        return np.array([
            pos[0], pos[1], heading,  # Vehicle position and heading
            velocity[0], velocity[1],  # Vehicle velocity
            rel_x, rel_y, rel_heading, distance,  # Target relative info
            float(vehicle.crash_vehicle),  # Collision status
            float(vehicle.crash_object),  # Object collision
            float(vehicle.on_lane),  # On lane status
        ], dtype=np.float32)


class TrailerParkingEnv(MetaDriveEnv):
    """
    Environment for testing truck trailer parking maneuvers.
    
    Features:
    - Large parking spaces designed for truck-trailer combinations
    - Multiple parking scenarios (parallel, perpendicular, reverse)
    - Precision-based success criteria
    - Configurable difficulty levels
    """
    
    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        config = super().default_config()
        config.update({
            # Environment settings
            "manual_control": False,
            "use_render": True,
            "crash_done": True,
            "out_of_road_done": True,
            "arrive_dest_done": True,
            
            # Map and scenario settings
            "num_scenarios": 1,
            "start_seed": 1000,
            "map": None,  # We'll create custom map
            
            # Parking space configuration
            "parking_space_length": 15.0,  # Length of parking space (for truck + trailer)
            "parking_space_width": 4.0,    # Width of parking space
            "parking_type": "perpendicular",  # "perpendicular", "parallel", "reverse"
            "target_precision": 1.0,       # Required precision for successful parking (meters)
            "heading_precision": 10.0,     # Required heading precision (degrees)
            
            # Vehicle configuration with trailer
            "vehicle_config": {
                "enable_reverse": True,
                "show_dest_mark": True,
                "show_line_to_dest": True,
                "trailer_kinematic": {
                    "enabled": True,
                    "length": 6.0,     # Trailer length
                    "width": 2.5,      # Trailer width  
                    "height": 2.5,     # Trailer height
                    "origin_to_hitch": [3.0, 0.0, 1.0],
                    "hitch_offset_on_tractor": [-3.0, 0.0, 1.0],
                }
            },
            
            # Custom observation
            "image_observation": False,
            "interface_panel": ["dashboard"],
            
            # Difficulty settings
            "parking_difficulty": "medium",  # "easy", "medium", "hard"
            "add_obstacles": False,          # Add static obstacles around parking area
            "wind_effect": False,            # Add wind disturbance
            
            # Reward configuration
            "reward_distance_factor": 1.0,
            "reward_heading_factor": 0.5,
            "reward_collision_penalty": -10.0,
            "reward_success_bonus": 50.0,
            "reward_step_penalty": -0.1,
        })
        return config
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.target_parking_space = None
        self.parking_spaces = []
        self.initial_distance = 0.0
        self.best_distance = float('inf')
        
    def _create_custom_map(self):
        """Create a custom map with parking spaces suitable for trailers"""
        # This creates a simple map with a parking lot area
        # In practice, you might want to use the PGBlock system for more complex layouts
        
        # Define parking space based on configuration
        space_length = self.config["parking_space_length"]
        space_width = self.config["parking_space_width"]
        
        # Create target parking space
        if self.config["parking_type"] == "perpendicular":
            # Perpendicular parking space
            self.target_parking_space = {
                "position": [20.0, 0.0],  # Center of parking space
                "heading": 0.0,           # Target heading
                "length": space_length,
                "width": space_width,
                "type": "perpendicular"
            }
        elif self.config["parking_type"] == "parallel":
            # Parallel parking space
            self.target_parking_space = {
                "position": [25.0, 2.0],
                "heading": 90.0,  # Parallel to road
                "length": space_length,
                "width": space_width,
                "type": "parallel"
            }
        else:  # reverse
            # Reverse parking space
            self.target_parking_space = {
                "position": [20.0, 0.0],
                "heading": 180.0,  # Backing in
                "length": space_length,
                "width": space_width,
                "type": "reverse"
            }
    
    def reset(self, seed: Optional[int] = None):
        """Reset environment and set up new parking challenge"""
        if seed is not None:
            self.seed(seed)
            
        # Create custom map
        self._create_custom_map()
        
        # Call parent reset
        obs = super().reset()
        
        # Set initial position away from parking space
        vehicle = self.vehicle
        if self.config["parking_type"] == "perpendicular":
            # Start at entrance to parking lot
            vehicle.set_position([0.0, 0.0])
            vehicle.set_heading_theta(np.deg2rad(0))
        elif self.config["parking_type"] == "parallel":
            # Start ahead of parking space
            vehicle.set_position([15.0, 2.0])
            vehicle.set_heading_theta(np.deg2rad(90))
        else:  # reverse
            # Start in front of space, need to reverse in
            vehicle.set_position([10.0, 0.0])
            vehicle.set_heading_theta(np.deg2rad(0))
        
        # Calculate initial distance for reward scaling
        self.initial_distance = self._get_distance_to_target()
        self.best_distance = self.initial_distance
        
        return self._get_observation()
    
    def step(self, action):
        """Step environment and check parking success"""
        obs, reward, done, truncated, info = super().step(action)
        
        # Calculate custom reward
        custom_reward = self._calculate_parking_reward()
        reward = custom_reward
        
        # Check parking success
        success, success_info = self._check_parking_success()
        if success:
            done = True
            reward += self.config["reward_success_bonus"]
            info["parking_success"] = True
            info.update(success_info)
        
        # Update info with parking metrics
        info.update(self._get_parking_metrics())
        
        return obs, reward, done, truncated, info
    
    def _get_observation(self):
        """Get custom observation including parking space info"""
        if hasattr(self, '_observation'):
            return self._observation.observe(self.vehicle)
        
        # Fallback to basic observation
        vehicle = self.vehicle
        pos = vehicle.position
        target_pos = self.target_parking_space['position']
        
        return np.array([
            pos[0], pos[1], vehicle.heading_theta,
            target_pos[0], target_pos[1], self.target_parking_space['heading'],
            self._get_distance_to_target(),
            self._get_heading_error(),
            float(vehicle.crash_vehicle),
            float(vehicle.crash_object),
        ], dtype=np.float32)
    
    def _get_distance_to_target(self) -> float:
        """Calculate distance from vehicle to target parking position"""
        vehicle_pos = self.vehicle.position
        target_pos = self.target_parking_space['position']
        return np.linalg.norm([vehicle_pos[0] - target_pos[0], 
                              vehicle_pos[1] - target_pos[1]])
    
    def _get_heading_error(self) -> float:
        """Calculate heading error from target orientation"""
        vehicle_heading = math.degrees(self.vehicle.heading_theta)
        target_heading = self.target_parking_space['heading']
        
        # Normalize angle difference to [-180, 180]
        error = target_heading - vehicle_heading
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        
        return abs(error)
    
    def _calculate_parking_reward(self) -> float:
        """Calculate reward based on parking progress"""
        reward = 0.0
        
        # Distance-based reward
        current_distance = self._get_distance_to_target()
        if current_distance < self.best_distance:
            reward += (self.best_distance - current_distance) * self.config["reward_distance_factor"]
            self.best_distance = current_distance
        
        # Heading alignment reward
        heading_error = self._get_heading_error()
        heading_reward = max(0, 1.0 - heading_error / 180.0) * self.config["reward_heading_factor"]
        reward += heading_reward
        
        # Penalties
        if self.vehicle.crash_vehicle or self.vehicle.crash_object:
            reward += self.config["reward_collision_penalty"]
        
        # Step penalty to encourage efficiency
        reward += self.config["reward_step_penalty"]
        
        return reward
    
    def _check_parking_success(self) -> Tuple[bool, Dict]:
        """Check if vehicle is successfully parked"""
        distance = self._get_distance_to_target()
        heading_error = self._get_heading_error()
        
        distance_ok = distance <= self.config["target_precision"]
        heading_ok = heading_error <= self.config["heading_precision"]
        
        success = distance_ok and heading_ok
        
        info = {
            "final_distance": distance,
            "final_heading_error": heading_error,
            "distance_precision_met": distance_ok,
            "heading_precision_met": heading_ok,
            "required_distance_precision": self.config["target_precision"],
            "required_heading_precision": self.config["heading_precision"]
        }
        
        return success, info
    
    def _get_parking_metrics(self) -> Dict:
        """Get detailed parking performance metrics"""
        return {
            "distance_to_target": self._get_distance_to_target(),
            "heading_error": self._get_heading_error(),
            "parking_type": self.config["parking_type"],
            "best_distance_achieved": self.best_distance,
            "initial_distance": self.initial_distance,
        }


def create_trailer_parking_demo():
    """Create and run a demo of the trailer parking environment"""
    config = {
        "use_render": True,
        "manual_control": True,
        "parking_type": "perpendicular",  # Try "parallel" or "reverse" 
        "parking_difficulty": "medium",
        "show_fps": True,
        "show_logo": False,
    }
    
    env = TrailerParkingEnv(config)
    
    try:
        print("=== Trailer Parking Test Environment ===")
        print("Controls: W/A/S/D to drive, SPACE for brake")
        print("Objective: Park the truck-trailer in the designated space")
        print(f"Parking type: {config['parking_type']}")
        print("Press ESC to quit")
        
        obs = env.reset()
        
        for i in range(10000):  # Max steps
            if config["manual_control"]:
                action = [0, 0]  # Manual control handles input
            else:
                # Simple AI for demo - drive towards target
                distance = env._get_distance_to_target()
                if distance > 2.0:
                    action = [0.3, 0.0]  # Drive forward
                else:
                    action = [0.0, 0.0]  # Stop when close
            
            obs, reward, done, truncated, info = env.step(action)
            
            # Print progress
            if i % 100 == 0:
                metrics = env._get_parking_metrics()
                print(f"Step {i}: Distance={metrics['distance_to_target']:.2f}m, "
                      f"Heading error={metrics['heading_error']:.1f}°, "
                      f"Reward={reward:.2f}")
            
            if done:
                if info.get("parking_success", False):
                    print("🎉 Parking successful!")
                    print(f"Final distance: {info['final_distance']:.2f}m")
                    print(f"Final heading error: {info['final_heading_error']:.1f}°")
                else:
                    print("❌ Parking failed")
                
                # Reset for another attempt
                obs = env.reset()
                
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    create_trailer_parking_demo()
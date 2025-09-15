#!/usr/bin/env python
"""
Simple trailer parking test scenario for MetaDrive.

This provides a minimal but effective setup for testing truck trailer parking:
- Single parking space clearly marked
- Simple success/failure detection
- Easy to modify and extend
- Good for testing parking algorithms
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple

from metadrive import MetaDriveEnv


class SimpleTrailerParkingEnv(MetaDriveEnv):
    """
    Simple environment for testing trailer parking maneuvers.
    
    Features:
    - Single large parking space for truck-trailer
    - Clear visual markers
    - Simple success criteria
    - Progress tracking
    """
    
    @classmethod
    def default_config(cls) -> Dict:
        config = super().default_config()
        config.update({
            # Basic setup
            "manual_control": True,
            "use_render": True,
            "num_scenarios": 1,
            "traffic_density": 0.0,  # No other vehicles
            "map": "S",  # Simple straight road
            
            # Parking space configuration
            "parking_space": {
                "position": [50, 10],    # Parking space center
                "heading": 270,          # Degrees (270 = facing left/north)
                "length": 12.0,          # Space length (truck + trailer)
                "width": 3.5,            # Space width
            },
            
            # Success criteria
            "success_distance": 2.0,     # Max distance from center (meters)
            "success_heading": 20.0,     # Max heading error (degrees)
            
            # Vehicle with trailer
            "vehicle_config": {
                "enable_reverse": True,
                "show_dest_mark": True,
                "trailer_kinematic": {
                    "enabled": True,
                    "length": 5.0,
                    "width": 2.2,
                    "height": 2.0,
                    "origin_to_hitch": [2.5, 0.0, 0.8],
                    "hitch_offset_on_tractor": [-2.5, 0.0, 0.8],
                }
            },
            
            # Visualization
            "show_fps": True,
            "camera_dist": 20,
            "camera_height": 8,
            
            # Episode settings  
            "horizon": 1500,  # Max steps per episode
        })
        return config
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.parking_space = self.config["parking_space"]
        self.start_distance = 0.0
        self.best_distance = float('inf')
        
    def reset(self, seed: Optional[int] = None):
        """Reset environment and position vehicle"""
        obs = super().reset(seed=seed)
        
        # Position vehicle at starting location
        self._position_vehicle_for_parking()
        
        # Update destination to parking space
        self.vehicle.set_dest_destination = self.parking_space["position"]
        
        # Initialize tracking
        self.start_distance = self._get_distance_to_parking()
        self.best_distance = self.start_distance
        
        print(f"🎯 Parking Challenge Started!")
        print(f"Target: Park at ({self.parking_space['position'][0]}, {self.parking_space['position'][1]})")
        print(f"Required heading: {self.parking_space['heading']}°")
        print(f"Starting distance: {self.start_distance:.1f}m")
        
        return obs
    
    def _position_vehicle_for_parking(self):
        """Position vehicle for parking approach"""
        # Start on the main road, approaching the parking space
        start_x = 20.0  # Start behind parking space
        start_y = 0.0   # On main road
        start_heading = 0.0  # Facing forward along road
        
        self.vehicle.set_position([start_x, start_y, 0.3])
        self.vehicle.set_heading_theta(math.radians(start_heading))
    
    def step(self, action):
        """Step environment with parking progress tracking"""
        obs, reward, done, truncated, info = super().step(action)
        
        # Calculate custom parking reward
        parking_reward = self._calculate_parking_reward()
        reward = parking_reward
        
        # Check parking success
        success, parking_info = self._check_parking_success()
        
        if success:
            done = True
            reward += 50.0  # Success bonus
            print("🎉 PARKING SUCCESSFUL!")
            info["parking_success"] = True
        
        # Add parking metrics to info
        info.update(parking_info)
        info.update(self._get_progress_info())
        
        return obs, reward, done, truncated, info
    
    def _get_distance_to_parking(self) -> float:
        """Distance from vehicle to parking space center"""
        vehicle_pos = np.array(self.vehicle.position[:2])
        parking_pos = np.array(self.parking_space["position"])
        return np.linalg.norm(parking_pos - vehicle_pos)
    
    def _get_heading_error(self) -> float:
        """Heading difference from target (degrees)"""
        vehicle_heading = math.degrees(self.vehicle.heading_theta)
        target_heading = self.parking_space["heading"]
        
        # Calculate shortest angle difference
        diff = target_heading - vehicle_heading
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
            
        return abs(diff)
    
    def _calculate_parking_reward(self) -> float:
        """Calculate reward based on parking progress"""
        reward = 0.0
        
        # Distance reward - encourage getting closer
        current_distance = self._get_distance_to_parking()
        if current_distance < self.best_distance:
            progress = (self.best_distance - current_distance) / self.start_distance
            reward += progress * 5.0  # Scale factor for distance progress
            self.best_distance = current_distance
        
        # Heading alignment reward
        heading_error = self._get_heading_error()
        heading_reward = max(0, 1.0 - heading_error / 180.0)  # 0-1 based on alignment
        reward += heading_reward * 2.0
        
        # Proximity bonus when very close
        if current_distance < 5.0:
            proximity_bonus = (5.0 - current_distance) / 5.0
            reward += proximity_bonus * 3.0
        
        # Small time penalty to encourage efficiency
        reward -= 0.02
        
        # Collision penalty
        if self.vehicle.crash_vehicle or self.vehicle.crash_object:
            reward -= 2.0
        
        return reward
    
    def _check_parking_success(self) -> Tuple[bool, Dict]:
        """Check if vehicle is successfully parked"""
        distance = self._get_distance_to_parking()
        heading_error = self._get_heading_error()
        
        distance_ok = distance <= self.config["success_distance"]
        heading_ok = heading_error <= self.config["success_heading"]
        
        success = distance_ok and heading_ok
        
        info = {
            "distance_to_parking": distance,
            "heading_error": heading_error,
            "distance_criterion_met": distance_ok,
            "heading_criterion_met": heading_ok,
            "required_distance": self.config["success_distance"],
            "required_heading_accuracy": self.config["success_heading"],
        }
        
        return success, info
    
    def _get_progress_info(self) -> Dict:
        """Get parking progress information"""
        current_distance = self._get_distance_to_parking()
        progress_percent = max(0, (self.start_distance - current_distance) / self.start_distance * 100)
        
        return {
            "parking_progress_percent": progress_percent,
            "best_distance_achieved": self.best_distance,
            "current_distance_to_parking": current_distance,
            "start_distance": self.start_distance,
        }


def test_simple_trailer_parking():
    """Test the simple trailer parking environment"""
    config = {
        "manual_control": True,  # Set to False for automated testing
        "use_render": True,
        "show_logo": False,
    }
    
    env = SimpleTrailerParkingEnv(config)
    
    try:
        print("🚛 Simple Trailer Parking Test")
        print("=" * 40)
        print("Controls:")
        print("  W/S - Accelerate/Brake")  
        print("  A/D - Steer left/right")
        print("  R - Reset episode")
        print("  ESC - Quit")
        print("\nObjective: Park the truck-trailer in the designated space")
        
        obs = env.reset()
        step_count = 0
        
        while True:
            # Manual control or simple AI
            if config["manual_control"]:
                action = [0, 0]  # Manual control handles input
            else:
                # Simple AI for testing - drive towards parking space
                distance = env._get_distance_to_parking()
                heading_error = env._get_heading_error()
                
                if distance > 10:
                    # Far away - drive forward
                    action = [0.3, 0]
                elif distance > 3:
                    # Getting close - slower approach with steering
                    steering = np.clip(heading_error / 45.0, -1, 1) * 0.3
                    action = [0.1, steering]
                else:
                    # Very close - precision maneuvers
                    action = [0.0, 0]
            
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            # Print progress occasionally
            if step_count % 100 == 0:
                progress = info.get("parking_progress_percent", 0)
                distance = info.get("current_distance_to_parking", 0)
                heading_err = info.get("heading_error", 0)
                
                print(f"Step {step_count}: Progress={progress:.1f}%, "
                      f"Distance={distance:.2f}m, "
                      f"Heading error={heading_err:.1f}°, "
                      f"Reward={reward:.2f}")
            
            if done or truncated:
                if info.get("parking_success"):
                    print(f"\n✅ SUCCESS! Parked in {step_count} steps")
                    print(f"Final distance: {info['distance_to_parking']:.2f}m")
                    print(f"Final heading error: {info['heading_error']:.1f}°")
                else:
                    print(f"\n❌ Episode ended after {step_count} steps")
                    if not info['distance_criterion_met']:
                        print(f"Distance not met: {info['distance_to_parking']:.2f}m > {info['required_distance']}m")
                    if not info['heading_criterion_met']:
                        print(f"Heading not met: {info['heading_error']:.1f}° > {info['required_heading_accuracy']}°")
                
                # Reset for another attempt
                obs = env.reset()
                step_count = 0
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    test_simple_trailer_parking()
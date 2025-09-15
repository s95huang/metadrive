#!/usr/bin/env python
"""
Real Trailer Parking Lot Environment for MetaDrive

This creates an actual parking lot with visible parking spaces, lane markings,
and proper visual representation of parking spots for trailers.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from metadrive import MetaDriveEnv
from metadrive.component.static_object.traffic_object import TrafficCone, TrafficBarrier
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road
from metadrive.utils.coordinates_shift import panda_vector
import random


@dataclass
class ParkingSpace:
    """Represents a parking space with visual markers"""
    id: str
    center_pos: Tuple[float, float, float]
    heading: float  # degrees
    length: float
    width: float
    is_reverse: bool  # True if requires backing in
    difficulty: str
    corners: List[Tuple[float, float, float]] = None
    
    def __post_init__(self):
        """Calculate corner positions"""
        self.corners = self._calculate_corners()
    
    def _calculate_corners(self):
        """Calculate the four corners of the parking space"""
        x, y, z = self.center_pos
        cos_h = math.cos(math.radians(self.heading))
        sin_h = math.sin(math.radians(self.heading))
        
        hl, hw = self.length / 2, self.width / 2
        
        # Local corner offsets (relative to center)
        local_corners = [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]]
        
        # Transform to world coordinates
        corners = []
        for lx, ly in local_corners:
            wx = x + (lx * cos_h - ly * sin_h)
            wy = y + (lx * sin_h + ly * cos_h)
            corners.append((wx, wy, z))
        
        return corners


class RealTrailerParkingLotEnv(MetaDriveEnv):
    """
    Environment that creates an actual parking lot with visual spaces.
    
    This creates:
    - A proper parking lot area with entrance/exit roads
    - Visible parking space markers (cones at corners)
    - Lane markings and proper road layout
    - Multiple trailer-appropriate parking spaces
    """
    
    @classmethod
    def default_config(cls) -> Dict:
        config = super().default_config()
        config.update({
            # Use a simple map and modify it
            "map": "C",  # Circular road as base
            "num_scenarios": 1,
            "traffic_density": 0.0,
            
            # Parking lot settings
            "lot_width": 40,     # Width of parking area
            "lot_length": 80,    # Length of parking area
            "spaces_per_row": 4, # Spaces per row
            "num_rows": 2,       # Number of rows
            
            # Space dimensions (trailer-appropriate)
            "space_length": 12.0,
            "space_width": 3.5,
            "aisle_width": 10.0,  # Width between rows
            
            # Visual settings
            "show_parking_lines": True,
            "show_space_numbers": True,
            "use_cones_for_markers": True,
            
            # Vehicle config with trailer
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
            "position_tolerance": 2.0,
            "heading_tolerance": 15.0,
            "success_hold_time": 2.0,
            
            # Controls
            "manual_control": True,
            "use_render": True,
            "camera_height": 20,
            "camera_dist": 30,
            "show_fps": True,
            
            # Episode settings
            "horizon": 3000,
        })
        return config
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Parking lot state
        self.parking_spaces: List[ParkingSpace] = []
        self.target_space: Optional[ParkingSpace] = None
        self.visual_objects = []  # Store created visual objects
        self.cone_objects = []    # Store cone markers
        
        # Performance tracking
        self.initial_distance = 0.0
        self.best_distance = float('inf')
        self.success_hold_timer = 0.0
    
    def reset(self, seed: Optional[int] = None):
        """Reset and create parking lot"""
        if seed is not None:
            self.seed(seed)
        
        # Clear previous visual objects
        self._clear_visual_objects()
        
        # Call parent reset first
        obs = super().reset()
        
        # Create parking lot after environment is initialized
        self._create_parking_lot()
        self._create_visual_markers()
        
        # Select target and position vehicle
        self._select_target_space()
        self._position_vehicle_at_entrance()
        
        # Initialize metrics
        self.initial_distance = self._get_distance_to_target()
        self.best_distance = self.initial_distance
        self.success_hold_timer = 0.0
        
        self._print_episode_info()
        
        return obs
    
    def _create_parking_lot(self):
        """Create the parking spaces layout"""
        self.parking_spaces = []
        
        # Parking lot center position (offset from road)
        lot_center_x = 60.0  # Away from main road
        lot_center_y = 0.0
        
        spaces_per_row = self.config["spaces_per_row"]
        num_rows = self.config["num_rows"] 
        space_length = self.config["space_length"]
        space_width = self.config["space_width"]
        aisle_width = self.config["aisle_width"]
        
        # Create rows of parking spaces
        for row in range(num_rows):
            # Alternate sides of the aisle
            y_offset = (aisle_width/2 + space_width/2) * (1 if row % 2 == 0 else -1)
            
            # Determine if this row is for reverse or forward parking
            is_reverse_row = (row % 2 == 0)
            heading = 270 if is_reverse_row else 90  # 270° = backing in from below
            
            for space_idx in range(spaces_per_row):
                # Space position along the row
                x_pos = lot_center_x - (spaces_per_row * space_length / 2) + (space_idx * space_length) + (space_length / 2)
                y_pos = lot_center_y + y_offset
                
                space_id = f"{'R' if is_reverse_row else 'F'}{row+1}-{space_idx+1}"
                
                # Assign difficulty based on position
                if space_idx == 0 or space_idx == spaces_per_row - 1:
                    difficulty = "easy"  # End spaces
                elif space_idx == 1 or space_idx == spaces_per_row - 2:
                    difficulty = "medium"
                else:
                    difficulty = "hard"  # Middle spaces
                
                space = ParkingSpace(
                    id=space_id,
                    center_pos=(x_pos, y_pos, 0.0),
                    heading=heading,
                    length=space_length,
                    width=space_width,
                    is_reverse=is_reverse_row,
                    difficulty=difficulty
                )
                
                self.parking_spaces.append(space)
        
        print(f"Created parking lot with {len(self.parking_spaces)} spaces")
        for space in self.parking_spaces:
            reverse_str = "REVERSE" if space.is_reverse else "FORWARD"
            print(f"  {space.id}: {reverse_str} at ({space.center_pos[0]:.1f}, {space.center_pos[1]:.1f}) - {space.difficulty}")
    
    def _create_visual_markers(self):
        """Create visual markers for parking spaces"""
        if not self.config["use_cones_for_markers"]:
            return
            
        # Create cones at parking space corners
        for space in self.parking_spaces:
            for i, corner in enumerate(space.corners):
                try:
                    # Create traffic cone at corner
                    cone = TrafficCone(
                        position=corner,
                        heading=0,
                        random_seed=self.engine.global_random_seed + hash(space.id + str(i))
                    )
                    
                    # Spawn the cone in the world
                    cone_obj = self.engine.spawn_object(cone, force_spawn=True)
                    self.cone_objects.append(cone_obj)
                    
                except Exception as e:
                    print(f"Warning: Could not create cone at {corner}: {e}")
        
        print(f"Created {len(self.cone_objects)} visual markers")
    
    def _create_ground_markings(self):
        """Create ground markings for parking spaces (alternative to cones)"""
        # This would create painted lines on the ground
        # For now, we'll use the cone approach which is more visible
        pass
    
    def _clear_visual_objects(self):
        """Clear all visual objects from previous episodes"""
        for obj in self.visual_objects + self.cone_objects:
            try:
                if hasattr(obj, 'destroy'):
                    obj.destroy()
                elif hasattr(obj, 'remove'):
                    obj.remove()
            except:
                pass
        
        self.visual_objects.clear()
        self.cone_objects.clear()
    
    def _select_target_space(self):
        """Select a target parking space"""
        if not self.parking_spaces:
            return
            
        # For demo, select a random space
        self.target_space = random.choice(self.parking_spaces)
        print(f"Target space: {self.target_space.id}")
    
    def _position_vehicle_at_entrance(self):
        """Position vehicle at parking lot entrance"""
        # Position vehicle at entrance to parking lot
        entrance_x = 20.0
        entrance_y = 0.0
        entrance_heading = 0.0  # Facing towards parking lot
        
        self.vehicle.set_position([entrance_x, entrance_y, 0.3])
        self.vehicle.set_heading_theta(math.radians(entrance_heading))
    
    def step(self, action):
        """Step with parking lot logic"""
        obs, reward, done, truncated, info = super().step(action)
        
        if not self.target_space:
            return obs, reward, done, truncated, info
        
        # Calculate parking reward
        parking_reward = self._calculate_parking_reward()
        reward = parking_reward
        
        # Check parking success
        success, success_info = self._check_parking_success()
        
        if success:
            done = True
            reward += 100.0  # Success bonus
            info["parking_success"] = True
            print(f"🎉 PARKING SUCCESS in space {self.target_space.id}!")
        
        # Add parking info
        info.update(success_info)
        info.update(self._get_parking_metrics())
        
        return obs, reward, done, truncated, info
    
    def _calculate_parking_reward(self) -> float:
        """Calculate reward for parking progress"""
        if not self.target_space:
            return 0.0
            
        reward = 0.0
        
        # Distance reward
        current_distance = self._get_distance_to_target()
        if current_distance < self.best_distance:
            progress = (self.best_distance - current_distance) / max(self.initial_distance, 1.0)
            reward += progress * 5.0
            self.best_distance = current_distance
        
        # Heading alignment reward  
        heading_error = self._get_heading_error()
        heading_reward = max(0, 1.0 - heading_error / 180.0)
        reward += heading_reward * 2.0
        
        # Close proximity bonus
        if current_distance < 5.0:
            proximity_bonus = (5.0 - current_distance) / 5.0
            reward += proximity_bonus * 3.0
        
        # Small time penalty
        reward -= 0.01
        
        # Collision penalty
        if self.vehicle.crash_vehicle or self.vehicle.crash_object:
            reward -= 2.0
        
        return reward
    
    def _check_parking_success(self) -> Tuple[bool, Dict]:
        """Check if successfully parked with hold time"""
        if not self.target_space:
            return False, {}
            
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
            "target_space_id": self.target_space.id,
            "target_is_reverse": self.target_space.is_reverse,
        }
        
        return success, info
    
    def _get_distance_to_target(self) -> float:
        """Distance to target parking space center"""
        if not self.target_space:
            return 0.0
            
        vehicle_pos = np.array(self.vehicle.position[:2])
        target_pos = np.array(self.target_space.center_pos[:2])
        return np.linalg.norm(target_pos - vehicle_pos)
    
    def _get_heading_error(self) -> float:
        """Heading error from target orientation (degrees)"""
        if not self.target_space:
            return 0.0
            
        vehicle_heading = math.degrees(self.vehicle.heading_theta)
        target_heading = self.target_space.heading
        
        # Normalize angle difference
        error = target_heading - vehicle_heading
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
            
        return abs(error)
    
    def _get_parking_metrics(self) -> Dict:
        """Get detailed parking metrics"""
        return {
            "current_distance": self._get_distance_to_target(),
            "current_heading_error": self._get_heading_error(),
            "best_distance_achieved": self.best_distance,
            "initial_distance": self.initial_distance,
            "total_spaces": len(self.parking_spaces),
            "progress_percent": max(0, (self.initial_distance - self.best_distance) / max(self.initial_distance, 1.0) * 100),
        }
    
    def _print_episode_info(self):
        """Print episode information"""
        if not self.target_space:
            return
            
        print(f"\n🅿️  Trailer Parking Lot Challenge")
        print("=" * 50)
        print(f"Target Space: {self.target_space.id}")
        print(f"Parking Type: {'REVERSE (Back in)' if self.target_space.is_reverse else 'FORWARD (Drive in)'}")
        print(f"Difficulty: {self.target_space.difficulty.upper()}")
        print(f"Position: ({self.target_space.center_pos[0]:.1f}, {self.target_space.center_pos[1]:.1f})")
        print(f"Target Heading: {self.target_space.heading}°")
        print(f"Total Spaces: {len(self.parking_spaces)}")
        
        print(f"\nControls:")
        print(f"  W/S - Forward/Reverse")
        print(f"  A/D - Steer left/right")
        print(f"  R - Reset")
        print(f"  ESC - Quit")
        
        # Show space layout
        print(f"\nParking Space Layout:")
        reverse_spaces = [s for s in self.parking_spaces if s.is_reverse]
        forward_spaces = [s for s in self.parking_spaces if not s.is_reverse]
        
        if reverse_spaces:
            print(f"  Reverse Spaces: {', '.join(s.id for s in reverse_spaces)}")
        if forward_spaces:
            print(f"  Forward Spaces: {', '.join(s.id for s in forward_spaces)}")
    
    def get_space_info(self, space_id: str) -> Optional[Dict]:
        """Get information about a specific parking space"""
        space = next((s for s in self.parking_spaces if s.id == space_id), None)
        if not space:
            return None
            
        return {
            "id": space.id,
            "position": space.center_pos,
            "heading": space.heading,
            "is_reverse": space.is_reverse,
            "difficulty": space.difficulty,
            "dimensions": (space.length, space.width)
        }
    
    def set_target_space(self, space_id: str) -> bool:
        """Set specific target space"""
        space = next((s for s in self.parking_spaces if s.id == space_id), None)
        if space:
            self.target_space = space
            print(f"Target set to: {space_id}")
            return True
        return False


def demo_real_parking_lot():
    """Demo the real parking lot environment"""
    config = {
        "manual_control": True,
        "use_render": True,
        "show_logo": False,
        "spaces_per_row": 4,
        "num_rows": 2,
        "use_cones_for_markers": True,
    }
    
    env = RealTrailerParkingLotEnv(config)
    
    try:
        print("🚛 Real Trailer Parking Lot")
        print("This creates an actual parking lot with visible parking spaces!")
        
        obs = env.reset()
        
        # Show available spaces
        spaces = [s.id for s in env.parking_spaces]
        print(f"Available spaces: {', '.join(spaces)}")
        
        step_count = 0
        while True:
            obs, reward, done, truncated, info = env.step([0, 0])  # Manual control
            step_count += 1
            
            # Progress updates
            if step_count % 150 == 0:
                metrics = info
                print(f"Step {step_count}: Distance={metrics.get('current_distance', 0):.2f}m, "
                      f"Heading error={metrics.get('current_heading_error', 0):.1f}°, "
                      f"Progress={metrics.get('progress_percent', 0):.1f}%")
            
            if done or truncated:
                if info.get("parking_success"):
                    print(f"✅ SUCCESS! Parked in space {info['target_space_id']}")
                else:
                    print(f"❌ Episode ended")
                
                # Reset for another attempt
                obs = env.reset()
                step_count = 0
                
    except KeyboardInterrupt:
        print("\nDemo ended by user")
    finally:
        env.close()


if __name__ == "__main__":
    demo_real_parking_lot()
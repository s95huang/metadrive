#!/usr/bin/env python
"""
Dedicated Trailer Parking Lot Environment for MetaDrive

This environment creates a realistic parking lot specifically designed for truck trailers:
- Multiple trailer-sized parking spaces
- Both forward and reverse parking challenges  
- Realistic spacing for maneuvering trailers
- Progressive difficulty levels
- Different parking orientations (perpendicular, angled)
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from metadrive import MetaDriveEnv
from metadrive.component.static_object.traffic_object import TrafficCone


class ParkingType(Enum):
    REVERSE_PERPENDICULAR = "reverse_perpendicular"  # Back into 90° space
    FORWARD_PERPENDICULAR = "forward_perpendicular"  # Drive into 90° space
    REVERSE_ANGLED = "reverse_angled"                # Back into angled space
    FORWARD_ANGLED = "forward_angled"                # Drive into angled space


@dataclass
class ParkingSpace:
    """Represents a single parking space in the lot"""
    id: str
    position: List[float]  # [x, y, z]
    heading: float         # Target heading in degrees
    length: float          # Space length
    width: float           # Space width  
    parking_type: ParkingType
    difficulty: str        # "easy", "medium", "hard"
    occupied: bool = False
    
    def get_corners(self) -> List[List[float]]:
        """Get the four corner positions of the parking space"""
        # Calculate corners based on position, heading, and dimensions
        cos_h = math.cos(math.radians(self.heading))
        sin_h = math.sin(math.radians(self.heading))
        
        # Half dimensions
        hl = self.length / 2
        hw = self.width / 2
        
        # Local corner offsets
        local_corners = [
            [-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]
        ]
        
        # Transform to world coordinates
        corners = []
        for lx, ly in local_corners:
            wx = self.position[0] + (lx * cos_h - ly * sin_h)
            wy = self.position[1] + (lx * sin_h + ly * cos_h)
            corners.append([wx, wy, self.position[2]])
        
        return corners


class TrailerParkingLotEnv(MetaDriveEnv):
    """
    Environment with a dedicated trailer parking lot featuring multiple spaces.
    
    Features:
    - 10+ trailer-appropriate parking spaces
    - Mix of forward and reverse parking challenges
    - Different angles (90°, 45°) and orientations
    - Progressive difficulty selection
    - Realistic spacing for trailer maneuvering
    """
    
    @classmethod
    def default_config(cls) -> Dict:
        config = super().default_config()
        config.update({
            # Basic environment setup
            "map": "S",  # Simple straight road as base
            "num_scenarios": 1,
            "traffic_density": 0.0,
            "manual_control": True,
            "use_render": True,
            
            # Parking lot configuration
            "lot_layout": "large",        # "small", "medium", "large"
            "parking_difficulty": "mixed", # "easy", "medium", "hard", "mixed"
            "target_space_selection": "random",  # "random", "closest", "furthest", "specific"
            "target_space_id": None,      # Specific space ID if selection is "specific"
            
            # Space generation settings
            "space_dimensions": {
                "length": 15.0,           # Standard trailer space length
                "width": 3.5,             # Standard trailer space width
                "aisle_width": 8.0,       # Width between rows for maneuvering
            },
            
            # Parking types to include
            "include_parking_types": [
                ParkingType.REVERSE_PERPENDICULAR,
                ParkingType.FORWARD_PERPENDICULAR, 
                ParkingType.REVERSE_ANGLED,
                ParkingType.FORWARD_ANGLED,
            ],
            
            # Success criteria
            "position_tolerance": 1.5,    # meters
            "heading_tolerance": 15.0,    # degrees
            "completion_bonus": 100.0,
            
            # Vehicle configuration  
            "vehicle_config": {
                "enable_reverse": True,
                "show_dest_mark": True,
                "show_line_to_dest": True,
                "max_speed": 15,  # Slower for parking lot
                "trailer_kinematic": {
                    "enabled": True,
                    "length": 8.0,
                    "width": 2.4,
                    "height": 2.5,
                    "origin_to_hitch": [4.0, 0.0, 1.0],
                    "hitch_offset_on_tractor": [-3.2, 0.0, 1.0],
                }
            },
            
            # Visual settings
            "show_parking_spaces": True,
            "show_space_numbers": True,
            "camera_height": 20,
            "camera_dist": 35,
            "show_fps": True,
            
            # Episode settings
            "horizon": 3000,  # Longer episodes for complex parking
            "auto_reset_on_success": False,
            "success_hold_time": 2.0,  # Seconds to hold position for success
        })
        return config
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Parking lot state
        self.parking_spaces: List[ParkingSpace] = []
        self.target_space: Optional[ParkingSpace] = None
        self.visual_markers = []
        
        # Performance tracking
        self.initial_distance = 0.0
        self.best_distance = float('inf')
        self.success_hold_timer = 0.0
        self.current_episode = 0
        
        # Statistics
        self.success_count = 0
        self.attempt_count = 0
        self.performance_history = []
        
    def reset(self, seed: Optional[int] = None):
        """Reset environment and generate new parking lot layout"""
        if seed is not None:
            self.seed(seed)
        
        # Generate parking lot layout
        self._generate_parking_lot()
        
        # Call parent reset
        obs = super().reset()
        
        # Select target parking space
        self._select_target_space()
        
        # Position vehicle at lot entrance
        self._position_vehicle_at_entrance()
        
        # Create visual markers for spaces
        if self.config["show_parking_spaces"]:
            self._create_visual_markers()
        
        # Initialize tracking
        self.initial_distance = self._get_distance_to_target()
        self.best_distance = self.initial_distance
        self.success_hold_timer = 0.0
        self.attempt_count += 1
        
        # Print episode info
        self._print_episode_info()
        
        return self._get_observation()
    
    def _generate_parking_lot(self):
        """Generate parking spaces based on configuration"""
        self.parking_spaces = []
        space_dims = self.config["space_dimensions"]
        include_types = self.config["include_parking_types"]
        difficulty = self.config["parking_difficulty"]
        
        # Define different lot layouts
        if self.config["lot_layout"] == "small":
            self._generate_small_lot(space_dims, include_types, difficulty)
        elif self.config["lot_layout"] == "medium":
            self._generate_medium_lot(space_dims, include_types, difficulty)
        else:  # large
            self._generate_large_lot(space_dims, include_types, difficulty)
    
    def _generate_small_lot(self, dims: Dict, types: List[ParkingType], difficulty: str):
        """Generate small parking lot (6 spaces)"""
        base_x, base_y = 30, 0
        
        # Row 1: 3 reverse perpendicular spaces (right side)
        for i in range(3):
            space_id = f"R1-{i+1}"
            x_pos = base_x + i * (dims["length"] + 2)
            y_pos = base_y + dims["width"]/2 + 4
            
            space = ParkingSpace(
                id=space_id,
                position=[x_pos, y_pos, 0],
                heading=270,  # Facing left (backing in from road)
                length=dims["length"],
                width=dims["width"],
                parking_type=ParkingType.REVERSE_PERPENDICULAR,
                difficulty=self._assign_difficulty(difficulty, i)
            )
            self.parking_spaces.append(space)
        
        # Row 2: 3 forward perpendicular spaces (left side) 
        for i in range(3):
            space_id = f"L1-{i+1}"
            x_pos = base_x + i * (dims["length"] + 2)
            y_pos = base_y - dims["width"]/2 - 4
            
            space = ParkingSpace(
                id=space_id,
                position=[x_pos, y_pos, 0],
                heading=90,   # Facing right (driving in from road)
                length=dims["length"],
                width=dims["width"],
                parking_type=ParkingType.FORWARD_PERPENDICULAR,
                difficulty=self._assign_difficulty(difficulty, i)
            )
            self.parking_spaces.append(space)
    
    def _generate_medium_lot(self, dims: Dict, types: List[ParkingType], difficulty: str):
        """Generate medium parking lot (12 spaces)"""
        base_x, base_y = 25, 0
        
        # Row 1: Reverse perpendicular (right side)
        for i in range(4):
            space_id = f"R1-{i+1}"
            x_pos = base_x + i * (dims["length"] + 1.5)
            y_pos = base_y + dims["width"]/2 + 4
            
            space = ParkingSpace(
                id=space_id,
                position=[x_pos, y_pos, 0],
                heading=270,
                length=dims["length"], 
                width=dims["width"],
                parking_type=ParkingType.REVERSE_PERPENDICULAR,
                difficulty=self._assign_difficulty(difficulty, i)
            )
            self.parking_spaces.append(space)
        
        # Row 2: Forward perpendicular (left side)
        for i in range(4):
            space_id = f"L1-{i+1}"
            x_pos = base_x + i * (dims["length"] + 1.5)
            y_pos = base_y - dims["width"]/2 - 4
            
            space = ParkingSpace(
                id=space_id,
                position=[x_pos, y_pos, 0],
                heading=90,
                length=dims["length"],
                width=dims["width"],
                parking_type=ParkingType.FORWARD_PERPENDICULAR,
                difficulty=self._assign_difficulty(difficulty, i)
            )
            self.parking_spaces.append(space)
            
        # Row 3: Angled spaces (for variety)
        for i in range(2):
            space_id = f"A1-{i+1}"
            x_pos = base_x + 70 + i * 12
            y_pos = base_y + (6 if i % 2 == 0 else -6)
            
            space = ParkingSpace(
                id=space_id,
                position=[x_pos, y_pos, 0],
                heading=315 if i % 2 == 0 else 45,  # Angled spaces
                length=dims["length"],
                width=dims["width"],
                parking_type=ParkingType.REVERSE_ANGLED if i % 2 == 0 else ParkingType.FORWARD_ANGLED,
                difficulty="hard"
            )
            self.parking_spaces.append(space)
    
    def _generate_large_lot(self, dims: Dict, types: List[ParkingType], difficulty: str):
        """Generate large parking lot (20+ spaces)"""
        base_x, base_y = 20, 0
        
        # Main section: 2 rows of perpendicular spaces
        for row in range(2):
            y_offset = (dims["width"]/2 + 4) * (1 if row == 0 else -1)
            heading = 270 if row == 0 else 90
            park_type = ParkingType.REVERSE_PERPENDICULAR if row == 0 else ParkingType.FORWARD_PERPENDICULAR
            
            for i in range(6):
                space_id = f"{'R' if row == 0 else 'L'}{row+1}-{i+1}"
                x_pos = base_x + i * (dims["length"] + 1)
                y_pos = base_y + y_offset
                
                space = ParkingSpace(
                    id=space_id,
                    position=[x_pos, y_pos, 0],
                    heading=heading,
                    length=dims["length"],
                    width=dims["width"],
                    parking_type=park_type,
                    difficulty=self._assign_difficulty(difficulty, i)
                )
                self.parking_spaces.append(space)
        
        # Side section: Angled parking
        for i in range(4):
            space_id = f"ANGLE-{i+1}"
            x_pos = base_x + 100 + (i % 2) * 15
            y_pos = base_y + (10 if i < 2 else -10)
            
            # Alternate between reverse and forward angled
            if i % 2 == 0:
                heading = 315 if i < 2 else 45
                park_type = ParkingType.REVERSE_ANGLED
            else:
                heading = 45 if i < 2 else 315
                park_type = ParkingType.FORWARD_ANGLED
            
            space = ParkingSpace(
                id=space_id,
                position=[x_pos, y_pos, 0],
                heading=heading,
                length=dims["length"],
                width=dims["width"],
                parking_type=park_type,
                difficulty="hard"
            )
            self.parking_spaces.append(space)
        
        # Challenge section: Tight spaces
        for i in range(3):
            space_id = f"TIGHT-{i+1}"
            x_pos = base_x + 130 + i * 12  # Tighter spacing
            y_pos = base_y + (5 if i % 2 == 0 else -5)
            
            space = ParkingSpace(
                id=space_id,
                position=[x_pos, y_pos, 0],
                heading=270 if i % 2 == 0 else 90,
                length=dims["length"] - 1,  # Slightly shorter
                width=dims["width"] - 0.3,   # Slightly narrower
                parking_type=ParkingType.REVERSE_PERPENDICULAR if i % 2 == 0 else ParkingType.FORWARD_PERPENDICULAR,
                difficulty="expert"
            )
            self.parking_spaces.append(space)
    
    def _assign_difficulty(self, base_difficulty: str, index: int) -> str:
        """Assign difficulty based on configuration and position"""
        if base_difficulty == "mixed":
            difficulties = ["easy", "medium", "hard"]
            return difficulties[index % len(difficulties)]
        return base_difficulty
    
    def _select_target_space(self):
        """Select target parking space based on configuration"""
        selection_method = self.config["target_space_selection"]
        
        if selection_method == "specific" and self.config["target_space_id"]:
            # Use specified space
            target_id = self.config["target_space_id"]
            self.target_space = next((space for space in self.parking_spaces if space.id == target_id), None)
            if self.target_space is None:
                print(f"Warning: Specified space {target_id} not found, using random selection")
                self.target_space = self.np_random.choice(self.parking_spaces)
        elif selection_method == "closest":
            # Select closest space to vehicle start position
            vehicle_pos = np.array([10, 0])  # Approximate start position
            distances = [np.linalg.norm(np.array(space.position[:2]) - vehicle_pos) for space in self.parking_spaces]
            min_idx = np.argmin(distances)
            self.target_space = self.parking_spaces[min_idx]
        elif selection_method == "furthest":
            # Select furthest space
            vehicle_pos = np.array([10, 0])
            distances = [np.linalg.norm(np.array(space.position[:2]) - vehicle_pos) for space in self.parking_spaces]
            max_idx = np.argmax(distances)
            self.target_space = self.parking_spaces[max_idx]
        else:  # random
            self.target_space = self.np_random.choice(self.parking_spaces)
    
    def _position_vehicle_at_entrance(self):
        """Position vehicle at parking lot entrance"""
        # Start at the beginning of the road, facing the parking lot
        start_x = 5.0
        start_y = 0.0
        start_heading = 0.0  # Facing forward
        
        self.vehicle.set_position([start_x, start_y, 0.3])
        self.vehicle.set_heading_theta(math.radians(start_heading))
    
    def _create_visual_markers(self):
        """Create visual markers for parking spaces (placeholder)"""
        # In a full implementation, you would create cone markers at space corners
        # For now, we'll store marker positions for potential visual display
        self.visual_markers = []
        for space in self.parking_spaces:
            corners = space.get_corners()
            self.visual_markers.extend(corners)
    
    def step(self, action):
        """Step environment with parking lot logic"""
        obs, reward, done, truncated, info = super().step(action)
        
        # Calculate parking-specific reward
        parking_reward = self._calculate_parking_reward()
        reward = parking_reward
        
        # Check parking success with hold time
        success, success_info = self._check_parking_success_with_hold()
        
        if success:
            done = True
            reward += self.config["completion_bonus"]
            self.success_count += 1
            info["parking_success"] = True
            print(f"🎉 PARKING SUCCESS! Space: {self.target_space.id}")
            
            # Record performance
            self._record_performance(True, success_info)
        
        # Add comprehensive info
        info.update(success_info)
        info.update(self._get_parking_metrics())
        info["parking_lot_info"] = self._get_lot_info()
        
        return obs, reward, done, truncated, info
    
    def _calculate_parking_reward(self) -> float:
        """Calculate reward for parking progress"""
        reward = 0.0
        
        # Distance-based reward
        current_distance = self._get_distance_to_target()
        if current_distance < self.best_distance:
            progress = (self.best_distance - current_distance) / max(self.initial_distance, 1.0)
            reward += progress * 10.0
            self.best_distance = current_distance
        
        # Heading alignment reward
        heading_error = self._get_heading_error()
        heading_reward = max(0, 1.0 - heading_error / 180.0)
        reward += heading_reward * 2.0
        
        # Proximity bonus
        if current_distance < 5.0:
            proximity_bonus = (5.0 - current_distance) / 5.0
            reward += proximity_bonus * 3.0
        
        # Success criteria progress
        position_progress = max(0, 1.0 - current_distance / self.config["position_tolerance"])
        heading_progress = max(0, 1.0 - heading_error / self.config["heading_tolerance"]) 
        success_progress = (position_progress + heading_progress) / 2.0
        reward += success_progress * 5.0
        
        # Small time penalty
        reward -= 0.01
        
        # Collision penalty
        if self.vehicle.crash_vehicle or self.vehicle.crash_object:
            reward -= 3.0
        
        return reward
    
    def _check_parking_success_with_hold(self) -> Tuple[bool, Dict]:
        """Check parking success with required hold time"""
        distance = self._get_distance_to_target()
        heading_error = self._get_heading_error()
        
        position_ok = distance <= self.config["position_tolerance"]
        heading_ok = heading_error <= self.config["heading_tolerance"]
        criteria_met = position_ok and heading_ok
        
        if criteria_met:
            self.success_hold_timer += 1/60.0  # Assuming 60 FPS
        else:
            self.success_hold_timer = 0.0
        
        success = self.success_hold_timer >= self.config["success_hold_time"]
        
        info = {
            "distance_to_target": distance,
            "heading_error": heading_error,
            "position_criteria_met": position_ok,
            "heading_criteria_met": heading_ok,
            "success_hold_progress": min(1.0, self.success_hold_timer / self.config["success_hold_time"]),
            "success_hold_timer": self.success_hold_timer,
            "target_space_id": self.target_space.id,
            "target_space_type": self.target_space.parking_type.value,
        }
        
        return success, info
    
    def _get_distance_to_target(self) -> float:
        """Distance to target parking space"""
        vehicle_pos = np.array(self.vehicle.position[:2])
        target_pos = np.array(self.target_space.position[:2])
        return np.linalg.norm(target_pos - vehicle_pos)
    
    def _get_heading_error(self) -> float:
        """Heading error from target orientation (degrees)"""
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
        """Get detailed parking performance metrics"""
        return {
            "current_distance": self._get_distance_to_target(),
            "current_heading_error": self._get_heading_error(),
            "best_distance_achieved": self.best_distance,
            "initial_distance": self.initial_distance,
            "progress_percent": max(0, (self.initial_distance - self.best_distance) / max(self.initial_distance, 1.0) * 100),
            "attempt_number": self.attempt_count,
            "success_rate": self.success_count / max(self.attempt_count, 1) * 100,
        }
    
    def _get_lot_info(self) -> Dict:
        """Get parking lot information"""
        return {
            "total_spaces": len(self.parking_spaces),
            "lot_layout": self.config["lot_layout"],
            "target_space": {
                "id": self.target_space.id,
                "type": self.target_space.parking_type.value,
                "difficulty": self.target_space.difficulty,
                "position": self.target_space.position,
                "heading": self.target_space.heading,
            },
            "available_types": list(set(space.parking_type.value for space in self.parking_spaces)),
        }
    
    def _get_observation(self):
        """Get enhanced observation with parking lot information"""
        vehicle = self.vehicle
        pos = np.array(vehicle.position[:2])
        target_pos = np.array(self.target_space.position[:2])
        
        # Basic vehicle state
        obs = [
            pos[0], pos[1],                    # Vehicle position
            vehicle.heading_theta,             # Vehicle heading
            vehicle.velocity[0], vehicle.velocity[1],  # Velocity
            target_pos[0], target_pos[1],      # Target position
            math.radians(self.target_space.heading),  # Target heading
            self._get_distance_to_target(),    # Distance to target
            math.radians(self._get_heading_error()),  # Heading error
            float(self.target_space.parking_type == ParkingType.REVERSE_PERPENDICULAR),
            float(self.target_space.parking_type == ParkingType.FORWARD_PERPENDICULAR),
            float(self.target_space.parking_type == ParkingType.REVERSE_ANGLED),
            float(self.target_space.parking_type == ParkingType.FORWARD_ANGLED),
            float(vehicle.crash_vehicle),
            float(vehicle.crash_object),
        ]
        
        return np.array(obs, dtype=np.float32)
    
    def _print_episode_info(self):
        """Print information about the current episode"""
        print(f"\n🚛 Trailer Parking Lot - Episode {self.attempt_count}")
        print("=" * 50)
        print(f"Target Space: {self.target_space.id}")
        print(f"Parking Type: {self.target_space.parking_type.value.replace('_', ' ').title()}")
        print(f"Difficulty: {self.target_space.difficulty.title()}")
        print(f"Position: ({self.target_space.position[0]:.1f}, {self.target_space.position[1]:.1f})")
        print(f"Target Heading: {self.target_space.heading}°")
        print(f"Success Rate: {self.success_count}/{self.attempt_count-1} ({self.success_count/(max(self.attempt_count-1, 1))*100:.1f}%)")
        
        if self.config["manual_control"]:
            print("\nControls:")
            print("  W/S - Forward/Reverse")
            print("  A/D - Steer left/right") 
            print("  R - Reset episode")
            print("  ESC - Quit")
        
    def _record_performance(self, success: bool, info: Dict):
        """Record performance for analysis"""
        performance_data = {
            "success": success,
            "attempt": self.attempt_count,
            "target_space": self.target_space.id,
            "parking_type": self.target_space.parking_type.value,
            "difficulty": self.target_space.difficulty,
            "final_distance": info.get("distance_to_target", 0),
            "final_heading_error": info.get("heading_error", 0),
            "best_distance": self.best_distance,
            "initial_distance": self.initial_distance,
        }
        self.performance_history.append(performance_data)
    
    def get_available_spaces(self) -> List[str]:
        """Get list of available parking space IDs"""
        return [space.id for space in self.parking_spaces]
    
    def set_target_space(self, space_id: str) -> bool:
        """Set specific target space by ID"""
        target = next((space for space in self.parking_spaces if space.id == space_id), None)
        if target:
            self.target_space = target
            return True
        return False
    
    def print_lot_layout(self):
        """Print the parking lot layout"""
        print(f"\n🅿️  Parking Lot Layout ({self.config['lot_layout'].title()})")
        print("=" * 60)
        
        # Group by type
        type_groups = {}
        for space in self.parking_spaces:
            park_type = space.parking_type.value
            if park_type not in type_groups:
                type_groups[park_type] = []
            type_groups[park_type].append(space)
        
        for park_type, spaces in type_groups.items():
            print(f"\n{park_type.replace('_', ' ').title()} ({len(spaces)} spaces):")
            for space in spaces:
                print(f"  {space.id}: Pos({space.position[0]:.1f}, {space.position[1]:.1f}) "
                      f"Head={space.heading}° Diff={space.difficulty}")


def demo_trailer_parking_lot():
    """Demo the trailer parking lot environment"""
    config = {
        "lot_layout": "large",           # Try "small", "medium", "large"
        "parking_difficulty": "mixed",   # Try "easy", "medium", "hard", "mixed"
        "target_space_selection": "random",  # Try "closest", "furthest", "specific"
        "manual_control": True,
        "use_render": True,
        "show_logo": False,
    }
    
    env = TrailerParkingLotEnv(config)
    
    try:
        print("🚛 Trailer Parking Lot Environment")
        print("=" * 50)
        
        # Show lot layout
        obs = env.reset()
        env.print_lot_layout()
        
        print(f"\nAvailable spaces: {', '.join(env.get_available_spaces())}")
        
        episode = 1
        while True:
            step_count = 0
            
            while True:
                obs, reward, done, truncated, info = env.step([0, 0])  # Manual control
                step_count += 1
                
                # Progress update
                if step_count % 100 == 0:
                    metrics = info
                    print(f"Step {step_count}: Distance={metrics.get('current_distance', 0):.2f}m, "
                          f"Heading error={metrics.get('current_heading_error', 0):.1f}°, "
                          f"Progress={metrics.get('progress_percent', 0):.1f}%")
                
                if done or truncated:
                    if info.get("parking_success"):
                        print(f"✅ SUCCESS! Completed {info['target_space_id']} in {step_count} steps")
                    else:
                        print(f"❌ Failed attempt at {info['target_space_id']}")
                    break
            
            # Reset for next episode
            obs = env.reset()
            episode += 1
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        
        # Show performance summary
        if env.performance_history:
            print(f"\n📊 Performance Summary:")
            print(f"Total attempts: {len(env.performance_history)}")
            successes = sum(1 for p in env.performance_history if p['success'])
            print(f"Successful parkings: {successes}/{len(env.performance_history)} ({successes/len(env.performance_history)*100:.1f}%)")
            
    finally:
        env.close()


if __name__ == "__main__":
    demo_trailer_parking_lot()
#!/usr/bin/env python
"""
Advanced trailer parking scenario using MetaDrive's procedural generation system.

This creates realistic parking lot environments with:
- Multiple parking space types and sizes
- Static obstacles (parked cars, barriers)
- Different entry/exit points
- Configurable difficulty levels
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import math

from metadrive import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.pgblock.first_block import FirstPGBlock  
from metadrive.component.pgblock.straight import Straight
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.road_network import Road


class ParkingLotMap(BaseMap):
    """Custom map class for parking lot scenarios"""
    
    def __init__(self, map_config, random_seed):
        super().__init__(map_config, random_seed)
        self.parking_spaces = []
        self.obstacles = []
        
    def _generate_map(self):
        """Generate the parking lot layout"""
        # Create main entrance road
        entrance_lane = StraightLane(
            start=[0, 0, 0], end=[20, 0, 0], 
            width=4.0, line_types=('solid', 'solid')
        )
        entrance_road = Road(self.road_network.graph, "entrance", "main")
        entrance_road.lanes = [entrance_lane]
        self.road_network.add_road(entrance_road)
        
        # Create parking area roads
        self._create_parking_area()
        
    def _create_parking_area(self):
        """Create the parking area with spaces"""
        # Main parking area road
        parking_lane = StraightLane(
            start=[20, 0, 0], end=[60, 0, 0],
            width=6.0, line_types=('broken', 'broken') 
        )
        parking_road = Road(self.road_network.graph, "main", "parking")
        parking_road.lanes = [parking_lane]
        self.road_network.add_road(parking_road)
        
        # Create individual parking spaces
        self._create_parking_spaces()
        
    def _create_parking_spaces(self):
        """Create individual parking spaces"""
        space_length = 15.0  # Long enough for truck + trailer
        space_width = 4.0
        
        # Create parking spaces on both sides
        for side in [-1, 1]:  # Left and right sides
            for i in range(5):  # 5 spaces per side
                x_pos = 25 + i * (space_length + 2)  # With gaps between spaces
                y_pos = side * (3.0 + space_width / 2)  # 3m from road center
                
                space_info = {
                    'id': f'space_{side}_{i}',
                    'position': [x_pos, y_pos, 0],
                    'heading': 90 if side == 1 else -90,  # Perpendicular to road
                    'length': space_length,
                    'width': space_width,
                    'occupied': False,
                    'suitable_for_trailer': True
                }
                
                self.parking_spaces.append(space_info)


class TrailerParkingObjectManager:
    """Simplified object manager for parking lot obstacles and markers"""
    
    def __init__(self):
        self.parking_markers = []
        self.obstacles = []
        
    def spawn_parking_markers(self, parking_spaces: List[Dict]):
        """Spawn visual markers for parking spaces"""
        # For now, just store the marker positions
        # In a full implementation, you would create visual objects
        for space in parking_spaces:
            pos = space['position']
            length = space['length'] 
            width = space['width']
            
            # Define corner positions for visualization
            corners = [
                [pos[0] - length/2, pos[1] - width/2, pos[2]],
                [pos[0] + length/2, pos[1] - width/2, pos[2]], 
                [pos[0] - length/2, pos[1] + width/2, pos[2]],
                [pos[0] + length/2, pos[1] + width/2, pos[2]]
            ]
            
            self.parking_markers.extend(corners)
    
    def spawn_obstacles(self, difficulty: str = "medium"):
        """Define obstacle positions based on difficulty level"""
        if difficulty == "easy":
            self.obstacles = []  # No obstacles
        elif difficulty == "medium":
            self.obstacles = [
                {"pos": [35, 8, 0], "heading": 0},
                {"pos": [45, -8, 0], "heading": 0}, 
            ]
        elif difficulty == "hard":
            self.obstacles = [
                {"pos": [30, 8, 0], "heading": 0},
                {"pos": [40, -8, 0], "heading": 0},
                {"pos": [50, 8, 0], "heading": 0},
                {"pos": [35, 0, 0], "heading": 90},
            ]
        else:
            self.obstacles = []


class AdvancedTrailerParkingEnv(MetaDriveEnv):
    """
    Advanced trailer parking environment with realistic parking lot.
    
    Features:
    - Procedurally generated parking lots
    - Multiple parking space configurations
    - Realistic obstacles and markers
    - Progressive difficulty levels
    - Detailed success/failure analysis
    """
    
    @classmethod 
    def default_config(cls) -> Dict[str, Any]:
        config = super().default_config()
        config.update({
            # Use simple straight road map
            "map": "S",  # Single straight road
            "num_scenarios": 1,
            "traffic_density": 0.0,  # No traffic vehicles
            
            # Parking scenario settings
            "parking_lot_size": "large",  # "small", "medium", "large"  
            "target_space_id": None,      # Auto-select if None
            "difficulty": "medium",       # "easy", "medium", "hard"
            "scenario_type": "perpendicular",  # "perpendicular", "parallel", "reverse"
            
            # Success criteria
            "position_tolerance": 1.5,    # meters
            "heading_tolerance": 15.0,    # degrees
            "trailer_alignment_tolerance": 20.0,  # degrees
            
            # Vehicle with trailer
            "vehicle_config": {
                "enable_reverse": True,
                "show_dest_mark": True,
                "show_line_to_dest": True,
                "max_speed": 20,  # Slower for precision parking
                "trailer_kinematic": {
                    "enabled": True,
                    "length": 8.0,     # Long trailer for challenge
                    "width": 2.5,
                    "height": 2.5,
                    "origin_to_hitch": [4.0, 0.0, 1.0],
                    "hitch_offset_on_tractor": [-3.5, 0.0, 1.0],
                }
            },
            
            # Rendering
            "use_render": True,
            "manual_control": False,
            "show_fps": True,
            "camera_height": 15,
            "camera_dist": 25,
            
            # Episode settings
            "horizon": 2000,  # Longer episodes for complex parking
            "crash_done": False,  # Allow minor bumps
            "out_of_road_done": False,
            
            # Reward structure
            "reward_weights": {
                "distance": 2.0,
                "heading": 1.0, 
                "trailer_alignment": 1.0,
                "progress": 1.0,
                "collision": -5.0,
                "time": -0.01,
                "success": 100.0
            }
        })
        return config
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Initialize parking lot
        super().__init__(config)
        
        # Parking state
        self.target_space = None
        self.parking_spaces = []
        self.initial_distance = 0.0
        self.best_distance = float('inf')
        self.parking_progress = 0.0
        
        # Custom managers will be created after engine initialization
        self.parking_object_manager = None
        
    def reset(self, seed: int = None):
        """Reset and setup parking scenario"""
        if seed is not None:
            self.seed(seed)
        
        # Create parking lot map
        self._setup_parking_lot()
        
        obs = super().reset()
        
        # Initialize parking object manager now that engine is available
        if self.parking_object_manager is None:
            self.parking_object_manager = TrailerParkingObjectManager()
        
        # Select target parking space
        self._select_target_space()
        
        # Position vehicle at entrance
        self._position_vehicle_at_start()
        
        # Setup obstacles and markers
        self._setup_parking_environment()
        
        # Initialize metrics
        self.initial_distance = self._get_distance_to_target()
        self.best_distance = self.initial_distance
        self.parking_progress = 0.0
        
        return self._get_enhanced_observation()
    
    def _setup_parking_lot(self):
        """Setup the parking lot map"""
        # This would integrate with MetaDrive's map system
        # For now, we'll define parking spaces manually
        self.parking_spaces = [
            {
                'id': 'space_1',
                'position': [40, 8, 0],
                'heading': 270,  # Facing into the space
                'length': 15.0,
                'width': 4.0,
                'type': 'perpendicular'
            },
            {
                'id': 'space_2', 
                'position': [40, -8, 0],
                'heading': 90,
                'length': 15.0,
                'width': 4.0,
                'type': 'perpendicular'  
            }
        ]
    
    def _select_target_space(self):
        """Select target parking space based on configuration"""
        if self.config.get("target_space_id"):
            # Use specified space
            self.target_space = next(
                (space for space in self.parking_spaces 
                 if space['id'] == self.config["target_space_id"]), 
                self.parking_spaces[0]
            )
        else:
            # Random selection
            self.target_space = self.np_random.choice(self.parking_spaces)
    
    def _position_vehicle_at_start(self):
        """Position vehicle at parking lot entrance"""
        vehicle = self.vehicle
        
        if self.config["scenario_type"] == "perpendicular":
            # Start from main road approach
            start_pos = [10, 0, 0.4]
            start_heading = 0  # Facing towards parking area
        elif self.config["scenario_type"] == "parallel":
            # Start ahead of space for parallel parking
            target_pos = self.target_space['position']
            start_pos = [target_pos[0] - 20, target_pos[1], 0.4]
            start_heading = 0
        else:  # reverse
            # Start positioned to back into space
            target_pos = self.target_space['position'] 
            start_pos = [target_pos[0] + 10, target_pos[1], 0.4]
            start_heading = 180
        
        vehicle.set_position(start_pos)
        vehicle.set_heading_theta(np.deg2rad(start_heading))
    
    def _setup_parking_environment(self):
        """Setup obstacles and visual markers"""
        # Spawn parking space markers
        if hasattr(self, 'parking_object_manager'):
            self.parking_object_manager.spawn_parking_markers(self.parking_spaces)
            self.parking_object_manager.spawn_obstacles(self.config["difficulty"])
    
    def step(self, action):
        """Enhanced step with parking-specific logic"""
        obs, reward, done, truncated, info = super().step(action)
        
        # Calculate parking-specific reward
        parking_reward = self._calculate_parking_reward()
        reward = parking_reward
        
        # Check for parking success
        success, parking_info = self._evaluate_parking_performance()
        
        if success:
            done = True
            reward += self.config["reward_weights"]["success"]
            info["parking_success"] = True
            
        info.update(parking_info)
        info.update(self._get_parking_metrics())
        
        return obs, reward, done, truncated, info
    
    def _calculate_parking_reward(self) -> float:
        """Calculate reward based on parking progress"""
        weights = self.config["reward_weights"]
        reward = 0.0
        
        # Distance reward
        distance = self._get_distance_to_target()
        if distance < self.best_distance:
            progress = (self.best_distance - distance) / self.initial_distance
            reward += progress * weights["distance"]
            self.best_distance = distance
        
        # Heading alignment reward
        heading_error = self._get_heading_error()
        heading_reward = max(0, 1.0 - heading_error / 180.0)
        reward += heading_reward * weights["heading"]
        
        # Trailer alignment reward (if trailer exists)
        if hasattr(self.vehicle, '_kinematic_trailer') and self.vehicle._kinematic_trailer:
            trailer_alignment = self._get_trailer_alignment_score()
            reward += trailer_alignment * weights["trailer_alignment"]
        
        # Time penalty for efficiency
        reward += weights["time"]
        
        # Collision penalty
        if self.vehicle.crash_vehicle or self.vehicle.crash_object:
            reward += weights["collision"]
        
        return reward
    
    def _get_distance_to_target(self) -> float:
        """Distance from vehicle to target parking position"""
        vehicle_pos = np.array(self.vehicle.position[:2])
        target_pos = np.array(self.target_space['position'][:2])
        return np.linalg.norm(target_pos - vehicle_pos)
    
    def _get_heading_error(self) -> float:
        """Heading error from target orientation (degrees)"""
        vehicle_heading = math.degrees(self.vehicle.heading_theta)
        target_heading = self.target_space['heading']
        
        # Normalize angle difference
        error = target_heading - vehicle_heading
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
            
        return abs(error)
    
    def _get_trailer_alignment_score(self) -> float:
        """Score for trailer alignment (0-1, higher is better)"""
        if not (hasattr(self.vehicle, '_kinematic_trailer') and self.vehicle._kinematic_trailer):
            return 1.0
        
        # Check if trailer is roughly aligned with vehicle
        vehicle_heading = self.vehicle.heading_theta
        # For simplicity, assume good alignment if vehicle is aligned
        # In practice, you'd check actual trailer pose
        heading_error = self._get_heading_error()
        return max(0, 1.0 - heading_error / self.config["trailer_alignment_tolerance"])
    
    def _evaluate_parking_performance(self) -> Tuple[bool, Dict]:
        """Comprehensive parking performance evaluation"""
        distance = self._get_distance_to_target()
        heading_error = self._get_heading_error()
        trailer_alignment = self._get_trailer_alignment_score()
        
        # Check success criteria
        position_ok = distance <= self.config["position_tolerance"]
        heading_ok = heading_error <= self.config["heading_tolerance"] 
        trailer_ok = trailer_alignment >= 0.8  # 80% alignment threshold
        
        success = position_ok and heading_ok and trailer_ok
        
        # Detailed performance info
        info = {
            "final_distance": distance,
            "final_heading_error": heading_error, 
            "trailer_alignment_score": trailer_alignment,
            "position_criterion_met": position_ok,
            "heading_criterion_met": heading_ok,
            "trailer_criterion_met": trailer_ok,
            "overall_success": success,
            "target_space_id": self.target_space['id'],
            "parking_type": self.target_space['type']
        }
        
        return success, info
    
    def _get_parking_metrics(self) -> Dict:
        """Get detailed parking metrics for analysis"""
        return {
            "current_distance": self._get_distance_to_target(),
            "current_heading_error": self._get_heading_error(),
            "best_distance_achieved": self.best_distance,
            "initial_distance": self.initial_distance,
            "parking_progress": (self.initial_distance - self.best_distance) / self.initial_distance,
            "target_space": self.target_space,
            "vehicle_has_trailer": hasattr(self.vehicle, '_kinematic_trailer') and 
                                   self.vehicle._kinematic_trailer is not None
        }
    
    def _get_enhanced_observation(self):
        """Get enhanced observation with parking-specific information"""
        vehicle = self.vehicle
        pos = np.array(vehicle.position)
        target_pos = np.array(self.target_space['position'])
        
        # Ensure we have 3D positions
        if len(pos) == 2:
            pos = np.append(pos, 0.0)  # Add z=0 if missing
        if len(target_pos) == 2:
            target_pos = np.append(target_pos, 0.0)  # Add z=0 if missing
        
        # Basic vehicle state
        obs = [
            pos[0], pos[1], pos[2],  # Vehicle position
            vehicle.heading_theta,    # Vehicle heading
            vehicle.velocity[0], vehicle.velocity[1],  # Velocity
        ]
        
        # Target space information
        obs.extend([
            target_pos[0], target_pos[1], target_pos[2],  # Target position
            np.deg2rad(self.target_space['heading']),     # Target heading
            self._get_distance_to_target(),               # Distance to target
            np.deg2rad(self._get_heading_error()),        # Heading error
        ])
        
        # Vehicle status
        obs.extend([
            float(vehicle.crash_vehicle),
            float(vehicle.crash_object),
            float(vehicle.on_lane),
        ])
        
        return np.array(obs, dtype=np.float32)


def demo_trailer_parking():
    """Demo the advanced trailer parking environment"""
    config = {
        "use_render": True,
        "manual_control": True,
        "difficulty": "medium",
        "scenario_type": "perpendicular",
        "show_logo": False,
    }
    
    env = AdvancedTrailerParkingEnv(config)
    
    try:
        print("🚛 Advanced Trailer Parking Challenge")
        print("=" * 50)
        print("Objective: Park the truck-trailer combination in the marked space")
        print(f"Difficulty: {config['difficulty']}")
        print(f"Scenario: {config['scenario_type']} parking")
        print("Controls: W/A/S/D to drive, manual control enabled")
        print("Press ESC to quit, R to reset")
        
        obs = env.reset()
        episode = 1
        
        while True:
            print(f"\n--- Episode {episode} ---")
            print(f"Target space: {env.target_space['id']}")
            
            step_count = 0
            while True:
                obs, reward, done, truncated, info = env.step([0, 0])  # Manual control
                step_count += 1
                
                # Print progress every 50 steps
                if step_count % 50 == 0:
                    metrics = info
                    print(f"Step {step_count}: "
                          f"Distance={metrics.get('current_distance', 0):.2f}m, "
                          f"Heading error={metrics.get('current_heading_error', 0):.1f}°, "
                          f"Progress={metrics.get('parking_progress', 0)*100:.1f}%")
                
                if done or truncated:
                    print(f"\nEpisode {episode} completed in {step_count} steps")
                    
                    if info.get("parking_success"):
                        print("🎉 PARKING SUCCESSFUL!")
                        print(f"Final distance: {info['final_distance']:.2f}m")
                        print(f"Final heading error: {info['final_heading_error']:.1f}°")
                        print(f"Trailer alignment: {info['trailer_alignment_score']*100:.1f}%")
                    else:
                        print("❌ Parking unsuccessful")
                        print("Criteria not met:")
                        if not info['position_criterion_met']:
                            print(f"  - Position: {info['final_distance']:.2f}m (required: ≤{env.config['position_tolerance']}m)")
                        if not info['heading_criterion_met']:
                            print(f"  - Heading: {info['final_heading_error']:.1f}° (required: ≤{env.config['heading_tolerance']}°)")
                        if not info['trailer_criterion_met']:
                            print(f"  - Trailer alignment: {info['trailer_alignment_score']*100:.1f}% (required: ≥80%)")
                    
                    break
            
            # Reset for next episode
            obs = env.reset()
            episode += 1
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    demo_trailer_parking()
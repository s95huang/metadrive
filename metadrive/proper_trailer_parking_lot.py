#!/usr/bin/env python
"""
Proper Trailer Parking Lot Environment using MetaDrive's built-in PGBlock system.

This creates a REAL parking lot with visible parking spaces using MetaDrive's 
native parking lot infrastructure and extends it for trailer-specific challenges.
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple, List
import random

from metadrive.envs.marl_envs.marl_parking_lot import MultiAgentParkingLotEnv
from metadrive.component.road_network import Road
from metadrive.component.pgblock.parking_lot import ParkingLot


class ProperTrailerParkingEnv(MultiAgentParkingLotEnv):
    """
    Proper trailer parking environment using MetaDrive's real parking lot blocks.
    
    This creates an actual parking lot with visible parking spaces and extends it
    for trailer parking challenges with proper space allocation and success criteria.
    """
    
    @classmethod
    def default_config(cls) -> Dict:
        config = super().default_config()
        config.update({
            # Use single agent for simplicity
            "num_agents": 1,
            
            # Parking lot configuration - these create REAL parking spaces
            "parking_space_num": 16,  # Total parking spaces (8 per side)
            "map_config": {
                "exit_length": 25,
                "lane_num": 1,
            },
            
            # Vehicle with trailer configuration
            "vehicle_config": {
                "enable_reverse": True,
                "show_dest_mark": True,
                "show_line_to_dest": True,
                "show_navi_mark": True,
                
                # Trailer configuration optimized for parking
                "trailer_kinematic": {
                    "enabled": True,
                    "length": 8.5,     # Realistic trailer length
                    "width": 2.5,      # Trailer width 
                    "height": 2.6,     # Trailer height
                    "origin_to_hitch": [4.2, 0.0, 1.0],
                    "hitch_offset_on_tractor": [-3.2, 0.0, 1.0],
                }
            },
            
            # Camera settings for better parking lot view
            "top_down_camera_initial_x": 80,
            "top_down_camera_initial_y": 0,
            "top_down_camera_initial_z": 150,  # Higher for better view of whole lot
            
            # Trailer-specific success criteria (more lenient due to trailer complexity)
            "trailer_position_tolerance": 3.5,    # meters (larger for trailer)
            "trailer_heading_tolerance": 25.0,    # degrees
            "trailer_success_hold_time": 2.5,     # seconds to hold position
            
            # Episode settings
            "horizon": 4000,  # Longer horizon for trailer parking
            "manual_control": True,
            "use_render": True,
            "show_fps": True,
            "show_logo": False,

            # Relax early termination so episode doesn't end immediately when slightly out of lane or near lines
            "out_of_road_done": False,
            "on_continuous_line_done": False,
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "crash_human_done": False,
            
            # Parking challenge configuration
            "challenge_mode": "random",  # "random", "reverse_only", "forward_only", "specific"
            "specific_target_space": 0,  # When using "specific" mode
            "difficulty_level": "medium",  # "easy", "medium", "hard"
            
            # Ignore success/termination checks for the first N steps to avoid instant reset
            "grace_steps": 90,
        })
        return config
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Trailer parking specific state
        self.success_hold_timer = 0.0
        self.initial_distance = 0.0
        self.best_distance = float('inf')
        self.target_parking_space = None
        
        # Performance tracking
        self.episode_start_time = 0
        self.collision_count = 0
        self.max_distance_from_target = 0.0
        
        print("🅿️  Proper Trailer Parking Environment Initialized")
        print("Using MetaDrive's native PGBlock parking lot system!")
    
    def reset(self, *args, **kwargs):
        """Reset with trailer parking logic"""
        obs = super().reset(*args, **kwargs)
        
        # Reset counters
        self.success_hold_timer = 0.0
        self.collision_count = 0
        self.episode_start_time = 0
        
        # Get agent and parking space info
        agent_name = list(self.agents.keys())[0]  # Single agent
        agent = self.agents[agent_name]
        
        # Get the assigned parking space for this agent
        if (hasattr(self.engine, 'spawn_manager') and
                hasattr(self.engine.spawn_manager, 'v_dest_pair') and
                agent_name in self.engine.spawn_manager.v_dest_pair):
            self.target_parking_space = self.engine.spawn_manager.v_dest_pair[agent_name]
        else:
            # Fallback: get a random parking space
            parking_spaces = self.engine.map_manager.current_map.parking_space
            self.target_parking_space = self.np_random.choice(list(parking_spaces))
        
        # Initialize distance metrics
        self.initial_distance = self._get_distance_to_parking_space(agent)
        self.best_distance = self.initial_distance
        self.max_distance_from_target = self.initial_distance
        
        self._print_episode_info()
        
        return obs
    
    def step(self, actions):
        """Step with trailer parking logic"""
        # Ensure MultiAgent action dict exists; fill zeros for missing agents
        if not isinstance(actions, dict):
            actions = {}
        for aid in self.agents.keys():
            if aid not in actions:
                actions[aid] = np.array([0.0, 0.0], dtype=np.float32)

        obs, reward, done, truncated, info = super().step(actions)
        self.episode_start_time += 1

        # Check if we have any agents left
        if not self.agents:
            return obs, reward, done, truncated, info

        # Get the single agent
        agent_name = list(self.agents.keys())[0]
        agent = self.agents[agent_name]

        # Calculate trailer parking reward
        parking_reward = self._calculate_trailer_parking_reward(agent)

        # Check parking success
        success, success_info = self._check_trailer_parking_success(agent)

        # Update reward for the agent
        if isinstance(reward, dict):
            reward[agent_name] = parking_reward
        else:
            reward = parking_reward

        # Force done to False unless we have genuine success - ignore parent's done status
        if isinstance(done, dict):
            done[agent_name] = success
        else:
            done = success

        # Update done status if successful
        if success:
            space_info = self._get_parking_space_info()
            print(f"🎉 TRAILER PARKING SUCCESS!")
            print(f"Successfully parked in space: {space_info}")
            print(f"Time taken: {self.episode_start_time} steps")
        
        # Add trailer parking info
        trailer_info = success_info.copy()
        trailer_info.update(self._get_trailer_parking_metrics(agent))
        
        if isinstance(info, dict) and agent_name in info:
            info[agent_name].update(trailer_info)
        else:
            info = trailer_info
        
        return obs, reward, done, truncated, info
    
    def _calculate_trailer_parking_reward(self, agent) -> float:
        """Calculate reward for trailer parking progress"""
        if not self.target_parking_space:
            return 0.0
            
        reward = 0.0
        
        # Distance-based reward
        current_distance = self._get_distance_to_parking_space(agent)
        self.max_distance_from_target = max(self.max_distance_from_target, current_distance)
        
        # Progress reward when getting closer
        if current_distance < self.best_distance:
            progress = (self.best_distance - current_distance) / max(self.initial_distance, 1.0)
            reward += progress * 15.0  # Higher reward for trailer progress
            self.best_distance = current_distance
        
        # Heading alignment reward
        heading_error = self._get_heading_error_to_parking_space(agent)
        heading_reward = max(0, 1.0 - heading_error / 180.0)
        reward += heading_reward * 4.0
        
        # Proximity bonus (close to parking space)
        if current_distance < 8.0:
            proximity_bonus = (8.0 - current_distance) / 8.0
            reward += proximity_bonus * 8.0
        
        # Success criteria progress
        position_progress = max(0, 1.0 - current_distance / self.config["trailer_position_tolerance"])
        heading_progress = max(0, 1.0 - heading_error / self.config["trailer_heading_tolerance"])
        success_progress = (position_progress + heading_progress) / 2.0
        reward += success_progress * 12.0
        
        # Small time penalty to encourage efficiency
        reward -= 0.03
        
        # Collision penalty (higher for trailers)
        if agent.crash_vehicle or agent.crash_object:
            reward -= 8.0
            self.collision_count += 1
        
        # Bonus for smooth driving (low acceleration changes)
        if hasattr(agent, 'last_actions') and hasattr(agent, 'current_actions'):
            smoothness = 1.0 - abs(agent.current_actions[0] - agent.last_actions[0])
            reward += smoothness * 0.5
        
        return reward
    
    def _check_trailer_parking_success(self, agent) -> Tuple[bool, Dict]:
        """Check trailer parking success with hold time"""
        if not self.target_parking_space:
            return False, {}
        
        # Grace period to prevent instant success/termination right after reset
        if self.episode_start_time < self.config.get("grace_steps", 0):
            return False, {
                "parking_success": False,
                "success_hold_progress": 0.0,
                "grace_remaining": max(0, self.config.get("grace_steps", 0) - self.episode_start_time)
            }
        
        distance = self._get_distance_to_parking_space(agent)
        heading_error = self._get_heading_error_to_parking_space(agent)
        
        position_ok = distance <= self.config["trailer_position_tolerance"]
        heading_ok = heading_error <= self.config["trailer_heading_tolerance"]
        criteria_met = position_ok and heading_ok
        
        if criteria_met:
            self.success_hold_timer += 1.0/60.0  # Assuming 60 FPS
        else:
            self.success_hold_timer = 0.0
        
        # Avoid degenerate success when initial distance is effectively zero
        min_start_dist = 1.0
        success = (
            self.initial_distance >= min_start_dist and
            self.success_hold_timer >= self.config["trailer_success_hold_time"]
        )
        
        info = {
            "distance_to_parking_space": distance,
            "heading_error_to_parking_space": heading_error,
            "position_criteria_met": position_ok,
            "heading_criteria_met": heading_ok,
            "success_hold_progress": min(1.0, self.success_hold_timer / self.config["trailer_success_hold_time"]),
            "parking_success": success,
        }
        
        return success, info

    def done_function(self, vehicle_id):
        """Override termination: only succeed when trailer parking success passes grace + hold time.

        Keep base MAX_STEP/truncation metadata by querying the generic multi-agent parent, but avoid
        the ParkingLotEnv's immediate respawn/cleanup logic.
        """
        # Base multi-agent done info (without ParkingLot-specific side effects)
        from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
        done_parent, info = MultiAgentMetaDrive.done_function(self, vehicle_id)

        # Our success rule
        agent = self.agents[vehicle_id]
        success, extra = self._check_trailer_parking_success(agent)
        info.update(extra)

        # Only terminate on our success; ignore parent's termination
        return success, info

    def _is_out_of_road(self, vehicle):
        """Override the base environment's strict out-of-road check to be more lenient for trailer parking"""
        # Only terminate on major crashes, not lane violations
        # This allows the vehicle to maneuver across lanes and yellow lines during parking
        return vehicle.crash_sidewalk or vehicle.crash_vehicle or vehicle.crash_object
    
    def _get_distance_to_parking_space(self, agent) -> float:
        """Get distance to assigned parking space"""
        if not self.target_parking_space:
            print("DEBUG: No target parking space assigned")
            return 0.0

        try:
            # Get parking space center position
            parking_road = self.target_parking_space
            road_network = self.engine.map_manager.current_map.road_network
            lanes = parking_road.get_lanes(road_network)
            if lanes:
                lane = lanes[0]
                # Get center of parking space
                space_center = lane.position(lane.length / 2, 0)

                # Vehicle position (use trailer rear if available)
                agent_pos = np.array(agent.position[:2])
                if hasattr(agent, '_kinematic_trailer') and agent._kinematic_trailer:
                    # Use trailer position for more accurate parking
                    trailer_pos = agent._kinematic_trailer.position
                    agent_pos = np.array(trailer_pos[:2])

                space_pos = np.array(space_center[:2])
                distance = np.linalg.norm(space_pos - agent_pos)

                # Debug output for first few steps
                if self.episode_start_time <= 3:
                    print(f"DEBUG Distance: agent_pos={agent_pos}, space_pos={space_pos}, distance={distance}")

                return distance
            else:
                if self.episode_start_time <= 3:
                    print("DEBUG: No lanes found for parking road")
                return 50.0
        except Exception as e:
            # Fallback distance calculation
            if self.episode_start_time <= 3:
                print(f"DEBUG: Exception in distance calculation: {e}")
            pass

        return 50.0  # Default high distance if calculation fails
    
    def _get_heading_error_to_parking_space(self, agent) -> float:
        """Get heading error relative to parking space orientation"""
        if not self.target_parking_space:
            return 0.0
        
        try:
            # Get parking space orientation
            parking_road = self.target_parking_space
            road_network = self.engine.map_manager.current_map.road_network
            lanes = parking_road.get_lanes(road_network)
            if lanes:
                lane = lanes[0]
                space_heading = math.degrees(lane.heading_theta_at(lane.length / 2))
                
                # Vehicle heading (use trailer if available)
                vehicle_heading = math.degrees(agent.heading_theta)
                if hasattr(agent, '_kinematic_trailer') and agent._kinematic_trailer:
                    vehicle_heading = math.degrees(agent._kinematic_trailer.heading_theta)
                
                # Calculate heading error
                error = space_heading - vehicle_heading
                while error > 180:
                    error -= 360
                while error < -180:
                    error += 360
                    
                return abs(error)
        except Exception as e:
            pass
        
        return 90.0  # Default high error if calculation fails
    
    def _get_parking_space_info(self) -> str:
        """Get human-readable parking space information"""
        if not self.target_parking_space:
            return "Unknown"
        
        try:
            road = self.target_parking_space
            start_node = road.start_node
            end_node = road.end_node
            
            # Extract space info from road node naming
            if ParkingLot.is_out_direction_parking_space(road):
                direction = "OUT (Forward parking)"
            elif ParkingLot.is_in_direction_parking_space(road):
                direction = "IN (Reverse parking)"
            else:
                direction = "Unknown"
            
            return f"Road {start_node} → {end_node} ({direction})"
        except:
            return f"Space: {self.target_parking_space}"
    
    def _get_trailer_parking_metrics(self, agent) -> Dict:
        """Get trailer parking performance metrics"""
        return {
            "current_distance_to_space": self._get_distance_to_parking_space(agent),
            "current_heading_error": self._get_heading_error_to_parking_space(agent),
            "best_distance_achieved": self.best_distance,
            "initial_distance": self.initial_distance,
            "progress_percent": max(0, (self.initial_distance - self.best_distance) / max(self.initial_distance, 1.0) * 100),
            "collision_count": self.collision_count,
            "time_elapsed": self.episode_start_time,
            "has_trailer": hasattr(agent, '_kinematic_trailer') and agent._kinematic_trailer is not None,
            "target_parking_space_info": self._get_parking_space_info(),
            "max_distance_from_target": self.max_distance_from_target,
        }
    
    def _print_episode_info(self):
        """Print episode information"""
        print(f"\n🚛 PROPER TRAILER PARKING LOT")
        print("=" * 60)
        print(f"Environment: MetaDrive's REAL parking lot with visible spaces")
        print(f"Total Parking Spaces: {self.config['parking_space_num']}")
        
        if self.target_parking_space:
            space_info = self._get_parking_space_info()
            print(f"Assigned Parking Space: {space_info}")
        
        print(f"\nTrailer Success Criteria:")
        print(f"  Position tolerance: {self.config['trailer_position_tolerance']}m")
        print(f"  Heading tolerance: {self.config['trailer_heading_tolerance']}°")
        print(f"  Hold time: {self.config['trailer_success_hold_time']}s")
        
        print(f"\nControls:")
        print(f"  W/S - Forward/Reverse (TRAILER REVERSE ENABLED!)")
        print(f"  A/D - Steer left/right")
        print(f"  V - Toggle reverse mode")
        print(f"  R - Reset episode")
        print(f"  ESC - Quit")
        
        print(f"\nTrailer Parking Tips:")
        print(f"  • Use reverse mode (V) for backing into tight spaces")
        print(f"  • Take wide turns to account for trailer swing")
        print(f"  • Be patient - trailer parking requires precision!")
        print(f"  • Look for the destination marker and navigation line")


def demo_proper_trailer_parking():
    """Demo the proper trailer parking environment"""

    # Configuration scenarios
    scenarios = {
        "random_parking": {
            "challenge_mode": "random",
            "difficulty_level": "medium",
            "description": "Random parking space assignment",
            "use_render": True,  # Disable for debugging distance
            "manual_control": True  # Disable for debugging
        },
        "reverse_challenge": {
            "challenge_mode": "reverse_only",
            "trailer_position_tolerance": 3.0,
            "trailer_heading_tolerance": 20.0,
            "difficulty_level": "hard",
            "description": "Reverse parking challenge"
        },
        "precision_parking": {
            "challenge_mode": "random",
            "trailer_position_tolerance": 2.5,
            "trailer_heading_tolerance": 15.0,
            "trailer_success_hold_time": 3.0,
            "difficulty_level": "hard",
            "description": "High precision trailer parking"
        },
        "easy_training": {
            "challenge_mode": "random",
            "trailer_position_tolerance": 4.0,
            "trailer_heading_tolerance": 30.0,
            "difficulty_level": "easy",
            "description": "Easy training mode"
        }
    }
    
    # Select scenario
    scenario_name = "random_parking"  # Change this to try different scenarios
    config = scenarios[scenario_name]
    
    env = ProperTrailerParkingEnv(config)
    
    try:
        print(f"🚛 Proper Trailer Parking - {scenario_name.replace('_', ' ').title()}")
        print(f"Scenario: {config['description']}")
        print("This uses MetaDrive's ACTUAL parking lot with REAL visible parking spaces!")
        print("\nThe parking lot will have proper parking space markings and layout.")
        
        obs = env.reset()
        
        step_count = 0
        total_reward = 0.0
        
        while True:
            # Manual control (empty action for manual control)
            obs, reward, done, truncated, info = env.step({})
            step_count += 1
            
            if isinstance(reward, dict):
                episode_reward = list(reward.values())[0] if reward else 0
            else:
                episode_reward = reward
            total_reward += episode_reward
            
            # Progress updates every 2 seconds (120 steps at 60 FPS)
            if step_count % 120 == 0:
                # Get agent-specific metrics
                agent_name = list(env.agents.keys())[0] if env.agents else None
                if agent_name and isinstance(info, dict) and agent_name in info:
                    metrics = info[agent_name]
                elif isinstance(info, dict):
                    metrics = info
                else:
                    metrics = {}

                print(f"Step {step_count}: Distance={metrics.get('current_distance_to_space', 0):.2f}m, "
                      f"Heading error={metrics.get('current_heading_error', 0):.1f}°, "
                      f"Progress={metrics.get('progress_percent', 0):.1f}%, "
                      f"Reward={total_reward:.1f}")
            
            # Check if episode ended
            if isinstance(done, dict):
                # Only check agent-specific done status, ignore '__all__'
                agent_dones = {k: v for k, v in done.items() if k != '__all__'}
                episode_done = any(agent_dones.values())
            else:
                episode_done = done

            # Check truncation status properly
            if isinstance(truncated, dict):
                agent_truncated = {k: v for k, v in truncated.items() if k != '__all__'}
                episode_truncated = any(agent_truncated.values())
            else:
                episode_truncated = truncated

            if episode_done or episode_truncated:
                # Get agent-specific metrics for end-of-episode reporting
                agent_name = list(env.agents.keys())[0] if env.agents else None
                if agent_name and isinstance(info, dict) and agent_name in info:
                    metrics = info[agent_name]
                    success = metrics.get("parking_success", False)
                elif isinstance(info, dict):
                    metrics = info
                    success = info.get("parking_success", False)
                else:
                    success = False
                    metrics = {}
                    
                if success:
                    print(f"✅ PARKING SUCCESS! Trailer parked successfully!")
                    print(f"Final distance: {metrics.get('distance_to_parking_space', 0):.2f}m")
                    print(f"Final heading error: {metrics.get('heading_error_to_parking_space', 0):.1f}°")
                    print(f"Total reward: {total_reward:.1f}")
                    print(f"Collisions: {metrics.get('collision_count', 0)}")
                else:
                    print(f"❌ Episode ended after {step_count} steps")
                    print(f"Best distance achieved: {metrics.get('best_distance_achieved', 0):.2f}m")
                    print(f"Total reward: {total_reward:.1f}")
                
                print(f"\nStarting new episode...")
                obs = env.reset()
                step_count = 0
                total_reward = 0.0
                
    except KeyboardInterrupt:
        print("\nDemo ended by user")
    finally:
        env.close()


if __name__ == "__main__":
    demo_proper_trailer_parking()

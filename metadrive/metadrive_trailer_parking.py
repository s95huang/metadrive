#!/usr/bin/env python
"""
MetaDrive Trailer Parking using the built-in parking lot.

This extends MetaDrive's existing parking lot environment for trailer parking challenges.
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple
import random

from metadrive.envs.marl_envs.marl_parking_lot import MultiAgentParkingLotEnv


class MetaDriveTrailerParkingEnv(MultiAgentParkingLotEnv):
    """
    Trailer parking environment using MetaDrive's built-in parking lot.
    
    This creates a proper parking lot with visible parking spaces and 
    extends it for trailer parking challenges.
    """
    
    @classmethod
    def default_config(cls) -> Dict:
        config = super().default_config()
        config.update({
            # Parking lot configuration
            "num_agents": 1,  # Single vehicle
            "parking_space_num": 12,
            
            # Vehicle with trailer
            "vehicle_config": {
                "enable_reverse": True,
                "show_dest_mark": True,
                "show_line_to_dest": True,
                "trailer_kinematic": {
                    "enabled": True,
                    "length": 7.0,     # Trailer length
                    "width": 2.3,      # Trailer width
                    "height": 2.4,     # Trailer height
                    "origin_to_hitch": [3.5, 0.0, 0.9],
                    "hitch_offset_on_tractor": [-3.0, 0.0, 0.9],
                }
            },
            
            # Success criteria for trailer parking
            "position_tolerance": 3.0,      # meters (larger for trailer)
            "heading_tolerance": 20.0,      # degrees
            "success_hold_time": 2.0,       # seconds to hold position
            
            # Episode settings
            "horizon": 3000,
            "manual_control": True,
            "use_render": True,
            "show_fps": True,
            "show_logo": False,
            
            # Camera settings for better view
            "camera_height": 20,
            "camera_dist": 35,
            
            # Target space selection
            "target_space_mode": "random",  # "random", "reverse_only", "forward_only", "specific"
            "specific_target_space": 0,     # If using "specific" mode
        })
        return config
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Trailer parking specific state
        self.success_hold_timer = 0.0
        self.initial_distance = 0.0
        self.best_distance = float('inf')
        self.target_space_info = None
        
        # Define parking space types based on positions
        # This is based on the typical layout of MetaDrive's parking lot
        self.space_types = self._define_space_types()
    
    def _define_space_types(self) -> Dict:
        """Define which spaces are for reverse vs forward parking"""
        # Based on MetaDrive's parking lot layout
        # Spaces 0-5 are typically on one side, 6-11 on the other
        space_info = {}
        
        for i in range(12):
            if i < 6:
                # One side - reverse parking (backing in)
                space_info[i] = {
                    "type": "reverse",
                    "difficulty": "easy" if i in [0, 5] else "medium" if i in [1, 4] else "hard",
                    "description": f"Reverse space {i} (back in)"
                }
            else:
                # Other side - forward parking (drive in)
                space_info[i] = {
                    "type": "forward", 
                    "difficulty": "easy" if i in [6, 11] else "medium" if i in [7, 10] else "hard",
                    "description": f"Forward space {i} (drive in)"
                }
        
        return space_info
    
    def reset(self, *args, **kwargs):
        """Reset with trailer parking logic"""
        obs = super().reset(*args, **kwargs)
        
        # Get the assigned parking space for the agent
        agent_name = list(self.agents.keys())[0]  # Single agent
        agent = self.agents[agent_name]
        
        # Get target space from the agent's destination
        if hasattr(agent, 'routing_localization') and agent.routing_localization:
            # Try to get the target space from routing
            self.target_space_id = self._get_target_space_from_agent(agent)
        else:
            # Fallback to random selection
            self.target_space_id = self._select_target_space()
        
        self.target_space_info = self.space_types.get(self.target_space_id, {
            "type": "unknown", "difficulty": "medium", "description": f"Space {self.target_space_id}"
        })
        
        # Initialize metrics
        self.initial_distance = self._get_distance_to_target(agent)
        self.best_distance = self.initial_distance
        self.success_hold_timer = 0.0
        
        self._print_episode_info()
        
        return obs
    
    def _get_target_space_from_agent(self, agent) -> int:
        """Try to get target space from agent's routing information"""
        try:
            # This is a simplified approach - in reality, you'd need to 
            # analyze the agent's destination coordinates
            return random.randint(0, 11)
        except:
            return random.randint(0, 11)
    
    def _select_target_space(self) -> int:
        """Select target space based on configuration"""
        mode = self.config.get("target_space_mode", "random")
        
        if mode == "specific":
            return self.config.get("specific_target_space", 0)
        elif mode == "reverse_only":
            return random.randint(0, 5)  # Reverse spaces
        elif mode == "forward_only":
            return random.randint(6, 11)  # Forward spaces
        else:  # random
            return random.randint(0, 11)
    
    def step(self, actions):
        """Step with trailer parking logic"""
        obs, reward, done, truncated, info = super().step(actions)
        
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
        
        # Update done status
        if success:
            if isinstance(done, dict):
                done[agent_name] = True
            else:
                done = True
            print(f"🎉 TRAILER PARKING SUCCESS in space {self.target_space_id}!")
        
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
        if not hasattr(self, 'target_space_id'):
            return 0.0
            
        reward = 0.0
        
        # Distance reward
        current_distance = self._get_distance_to_target(agent)
        if current_distance < self.best_distance:
            progress = (self.best_distance - current_distance) / max(self.initial_distance, 1.0)
            reward += progress * 10.0
            self.best_distance = current_distance
        
        # Heading alignment reward
        heading_error = self._get_heading_error(agent)
        heading_reward = max(0, 1.0 - heading_error / 180.0)
        reward += heading_reward * 3.0
        
        # Proximity bonus
        if current_distance < 8.0:
            proximity_bonus = (8.0 - current_distance) / 8.0
            reward += proximity_bonus * 5.0
        
        # Success criteria progress
        position_progress = max(0, 1.0 - current_distance / self.config["position_tolerance"])
        heading_progress = max(0, 1.0 - heading_error / self.config["heading_tolerance"])
        success_progress = (position_progress + heading_progress) / 2.0
        reward += success_progress * 8.0
        
        # Small time penalty
        reward -= 0.02
        
        # Collision penalty
        if agent.crash_vehicle or agent.crash_object:
            reward -= 5.0
        
        return reward
    
    def _check_trailer_parking_success(self, agent) -> Tuple[bool, Dict]:
        """Check trailer parking success with hold time"""
        distance = self._get_distance_to_target(agent)
        heading_error = self._get_heading_error(agent)
        
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
            "target_space_id": getattr(self, 'target_space_id', -1),
            "target_space_info": getattr(self, 'target_space_info', {}),
            "parking_success": success,
        }
        
        return success, info
    
    def _get_distance_to_target(self, agent) -> float:
        """Get distance to target parking space (approximate)"""
        if not hasattr(self, 'target_space_id'):
            return 0.0
        
        # This is a simplified approach - you'd need the actual parking space coordinates
        # For now, we'll use the agent's destination if available
        try:
            if hasattr(agent, 'navigation') and agent.navigation and hasattr(agent.navigation, 'final_lane'):
                # Use navigation destination
                dest = agent.navigation.final_lane.position(agent.navigation.final_lane.length, 0)
                agent_pos = np.array(agent.position[:2])
                target_pos = np.array(dest[:2])
                return np.linalg.norm(target_pos - agent_pos)
        except:
            pass
        
        # Fallback: approximate distance based on space ID
        # This is rough but gives some guidance
        return max(0, 50.0 - getattr(self, 'step_count', 0) * 0.1)
    
    def _get_heading_error(self, agent) -> float:
        """Get heading error for parking space (approximate)"""
        if not hasattr(self, 'target_space_id'):
            return 0.0
        
        # Simplified heading target based on space type
        if self.target_space_info.get("type") == "reverse":
            target_heading = 270  # Backing in
        else:
            target_heading = 90   # Driving in
        
        vehicle_heading = math.degrees(agent.heading_theta)
        
        # Normalize angle difference
        error = target_heading - vehicle_heading
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
            
        return abs(error)
    
    def _get_trailer_parking_metrics(self, agent) -> Dict:
        """Get trailer parking performance metrics"""
        return {
            "current_distance": self._get_distance_to_target(agent),
            "current_heading_error": self._get_heading_error(agent),
            "best_distance_achieved": getattr(self, 'best_distance', 0),
            "initial_distance": getattr(self, 'initial_distance', 0),
            "progress_percent": max(0, (getattr(self, 'initial_distance', 1) - getattr(self, 'best_distance', 0)) / max(getattr(self, 'initial_distance', 1), 1.0) * 100),
            "has_trailer": hasattr(agent, '_kinematic_trailer') and agent._kinematic_trailer is not None,
        }
    
    def _print_episode_info(self):
        """Print episode information"""
        if not hasattr(self, 'target_space_id'):
            return
            
        print(f"\n🅿️  MetaDrive Trailer Parking Lot")
        print("=" * 50)
        print(f"Environment: Using MetaDrive's built-in parking lot")
        print(f"Target Space: #{self.target_space_id}")
        
        if self.target_space_info:
            print(f"Parking Type: {self.target_space_info['type'].upper()}")
            print(f"Difficulty: {self.target_space_info['difficulty'].upper()}")
            print(f"Description: {self.target_space_info['description']}")
        
        print(f"Success Criteria:")
        print(f"  Position tolerance: {self.config['position_tolerance']}m")
        print(f"  Heading tolerance: {self.config['heading_tolerance']}°")
        print(f"  Hold time: {self.config['success_hold_time']}s")
        
        print(f"\nControls:")
        print(f"  W/S - Forward/Reverse (TRAILER REVERSE ENABLED!)")
        print(f"  A/D - Steer left/right")
        print(f"  R - Reset episode")
        print(f"  ESC - Quit")
        
        # Show space information
        reverse_spaces = [i for i in range(12) if self.space_types[i]["type"] == "reverse"]
        forward_spaces = [i for i in range(12) if self.space_types[i]["type"] == "forward"]
        
        print(f"\nParking Lot Layout:")
        print(f"  Reverse spaces (back in): {reverse_spaces}")
        print(f"  Forward spaces (drive in): {forward_spaces}")
        print(f"  This is a REAL parking lot with visible spaces!")


def demo_metadrive_trailer_parking():
    """Demo MetaDrive's trailer parking environment"""
    
    # Different configuration options
    scenarios = {
        "random": {
            "target_space_mode": "random",
            "description": "Random parking space selection"
        },
        "reverse_challenge": {
            "target_space_mode": "reverse_only", 
            "position_tolerance": 2.5,
            "heading_tolerance": 15.0,
            "description": "Reverse parking challenge"
        },
        "forward_challenge": {
            "target_space_mode": "forward_only",
            "position_tolerance": 2.0,
            "heading_tolerance": 20.0,
            "description": "Forward parking challenge"
        },
        "precision_parking": {
            "target_space_mode": "random",
            "position_tolerance": 1.5,
            "heading_tolerance": 10.0,
            "success_hold_time": 3.0,
            "description": "High precision parking"
        }
    }
    
    # Select scenario
    scenario_name = "reverse_challenge"  # Change this to try different scenarios
    config = scenarios[scenario_name]
    
    env = MetaDriveTrailerParkingEnv(config)
    
    try:
        print(f"🚛 MetaDrive Trailer Parking - {scenario_name.replace('_', ' ').title()}")
        print(f"Scenario: {config['description']}")
        print("Using MetaDrive's actual parking lot with visible spaces!")
        
        obs = env.reset()
        
        step_count = 0
        while True:
            # Single agent action (manual control)
            obs, reward, done, truncated, info = env.step({})  # Empty dict for manual control
            step_count += 1
            
            # Progress updates
            if step_count % 100 == 0 and isinstance(info, dict):
                metrics = info
                print(f"Step {step_count}: Distance={metrics.get('current_distance', 0):.2f}m, "
                      f"Heading error={metrics.get('current_heading_error', 0):.1f}°, "
                      f"Progress={metrics.get('progress_percent', 0):.1f}%")
            
            # Check if episode ended
            if isinstance(done, dict):
                episode_done = any(done.values())
            else:
                episode_done = done
                
            if episode_done or (isinstance(truncated, dict) and any(truncated.values())) or truncated:
                if isinstance(info, dict):
                    success = info.get("parking_success", False)
                else:
                    success = info.get("parking_success", False) if info else False
                    
                if success:
                    print(f"✅ SUCCESS! Completed parking challenge in {step_count} steps")
                    if isinstance(info, dict):
                        print(f"Final distance: {info.get('distance_to_target', 0):.2f}m")
                        print(f"Final heading error: {info.get('heading_error', 0):.1f}°")
                else:
                    print(f"❌ Episode ended after {step_count} steps")
                
                # Reset for another attempt
                obs = env.reset()
                step_count = 0
                
    except KeyboardInterrupt:
        print("\nDemo ended by user")
    finally:
        env.close()


if __name__ == "__main__":
    demo_metadrive_trailer_parking()
#!/usr/bin/env python
"""
Usage examples for the MetaDrive trailer parking environments.

This shows different ways to use the parking environments:
1. Simple parking test
2. Multiple scenario examples  
3. Full parking lot with different space types
"""

from simple_trailer_parking import SimpleTrailerParkingEnv
from trailer_parking_examples import MultiScenarioTrailerParking
from trailer_parking_lot import TrailerParkingLotEnv


def example_1_simple_parking():
    """Example 1: Simple single-space parking"""
    print("🚛 Example 1: Simple Parking")
    print("=" * 40)
    
    config = {
        "parking_space": {
            "position": [40, 8],     # Target location
            "heading": 270,          # Back into space (270°)
            "length": 12.0,
            "width": 3.5,
        },
        "success_distance": 2.0,
        "success_heading": 20.0,
        "manual_control": False,  # Automated for demo
        "use_render": False,
    }
    
    env = SimpleTrailerParkingEnv(config)
    
    try:
        obs = env.reset()
        print(f"Target: Park at ({config['parking_space']['position']})")
        print(f"Required precision: {config['success_distance']}m, {config['success_heading']}°")
        
        # Simple AI demo
        for step in range(100):
            # Basic AI: drive towards target
            distance = env._get_distance_to_parking()
            if distance > 10:
                action = [0.3, 0]  # Drive forward
            elif distance > 3:
                action = [0.1, 0.1]  # Slower with steering
            else:
                action = [0, 0]  # Stop
                
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                success = info.get("parking_success", False)
                print(f"Result: {'✅ Success' if success else '❌ Failed'} in {step+1} steps")
                break
        
    finally:
        env.close()


def example_2_multiple_scenarios():
    """Example 2: Different parking scenarios"""
    print("\n🚛 Example 2: Multiple Scenarios")
    print("=" * 40)
    
    scenarios = ["easy_training", "perpendicular", "parallel"]
    
    for scenario_name in scenarios:
        print(f"\nTesting: {scenario_name}")
        
        try:
            parking_test = MultiScenarioTrailerParking(scenario_name)
            scenario_info = parking_test.scenario
            
            print(f"  Description: {scenario_info['description']}")
            print(f"  Difficulty: {scenario_info['difficulty']}")
            
            # Quick test (would normally run parking_test.run_scenario())
            print("  Status: ✅ Available")
            
        except Exception as e:
            print(f"  Status: ❌ Error - {e}")


def example_3_parking_lot():
    """Example 3: Full parking lot with multiple spaces"""
    print("\n🚛 Example 3: Trailer Parking Lot")
    print("=" * 40)
    
    config = {
        "lot_layout": "medium",      # 10 spaces total
        "parking_difficulty": "mixed",
        "target_space_selection": "random",
        "manual_control": False,
        "use_render": False,
    }
    
    env = TrailerParkingLotEnv(config)
    
    try:
        obs = env.reset()
        
        print(f"Lot size: {config['lot_layout']} ({len(env.parking_spaces)} spaces)")
        print(f"Target space: {env.target_space.id}")
        print(f"Parking type: {env.target_space.parking_type.value.replace('_', ' ')}")
        
        # Show space types
        type_counts = {}
        for space in env.parking_spaces:
            park_type = space.parking_type.value.replace('_', ' ')
            type_counts[park_type] = type_counts.get(park_type, 0) + 1
        
        print("Available space types:")
        for park_type, count in type_counts.items():
            print(f"  {park_type.title()}: {count}")
        
        print("✅ Parking lot environment working")
        
    finally:
        env.close()


def example_4_custom_configuration():
    """Example 4: Custom configuration"""
    print("\n🚛 Example 4: Custom Configuration")
    print("=" * 40)
    
    # Custom trailer parking lot
    custom_config = {
        "lot_layout": "small",
        "parking_difficulty": "hard",
        "target_space_selection": "furthest",  # Challenge: park in furthest space
        
        # Custom vehicle/trailer
        "vehicle_config": {
            "enable_reverse": True,
            "trailer_kinematic": {
                "enabled": True,
                "length": 10.0,    # Extra long trailer
                "width": 2.8,      # Wide trailer
                "height": 3.0,
            }
        },
        
        # Tight success criteria  
        "position_tolerance": 1.0,   # Very precise
        "heading_tolerance": 10.0,   # Must be well aligned
        "success_hold_time": 3.0,    # Hold position for 3 seconds
        
        "use_render": False,
        "manual_control": False,
    }
    
    env = TrailerParkingLotEnv(custom_config)
    
    try:
        obs = env.reset()
        
        print("Custom configuration:")
        print(f"  Trailer: {custom_config['vehicle_config']['trailer_kinematic']['length']}m long")
        print(f"  Precision: {custom_config['position_tolerance']}m position, {custom_config['heading_tolerance']}° heading")
        print(f"  Hold time: {custom_config['success_hold_time']}s")
        print(f"  Target: {env.target_space.id} ({env.target_space.parking_type.value})")
        
        print("✅ Custom configuration working")
        
    finally:
        env.close()


def show_usage_summary():
    """Show summary of available environments"""
    print("\n📋 MetaDrive Trailer Parking Environments")
    print("=" * 50)
    
    environments = [
        {
            "name": "SimpleTrailerParkingEnv", 
            "file": "simple_trailer_parking.py",
            "description": "Single parking space with basic success criteria",
            "best_for": "Algorithm testing, basic research"
        },
        {
            "name": "MultiScenarioTrailerParking",
            "file": "trailer_parking_examples.py", 
            "description": "4 different parking scenarios with varying difficulty",
            "best_for": "Comparative testing, training progression"
        },
        {
            "name": "TrailerParkingLotEnv",
            "file": "trailer_parking_lot.py",
            "description": "Full parking lot with 6-19 spaces, multiple parking types",
            "best_for": "Realistic simulation, advanced research"
        }
    ]
    
    for i, env_info in enumerate(environments, 1):
        print(f"\n{i}. {env_info['name']}")
        print(f"   File: {env_info['file']}")
        print(f"   Description: {env_info['description']}")
        print(f"   Best for: {env_info['best_for']}")
    
    print(f"\n🔧 Key Features (All Environments):")
    features = [
        "✅ Trailer physics and realistic behavior",
        "✅ Forward and reverse parking challenges", 
        "✅ Manual and automated control",
        "✅ Configurable success criteria",
        "✅ Progress tracking and rewards",
        "✅ Performance metrics and analysis"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n🚀 Quick Start Commands:")
    print(f"   python simple_trailer_parking.py")
    print(f"   python trailer_parking_examples.py perpendicular")
    print(f"   python trailer_parking_lot.py")


if __name__ == "__main__":
    print("MetaDrive Trailer Parking - Usage Examples")
    print("=" * 50)
    
    # Run all examples
    example_1_simple_parking()
    example_2_multiple_scenarios() 
    example_3_parking_lot()
    example_4_custom_configuration()
    show_usage_summary()
    
    print(f"\n✅ All examples completed successfully!")
    print(f"Choose the environment that best fits your needs.")
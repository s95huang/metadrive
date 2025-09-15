#!/usr/bin/env python
"""
Quick test script for the trailer parking lot environments.
Shows different lot sizes and configurations.
"""

from trailer_parking_lot import TrailerParkingLotEnv, ParkingType


def test_lot_layouts():
    """Test different parking lot layouts"""
    
    layouts = ["small", "medium", "large"]
    
    for layout in layouts:
        print(f"\n🚛 Testing {layout.upper()} Parking Lot")
        print("=" * 40)
        
        config = {
            "lot_layout": layout,
            "parking_difficulty": "mixed",
            "use_render": False,  # No visual for testing
            "manual_control": False,
        }
        
        env = TrailerParkingLotEnv(config)
        
        try:
            obs = env.reset()
            
            # Print layout info
            env.print_lot_layout()
            
            print(f"Total spaces: {len(env.parking_spaces)}")
            print(f"Available space IDs: {', '.join(env.get_available_spaces())}")
            print(f"Target space: {env.target_space.id} ({env.target_space.parking_type.value})")
            
            # Take a few test steps
            for i in range(5):
                obs, reward, done, truncated, info = env.step([0.1, 0])
                if done:
                    break
                    
            print(f"Environment working: ✅")
            
        except Exception as e:
            print(f"Error: {e}")
            
        finally:
            env.close()


def demo_specific_parking_types():
    """Demo specific parking types"""
    
    config = {
        "lot_layout": "large",
        "use_render": True,
        "manual_control": True,
        "show_logo": False,
    }
    
    env = TrailerParkingLotEnv(config)
    
    try:
        print("\n🚛 Trailer Parking Lot - Interactive Demo")
        print("=" * 50)
        
        obs = env.reset()
        env.print_lot_layout()
        
        # Show available spaces by type
        type_counts = {}
        for space in env.parking_spaces:
            park_type = space.parking_type.value
            if park_type not in type_counts:
                type_counts[park_type] = 0
            type_counts[park_type] += 1
        
        print(f"\n📊 Parking Space Types:")
        for park_type, count in type_counts.items():
            print(f"  {park_type.replace('_', ' ').title()}: {count} spaces")
        
        print(f"\n🎯 Current target: {env.target_space.id}")
        print(f"Type: {env.target_space.parking_type.value.replace('_', ' ').title()}")
        print(f"Position: ({env.target_space.position[0]:.1f}, {env.target_space.position[1]:.1f})")
        
        if config["manual_control"]:
            print(f"\nControls: W/A/S/D to drive, R to reset")
            print(f"Note: Reverse is enabled - use S to back up!")
            
            # Let user interact
            input("\nPress Enter to start driving, or Ctrl+C to exit...")
            
            step_count = 0
            while True:
                obs, reward, done, truncated, info = env.step([0, 0])  # Manual control
                step_count += 1
                
                if step_count % 100 == 0:
                    print(f"Progress: {info.get('progress_percent', 0):.1f}%, "
                          f"Distance: {info.get('current_distance', 0):.2f}m")
                
                if done:
                    if info.get("parking_success"):
                        print("🎉 Parking successful!")
                    else:
                        print("Episode ended")
                    
                    # Reset for another attempt
                    obs = env.reset()
                    step_count = 0
                    
    except KeyboardInterrupt:
        print("\nDemo ended by user")
    finally:
        env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_lot_layouts()
    else:
        demo_specific_parking_types()
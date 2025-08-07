#!/usr/bin/env python3
"""
Example script showing how to use the VIO-SLAM API.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_slam_example():
    """Run a complete SLAM example."""
    print("🎯 VIO-SLAM API Example")
    print("=" * 40)
    
    try:
        # Import the main components
        from vio_slam import SLAMPipeline, EuRoCDatasetLoader
        from vio_slam.features import ORBTracker
        from vio_slam.optimization import IMUPreintegrator
        
        print("✅ Successfully imported VIO-SLAM components")
        
        # Example 1: Using high-level pipeline
        print("\n📋 Example 1: High-level Pipeline API")
        config = {
            'dataset': {
                'type': 'euroc',
                'camera': 'cam0',
                'downsample_factor': 20,  # Process fewer frames for demo
            },
            'slam': {
                'window_size': 3,
                'orb_features': 500,
                'loop_closure': {
                    'enabled': False,  # Disable for quick demo
                },
            },
            'visualization': {
                'show_trajectory': False,  # Don't show plot automatically
                'save_plots': True,
            },
        }
        
        slam = SLAMPipeline(config=config)
        print("   ✅ SLAM pipeline configured")
        
        # Example 2: Using individual components
        print("\n📋 Example 2: Individual Components")
        
        # ORB Feature Tracker
        orb_tracker = ORBTracker(n_features=1000)
        print("   ✅ ORB tracker initialized")
        
        # IMU Preintegrator
        imu_preintegrator = IMUPreintegrator()
        print("   ✅ IMU preintegrator initialized")
        
        # Example 3: Dataset loader
        print("\n📋 Example 3: Dataset Operations")
        
        # Check if sample data exists
        sample_paths = ["data/mav0", "sample_data", "../datasets/euroc"]
        data_path = None
        
        for path in sample_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path:
            print(f"   📁 Found dataset at: {data_path}")
            try:
                loader = EuRoCDatasetLoader(data_path)
                if loader.validate_dataset():
                    print("   ✅ Dataset validation passed")
                    
                    # Load a small sample
                    ts_img, img_paths = loader.load_images()
                    ts_imu, gyro, accel = loader.load_imu()
                    
                    print(f"   📊 Dataset info:")
                    print(f"      - Images: {len(img_paths)}")
                    print(f"      - IMU samples: {len(ts_imu)}")
                    print(f"      - Duration: {(ts_img[-1] - ts_img[0]) / 1e9:.1f}s")
                    
                    # Get camera intrinsics
                    K = loader.get_camera_intrinsics()
                    if K is not None:
                        print(f"      - Camera fx,fy: {K[0,0]:.1f}, {K[1,1]:.1f}")
                    
                else:
                    print("   ❌ Dataset validation failed")
            except Exception as e:
                print(f"   ❌ Dataset loading failed: {e}")
        else:
            print("   📁 No sample dataset found")
            print("   💡 To test with real data:")
            print("      1. Download EuRoC MH_01_easy dataset")
            print("      2. Extract to data/mav0/")
            print("      3. Run: python run_example.py")
        
        print("\n🎉 Example completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Install: pip install -e .")
        print("   2. Run: vio-slam --help")
        print("   3. Test: python main.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\n💡 Please install dependencies:")
        print("   pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Example failed: {e}")
        return False

if __name__ == "__main__":
    success = run_slam_example()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Simple main entry point for VIO-SLAM.

This script provides a quick way to run the SLAM system without installation.
For full functionality, install the package and use the CLI interface.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point for quick testing."""
    print("🚀 VIO-SLAM Quick Start")
    print("=" * 50)
    
    # Check if we have the necessary components
    try:
        from vio_slam import SLAMPipeline
        print("✅ VIO-SLAM modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 To fix this, install dependencies:")
        print("   pip install -r requirements.txt")
        return 1
    
    # Check for data directory
    data_path = "data/mav0"
    if not os.path.exists(data_path):
        print(f"📁 No data directory found at: {data_path}")
        print("\n💡 To run SLAM, you need:")
        print("   1. Download EuRoC dataset")
        print("   2. Extract to data/mav0/")
        print("   3. Or use: vio-slam --data_path YOUR_DATA_PATH")
        
        # Show example usage
        print("\n🔧 Example usage:")
        print("   python main.py")
        print("   # or after installation:")
        print("   vio-slam --data_path data/mav0 --config config/default.yaml")
        return 0
    
    try:
        # Initialize SLAM pipeline
        print(f"🔧 Initializing SLAM pipeline...")
        slam = SLAMPipeline(config_path="config/default.yaml")
        
        # Load dataset
        print(f"📖 Loading dataset from {data_path}...")
        slam.load_dataset(data_path, dataset_type="euroc")
        
        # Run SLAM
        print("🏃 Running SLAM pipeline...")
        trajectory = slam.run()
        
        # Show results
        print(f"✅ SLAM completed successfully!")
        print(f"   Trajectory points: {len(trajectory)}")
        print(f"   Trajectory length: {slam.get_statistics().get('trajectory_length_m', 0):.2f}m")
        
        # Visualize if possible
        try:
            slam.visualize_trajectory()
            print("📊 Trajectory visualization displayed")
        except:
            print("📊 Visualization skipped (display not available)")
        
        # Save results
        output_path = "results/trajectory.pkl"
        os.makedirs("results", exist_ok=True)
        slam.save_results(output_path)
        print(f"💾 Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ SLAM failed: {e}")
        print("\n💡 Common issues:")
        print("   1. Missing dependencies: pip install -r requirements.txt")
        print("   2. Invalid data format: check dataset structure")
        print("   3. Configuration issues: check config/default.yaml")
        return 1

if __name__ == "__main__":
    sys.exit(main())
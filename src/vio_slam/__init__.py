"""
VIO-SLAM: Visual-Inertial Odometry SLAM
=======================================

A comprehensive Visual-Inertial Odometry SLAM implementation with loop closure detection.

Main Components:
- Dataset loaders for various SLAM datasets (EuRoC, TUM, KITTI)
- IMU preintegration with bias estimation
- ORB feature detection and tracking with outlier filtering
- Visual-inertial sliding window optimization
- Loop closure detection using Bag-of-Words
- Pose graph optimization for global consistency
- Real-time visualization and evaluation tools
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Import main classes for easy access
from .slam_pipeline import SLAMPipeline
from .dataset.euroc_loader import EuRoCDatasetLoader
from .features.orb_tracker import ORBTracker
from .optimization.imu_preintegrator import IMUPreintegrator
from .loop_closure.detector import LoopClosureDetector
from .optimization.pose_graph import PoseGraphOptimizer

# Version info tuple for programmatic access
VERSION_INFO = tuple(map(int, __version__.split('.')))

# Public API
__all__ = [
    # Main pipeline
    'SLAMPipeline',
    
    # Dataset loaders
    'EuRoCDatasetLoader',
    
    # Feature processing
    'ORBTracker',
    
    # Optimization components
    'IMUPreintegrator',
    'PoseGraphOptimizer',
    
    # Loop closure
    'LoopClosureDetector',
    
    # Metadata
    '__version__',
    'VERSION_INFO',
]

# Module-level configuration
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Configuration defaults
DEFAULT_CONFIG = {
    'dataset': {
        'type': 'euroc',
        'camera': 'cam0',
        'downsample_factor': 10,
    },
    'slam': {
        'window_size': 5,
        'orb_features': 1000,
        'loop_closure': {
            'enabled': True,
            'vocabulary_size': 500,
            'similarity_threshold': 0.7,
        },
        'optimization': {
            'max_iterations': 100,
            'verbose': False,
        },
    },
    'visualization': {
        'show_features': True,
        'show_trajectory': True,
        'save_plots': True,
    },
}
"""
EuRoC MAV dataset loader implementation.
"""

import os
import glob
import numpy as np
from typing import Tuple, List, Optional
import yaml
import logging

from .base_loader import DatasetLoader

logger = logging.getLogger(__name__)


class EuRoCDatasetLoader(DatasetLoader):
    """
    EuRoC MAV dataset loader for images and IMU data.
    
    Expected directory structure:
        data/mav0/
          ├── cam0/data/*.png
          ├── cam0/sensor.yaml
          ├── cam1/data/*.png  
          ├── cam1/sensor.yaml
          ├── imu0/data.csv
          ├── imu0/sensor.yaml
          └── ...
    """
    
    def __init__(self, root_dir: str):
        """
        Initialize EuRoC dataset loader.
        
        Args:
            root_dir: Path to the mav0 directory
        """
        super().__init__(root_dir)
        self.cam_dirs = {
            'cam0': os.path.join(root_dir, 'cam0', 'data'),
            'cam1': os.path.join(root_dir, 'cam1', 'data'),
        }
        self.cam_configs = {
            'cam0': os.path.join(root_dir, 'cam0', 'sensor.yaml'),
            'cam1': os.path.join(root_dir, 'cam1', 'sensor.yaml'),
        }
        self.imu_file = os.path.join(root_dir, 'imu0', 'data.csv')
        self.imu_config = os.path.join(root_dir, 'imu0', 'sensor.yaml')

    def load_images(self, camera: str = 'cam0') -> Tuple[np.ndarray, List[str]]:
        """
        Load image timestamps and file paths.
        
        Args:
            camera: Camera identifier ('cam0' or 'cam1')
            
        Returns:
            timestamps: Array of timestamps (int64, nanoseconds)
            image_paths: List of image file paths
        """
        cam_dir = self.cam_dirs.get(camera)
        if not cam_dir or not os.path.isdir(cam_dir):
            raise ValueError(f"Camera directory not found: {cam_dir}")
            
        # Get all PNG files and sort them
        files = sorted(glob.glob(os.path.join(cam_dir, '*.png')))
        if not files:
            raise ValueError(f"No PNG files found in {cam_dir}")
            
        # Extract timestamps from filenames
        timestamps = np.array([
            int(os.path.splitext(os.path.basename(f))[0]) for f in files
        ], dtype=np.int64)
        
        logger.info(f"Loaded {len(files)} images from {camera}")
        return timestamps, files

    def load_imu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load IMU data from CSV file.
        
        Expected format: timestamp[ns], gx, gy, gz, ax, ay, az
        
        Returns:
            timestamps: Array of timestamps (int64, nanoseconds)
            gyro: Array of gyroscope measurements (N, 3) in rad/s
            accel: Array of accelerometer measurements (N, 3) in m/s²
        """
        if not os.path.isfile(self.imu_file):
            raise FileNotFoundError(f"IMU file not found: {self.imu_file}")
        
        try:
            # Load data, skip header comments
            data = np.loadtxt(self.imu_file, delimiter=',', comments='#')
            
            timestamps = data[:, 0].astype(np.int64)
            gyro = data[:, 1:4]  # [gx, gy, gz]
            accel = data[:, 4:7]  # [ax, ay, az]
            
            logger.info(f"Loaded {len(timestamps)} IMU samples")
            return timestamps, gyro, accel
            
        except Exception as e:
            raise RuntimeError(f"Failed to load IMU data: {e}")

    def get_camera_intrinsics(self, camera: str = "cam0") -> Optional[np.ndarray]:
        """
        Load camera intrinsic parameters from sensor.yaml.
        
        Args:
            camera: Camera identifier
            
        Returns:
            Camera intrinsic matrix (3x3) or None if not available
        """
        config_file = self.cam_configs.get(camera)
        if not config_file or not os.path.isfile(config_file):
            logger.warning(f"Camera config file not found: {config_file}")
            return None
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract intrinsic parameters
            intrinsics = config.get('intrinsics', [])
            if len(intrinsics) >= 4:
                fx, fy, cx, cy = intrinsics[:4]
                K = np.array([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ])
                return K
            else:
                logger.warning(f"Invalid intrinsics format in {config_file}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load camera intrinsics: {e}")
            return None
    
    def get_imu_noise_params(self) -> Optional[dict]:
        """
        Load IMU noise parameters from sensor.yaml.
        
        Returns:
            Dictionary with noise parameters or None if not available
        """
        if not os.path.isfile(self.imu_config):
            logger.warning(f"IMU config file not found: {self.imu_config}")
            return None
        
        try:
            with open(self.imu_config, 'r') as f:
                config = yaml.safe_load(f)
            
            noise_params = {
                'gyro_noise_density': config.get('gyroscope_noise_density', 0.0),
                'gyro_random_walk': config.get('gyroscope_random_walk', 0.0),
                'accel_noise_density': config.get('accelerometer_noise_density', 0.0),
                'accel_random_walk': config.get('accelerometer_random_walk', 0.0),
            }
            
            return noise_params
            
        except Exception as e:
            logger.warning(f"Failed to load IMU noise parameters: {e}")
            return None
    
    def get_ground_truth(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load ground truth trajectory if available.
        
        Returns:
            Tuple of (timestamps, poses) or None if not available
            poses: Array of shape (N, 7) containing [x, y, z, qw, qx, qy, qz]
        """
        gt_file = os.path.join(self.root_dir, 'state_groundtruth_estimate0', 'data.csv')
        
        if not os.path.isfile(gt_file):
            logger.info("Ground truth file not found")
            return None
        
        try:
            data = np.loadtxt(gt_file, delimiter=',', comments='#')
            timestamps = data[:, 0].astype(np.int64)
            poses = data[:, 1:8]  # [x, y, z, qw, qx, qy, qz]
            
            logger.info(f"Loaded {len(timestamps)} ground truth poses")
            return timestamps, poses
            
        except Exception as e:
            logger.warning(f"Failed to load ground truth: {e}")
            return None
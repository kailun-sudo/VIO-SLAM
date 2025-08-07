"""
Base dataset loader interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np


class DatasetLoader(ABC):
    """
    Abstract base class for SLAM dataset loaders.
    
    This class defines the interface that all dataset loaders must implement
    to ensure compatibility with the SLAM pipeline.
    """
    
    def __init__(self, root_dir: str):
        """
        Initialize dataset loader.
        
        Args:
            root_dir: Path to the dataset root directory
        """
        self.root_dir = root_dir
    
    @abstractmethod
    def load_images(self, camera: str = "cam0") -> Tuple[np.ndarray, List[str]]:
        """
        Load image timestamps and file paths.
        
        Args:
            camera: Camera identifier (e.g., 'cam0', 'cam1')
            
        Returns:
            timestamps: Array of timestamps (int64, nanoseconds)
            image_paths: List of image file paths
        """
        pass
    
    @abstractmethod
    def load_imu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load IMU data.
        
        Returns:
            timestamps: Array of timestamps (int64, nanoseconds)
            gyro: Array of gyroscope measurements (N, 3) in rad/s
            accel: Array of accelerometer measurements (N, 3) in m/s²
        """
        pass
    
    def get_camera_intrinsics(self, camera: str = "cam0") -> Optional[np.ndarray]:
        """
        Get camera intrinsic parameters if available.
        
        Args:
            camera: Camera identifier
            
        Returns:
            Camera intrinsic matrix (3x3) or None if not available
        """
        return None
    
    def get_imu_noise_params(self) -> Optional[dict]:
        """
        Get IMU noise parameters if available.
        
        Returns:
            Dictionary with noise parameters or None if not available
        """
        return None
    
    def validate_dataset(self) -> bool:
        """
        Validate that the dataset is properly formatted and accessible.
        
        Returns:
            True if dataset is valid, False otherwise
        """
        try:
            ts_img, img_paths = self.load_images()
            ts_imu, gyro, accel = self.load_imu()
            
            # Basic validation checks
            if len(ts_img) == 0 or len(img_paths) == 0:
                return False
            if len(ts_imu) == 0 or len(gyro) == 0 or len(accel) == 0:
                return False
            if len(ts_img) != len(img_paths):
                return False
            if not (len(ts_imu) == len(gyro) == len(accel)):
                return False
                
            return True
        except Exception:
            return False
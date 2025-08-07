"""
Tests for dataset loader modules.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from vio_slam.dataset.euroc_loader import EuRoCDatasetLoader
from vio_slam.dataset.base_loader import DatasetLoader


class TestDatasetLoader:
    """Test base dataset loader interface."""
    
    def test_abstract_base_class(self):
        """Test that DatasetLoader is abstract."""
        with pytest.raises(TypeError):
            DatasetLoader("test_path")


class TestEuRoCDatasetLoader:
    """Test EuRoC dataset loader."""
    
    @pytest.fixture
    def temp_euroc_structure(self):
        """Create temporary EuRoC dataset structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            cam0_dir = os.path.join(temp_dir, 'cam0', 'data')
            cam1_dir = os.path.join(temp_dir, 'cam1', 'data')
            imu_dir = os.path.join(temp_dir, 'imu0')
            
            os.makedirs(cam0_dir)
            os.makedirs(cam1_dir)
            os.makedirs(imu_dir)
            
            # Create dummy image files
            for i, timestamp in enumerate([1403636579763555584, 1403636579813555456]):
                open(os.path.join(cam0_dir, f"{timestamp}.png"), 'w').close()
                open(os.path.join(cam1_dir, f"{timestamp}.png"), 'w').close()
            
            # Create dummy IMU file
            imu_data = """#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
1403636579758555392,0.0127,0.0108,-0.0001,8.1776,-1.9999,2.1043
1403636579763555392,0.0125,0.0110,-0.0002,8.1780,-2.0001,2.1040"""
            
            with open(os.path.join(imu_dir, 'data.csv'), 'w') as f:
                f.write(imu_data)
            
            # Create dummy sensor config
            sensor_config = """sensor_type: camera
comment: VI-Sensor cam0 (MT9M034)
T_BS:
  cols: 4
  rows: 4
  data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0]
image_dimension: [752, 480]
camera_model: pinhole
intrinsics: [458.654, 457.296, 367.215, 248.375]
distortion_model: radial-tangential
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]"""
            
            with open(os.path.join(temp_dir, 'cam0', 'sensor.yaml'), 'w') as f:
                f.write(sensor_config)
                
            yield temp_dir
    
    def test_initialization(self, temp_euroc_structure):
        """Test EuRoC loader initialization."""
        loader = EuRoCDatasetLoader(temp_euroc_structure)
        
        assert loader.root_dir == temp_euroc_structure
        assert 'cam0' in loader.cam_dirs
        assert 'cam1' in loader.cam_dirs
        assert loader.imu_file.endswith('data.csv')
    
    def test_load_images(self, temp_euroc_structure):
        """Test image loading."""
        loader = EuRoCDatasetLoader(temp_euroc_structure)
        
        timestamps, image_paths = loader.load_images('cam0')
        
        assert len(timestamps) == 2
        assert len(image_paths) == 2
        assert timestamps.dtype == np.int64
        assert all(path.endswith('.png') for path in image_paths)
        assert np.all(timestamps == np.array([1403636579763555584, 1403636579813555456]))
    
    def test_load_images_invalid_camera(self, temp_euroc_structure):
        """Test loading images with invalid camera."""
        loader = EuRoCDatasetLoader(temp_euroc_structure)
        
        with pytest.raises(ValueError, match="Camera directory not found"):
            loader.load_images('invalid_cam')
    
    def test_load_imu(self, temp_euroc_structure):
        """Test IMU loading."""
        loader = EuRoCDatasetLoader(temp_euroc_structure)
        
        timestamps, gyro, accel = loader.load_imu()
        
        assert len(timestamps) == 2
        assert gyro.shape == (2, 3)
        assert accel.shape == (2, 3)
        assert timestamps.dtype == np.int64
        
        # Check first sample values
        np.testing.assert_almost_equal(gyro[0], [0.0127, 0.0108, -0.0001])
        np.testing.assert_almost_equal(accel[0], [8.1776, -1.9999, 2.1043])
    
    def test_load_imu_missing_file(self):
        """Test IMU loading with missing file."""
        loader = EuRoCDatasetLoader("nonexistent_path")
        
        with pytest.raises(FileNotFoundError, match="IMU file not found"):
            loader.load_imu()
    
    def test_get_camera_intrinsics(self, temp_euroc_structure):
        """Test camera intrinsics loading."""
        loader = EuRoCDatasetLoader(temp_euroc_structure)
        
        K = loader.get_camera_intrinsics('cam0')
        
        assert K is not None
        assert K.shape == (3, 3)
        np.testing.assert_almost_equal(K[0, 0], 458.654)
        np.testing.assert_almost_equal(K[1, 1], 457.296)
        np.testing.assert_almost_equal(K[0, 2], 367.215)
        np.testing.assert_almost_equal(K[1, 2], 248.375)
    
    def test_validate_dataset(self, temp_euroc_structure):
        """Test dataset validation."""
        loader = EuRoCDatasetLoader(temp_euroc_structure)
        
        assert loader.validate_dataset() == True
    
    def test_validate_dataset_invalid(self):
        """Test dataset validation with invalid path."""
        loader = EuRoCDatasetLoader("nonexistent_path")
        
        assert loader.validate_dataset() == False


if __name__ == "__main__":
    pytest.main([__file__])
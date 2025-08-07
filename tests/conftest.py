"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import tempfile
import os
import cv2


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'dataset': {
            'type': 'euroc',
            'camera': 'cam0',
            'downsample_factor': 5,
        },
        'slam': {
            'window_size': 3,
            'orb_features': 500,
            'loop_closure': {
                'enabled': True,
                'vocabulary_size': 100,
                'similarity_threshold': 0.7,
            },
            'optimization': {
                'max_iterations': 50,
                'verbose': False,
            },
        },
        'visualization': {
            'show_features': False,
            'show_trajectory': False,
            'save_plots': False,
        },
    }


@pytest.fixture
def sample_camera_matrix():
    """Sample camera intrinsic matrix."""
    return np.array([
        [458.654, 0.0, 367.215],
        [0.0, 457.296, 248.375],
        [0.0, 0.0, 1.0]
    ])


@pytest.fixture
def sample_imu_data():
    """Generate sample IMU data for testing."""
    n_samples = 100
    dt = 0.01  # 100 Hz
    
    # Generate timestamps
    timestamps = np.arange(n_samples) * dt * 1e9  # Convert to nanoseconds
    timestamps = timestamps.astype(np.int64)
    
    # Generate gyroscope data (rad/s)
    gyro = np.random.normal(0, 0.01, (n_samples, 3))
    gyro[:, 2] += 0.1 * np.sin(np.arange(n_samples) * dt)  # Add some yaw rotation
    
    # Generate accelerometer data (m/s²)
    accel = np.random.normal([0, 0, -9.81], 0.1, (n_samples, 3))
    accel[:, 0] += 1.0 * np.sin(np.arange(n_samples) * dt * 0.5)  # Add forward acceleration
    
    return timestamps, gyro, accel


@pytest.fixture
def sample_image_sequence():
    """Generate sample image sequence for testing."""
    n_images = 10
    img_size = (480, 640)
    
    images = []
    timestamps = []
    
    for i in range(n_images):
        # Create image with moving features
        img = np.zeros(img_size, dtype=np.uint8)
        
        # Add moving rectangle
        x_offset = i * 10
        y_offset = i * 5
        cv2.rectangle(img, (100 + x_offset, 100 + y_offset), 
                     (200 + x_offset, 200 + y_offset), 255, -1)
        
        # Add some circles
        cv2.circle(img, (300 + i*2, 300 - i*3), 30, 128, -1)
        cv2.circle(img, (450 - i*5, 200 + i*2), 40, 200, 2)
        
        # Add noise
        noise = np.random.normal(0, 10, img_size).astype(np.uint8)
        img = cv2.add(img, noise)
        
        images.append(img)
        timestamps.append(i * 100000000)  # 100ms intervals in nanoseconds
    
    return np.array(timestamps, dtype=np.int64), images


@pytest.fixture
def temp_dataset_dir():
    """Create temporary dataset directory structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create EuRoC-like structure
        mav0_dir = os.path.join(temp_dir, 'mav0')
        cam0_dir = os.path.join(mav0_dir, 'cam0', 'data')
        imu_dir = os.path.join(mav0_dir, 'imu0')
        
        os.makedirs(cam0_dir)
        os.makedirs(imu_dir)
        
        # Create sample images
        for i in range(5):
            img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
            timestamp = 1403636579763555584 + i * 50000000  # 50ms intervals
            img_path = os.path.join(cam0_dir, f"{timestamp}.png")
            cv2.imwrite(img_path, img)
        
        # Create sample IMU data
        imu_data = """#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
1403636579763555584,0.0127,0.0108,-0.0001,8.1776,-1.9999,2.1043
1403636579783555584,0.0125,0.0110,-0.0002,8.1780,-2.0001,2.1040
1403636579803555584,0.0123,0.0112,-0.0003,8.1784,-2.0003,2.1037
1403636579823555584,0.0121,0.0114,-0.0004,8.1788,-2.0005,2.1034
1403636579843555584,0.0119,0.0116,-0.0005,8.1792,-2.0007,2.1031"""
        
        with open(os.path.join(imu_dir, 'data.csv'), 'w') as f:
            f.write(imu_data)
        
        # Create camera config
        cam_config = """sensor_type: camera
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
        
        with open(os.path.join(mav0_dir, 'cam0', 'sensor.yaml'), 'w') as f:
            f.write(cam_config)
        
        yield mav0_dir


@pytest.fixture(scope="session")
def opencv_available():
    """Check if OpenCV is available and working."""
    try:
        import cv2
        # Test basic functionality
        img = np.zeros((100, 100), dtype=np.uint8)
        orb = cv2.ORB_create(nfeatures=10)
        kp, des = orb.detectAndCompute(img, None)
        return True
    except:
        return False


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid or "test_slam_pipeline" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if any(slow_keyword in item.nodeid for slow_keyword in ["optimization", "pose_graph", "vocabulary"]):
            item.add_marker(pytest.mark.slow)
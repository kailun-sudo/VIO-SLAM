"""
Main SLAM pipeline orchestrating all components.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import yaml

from .dataset.euroc_loader import EuRoCDatasetLoader
from .features.orb_tracker import ORBTracker
from .optimization.imu_preintegrator import IMUPreintegrator
from .loop_closure.detector import LoopClosureDetector
from .optimization.pose_graph import PoseGraphOptimizer
from .visualization.plotter import TrajectoryPlotter
from .utils.config import load_config, validate_config

logger = logging.getLogger(__name__)


class SLAMPipeline:
    """
    Main SLAM pipeline class that orchestrates all components.
    
    This class provides a high-level interface for running VIO-SLAM
    with loop closure detection and pose graph optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 config_path: Optional[str] = None,
                 camera_matrix: Optional[np.ndarray] = None):
        """
        Initialize SLAM pipeline.
        
        Args:
            config: Configuration dictionary
            config_path: Path to configuration YAML file
            camera_matrix: Camera intrinsic matrix (3x3)
        """
        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = self._get_default_config()
            
        validate_config(self.config)
        
        # Set camera matrix
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        else:
            # Default EuRoC cam0 intrinsics
            self.camera_matrix = np.array([
                [458.654, 0.0, 367.215],
                [0.0, 457.296, 248.375], 
                [0.0, 0.0, 1.0]
            ])
        
        # Initialize components
        self._initialize_components()
        
        # State variables
        self.dataset_loader = None
        self.trajectory = []
        self.keyframe_poses = []
        self.processing_times = []
        
        logger.info("SLAM pipeline initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
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
    
    def _initialize_components(self):
        """Initialize SLAM components based on configuration."""
        slam_config = self.config['slam']
        
        # Feature tracker
        self.orb_tracker = ORBTracker(
            n_features=slam_config['orb_features']
        )
        
        # IMU preintegrator
        self.imu_preintegrator = IMUPreintegrator()
        
        # Loop closure detector
        if slam_config['loop_closure']['enabled']:
            self.loop_detector = LoopClosureDetector(
                vocabulary_size=slam_config['loop_closure']['vocabulary_size']
            )
        else:
            self.loop_detector = None
            
        # Pose graph optimizer
        self.pose_optimizer = PoseGraphOptimizer()
        
        # Visualization
        if self.config['visualization']['show_trajectory']:
            self.plotter = TrajectoryPlotter()
        else:
            self.plotter = None
    
    def load_dataset(self, data_path: str, dataset_type: str = "euroc"):
        """
        Load dataset for processing.
        
        Args:
            data_path: Path to dataset directory
            dataset_type: Type of dataset ('euroc', 'tum', 'kitti')
        """
        if dataset_type.lower() == "euroc":
            self.dataset_loader = EuRoCDatasetLoader(data_path)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        # Validate dataset
        if not self.dataset_loader.validate_dataset():
            raise RuntimeError("Dataset validation failed")
        
        logger.info(f"Loaded {dataset_type} dataset from {data_path}")
    
    def run(self) -> np.ndarray:
        """
        Run the complete SLAM pipeline.
        
        Returns:
            Final trajectory as array of shape (N, 3)
        """
        if self.dataset_loader is None:
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")
        
        # Load data
        ts_img, img_paths = self.dataset_loader.load_images(
            self.config['dataset']['camera']
        )
        ts_imu, gyro, accel = self.dataset_loader.load_imu()
        
        logger.info(f"Processing {len(ts_img)} images and {len(ts_imu)} IMU samples")
        
        # Update camera matrix from dataset if available
        dataset_K = self.dataset_loader.get_camera_intrinsics()
        if dataset_K is not None:
            self.camera_matrix = dataset_K
        
        # Process with sliding window
        trajectory = self.process(ts_img, img_paths, ts_imu, gyro, accel)
        
        return trajectory
    
    def process(self, ts_img: np.ndarray, img_paths: list, 
                ts_imu: np.ndarray, gyro: np.ndarray, 
                accel: np.ndarray) -> np.ndarray:
        """
        Process loaded data through SLAM pipeline.
        
        Args:
            ts_img: Image timestamps
            img_paths: Image file paths
            ts_imu: IMU timestamps
            gyro: Gyroscope measurements
            accel: Accelerometer measurements
            
        Returns:
            Trajectory array of shape (N, 3)
        """
        slam_config = self.config['slam']
        dataset_config = self.config['dataset']
        
        # Setup processing parameters
        downsample = dataset_config['downsample_factor']
        window_size = slam_config['window_size']
        num_windows = len(ts_img) - window_size
        starts = list(range(0, num_windows, downsample))
        
        logger.info(f"Processing {len(starts)} windows with size {window_size}")
        
        # Build vocabulary for loop detection if enabled
        if self.loop_detector:
            self._build_loop_closure_vocabulary(img_paths[:100:5])
        
        # Process sliding windows
        trajectory_increments = []
        start_time = time.time()
        
        for i, s in enumerate(tqdm(starts, desc="VIO Processing", ncols=80)):
            window_start_time = time.time()
            
            # Extract window data
            t_win = ts_img[s:s+window_size+1]
            img_win = img_paths[s:s+window_size+1]
            
            # Process window
            _, p_opt = self._process_window(
                t_win, ts_imu, gyro, accel, img_win, window_size
            )
            
            # Store result
            trajectory_increments.append(p_opt[-1])
            
            # Add keyframe for loop detection
            if self.loop_detector and s < len(img_paths):
                self._add_keyframe_for_loop_detection(img_paths[s], p_opt[-1])
            
            # Record processing time
            window_time = time.time() - window_start_time
            self.processing_times.append(window_time)
            
            # Progress logging
            if (i + 1) % 10 == 0:
                avg_time = np.mean(self.processing_times[-10:]) * 1000
                logger.info(f"Processed {i+1}/{len(starts)} windows, "
                           f"avg time: {avg_time:.1f}ms")
        
        total_time = time.time() - start_time
        logger.info(f"Processing completed in {total_time:.2f}s")
        logger.info(f"Average time per window: {total_time/len(starts)*1000:.1f}ms")
        
        # Global trajectory reconstruction
        trajectory = self._reconstruct_global_trajectory(trajectory_increments)
        
        # Perform pose graph optimization if loop closures were detected
        if self.pose_optimizer and len(self.pose_optimizer.loop_edges) > 0:
            trajectory = self._optimize_pose_graph(trajectory)
        
        # Store final trajectory
        self.trajectory = trajectory
        
        return trajectory
    
    def _build_loop_closure_vocabulary(self, img_paths: list):
        """Build vocabulary for loop closure detection."""
        logger.info("Building vocabulary for loop detection...")
        
        vocab_descriptors = []
        for img_path in img_paths:
            import cv2
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _, desc = self.orb_tracker.detect_and_compute(img)
                if desc is not None:
                    vocab_descriptors.append(desc)
        
        if vocab_descriptors:
            self.loop_detector.build_vocabulary(vocab_descriptors)
            logger.info(f"Built vocabulary from {len(vocab_descriptors)} images")
    
    def _process_window(self, ts_img: np.ndarray, ts_imu: np.ndarray,
                       gyro: np.ndarray, accel: np.ndarray, 
                       img_paths: list, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single sliding window."""
        # IMU preintegration
        dps, dRs = self._preintegrate_imu(ts_img, ts_imu, gyro, accel, K)
        
        # Visual measurements
        t_dirs = self._compute_visual_directions(img_paths, K)
        
        # Sliding window optimization
        p_init, p_opt = self._optimize_window(dps, dRs, t_dirs, K)
        
        return p_init, p_opt
    
    def _preintegrate_imu(self, ts_img: np.ndarray, ts_imu: np.ndarray,
                         gyro: np.ndarray, accel: np.ndarray, 
                         K: int) -> Tuple[list, list]:
        """Preintegrate IMU measurements between image frames."""
        dps, dRs = [], []
        
        for i in range(1, K+1):
            t0, t1 = ts_img[i-1], ts_img[i]
            mask = (ts_imu >= t0) & (ts_imu < t1)
            
            if not np.any(mask):
                # No IMU data in this interval
                dps.append(np.zeros(3))
                dRs.append(np.eye(3))
                continue
            
            ts_sec = ts_imu[mask] * 1e-9
            dt_arr = np.diff(ts_sec)
            omega_arr = gyro[mask][:-1]
            acc_arr = accel[mask][:-1]
            
            if len(dt_arr) > 0:
                dp, _, dR = self.imu_preintegrator.integrate(dt_arr, omega_arr, acc_arr)
                dps.append(dp)
                dRs.append(dR)
            else:
                dps.append(np.zeros(3))
                dRs.append(np.eye(3))
        
        return dps, dRs
    
    def _compute_visual_directions(self, img_paths: list, K: int) -> list:
        """Compute visual motion directions between consecutive frames."""
        import cv2
        from scipy.optimize import least_squares
        
        t_dirs = []
        
        for i in range(K):
            img1 = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img_paths[i+1], cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                t_dirs.append(np.array([1.0, 0.0, 0.0]))
                continue
            
            # Track features
            _, _, pts1, pts2, _ = self.orb_tracker.track(img1, img2)
            
            if len(pts1) > 8:
                try:
                    # Estimate essential matrix
                    E, _ = cv2.findEssentialMat(
                        pts1, pts2, self.camera_matrix,
                        method=cv2.RANSAC, prob=0.999, threshold=1.0
                    )
                    
                    # Recover pose
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
                    t_dir = t.flatten() / (np.linalg.norm(t.flatten()) + 1e-8)
                    t_dirs.append(t_dir)
                    
                except:
                    # Fallback direction
                    t_dirs.append(np.array([1.0, 0.0, 0.0]))
            else:
                # Fallback direction
                t_dirs.append(np.array([1.0, 0.0, 0.0]))
        
        return t_dirs
    
    def _optimize_window(self, dps: list, dRs: list, t_dirs: list, 
                        K: int) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize poses in sliding window."""
        from scipy.optimize import least_squares
        
        # Initial guess
        p_init = np.zeros((K+1, 3))
        theta = np.zeros(K+1)
        
        for i in range(1, K+1):
            if i-1 < len(dps):
                p_init[i] = p_init[i-1] + dps[i-1]
                
            if i-1 < len(dRs):
                yaw = np.arctan2(dRs[i-1][1,0], dRs[i-1][0,0])
                theta[i] = theta[i-1] + yaw
        
        x0 = np.hstack([p_init.flatten(), theta])
        
        # Define residuals
        def residual(x):
            p = x[:3*(K+1)].reshape((K+1, 3))
            th = x[3*(K+1):]
            res = []
            
            for i in range(1, K+1):
                # IMU position constraint
                if i-1 < len(dps):
                    res.extend((p[i] - p[i-1] - dps[i-1]).tolist())
                
                # IMU yaw constraint
                if i-1 < len(dRs):
                    dy = th[i] - th[i-1]
                    yaw_measured = np.arctan2(dRs[i-1][1,0], dRs[i-1][0,0])
                    res.append(dy - yaw_measured)
                
                # Visual direction constraint
                if i-1 < len(t_dirs):
                    dp_est = p[i] - p[i-1]
                    dir_est = dp_est / (np.linalg.norm(dp_est) + 1e-8)
                    res.extend((dir_est - t_dirs[i-1]).tolist())
            
            return np.array(res)
        
        # Solve optimization
        try:
            sol = least_squares(residual, x0, verbose=0)
            p_opt = sol.x[:3*(K+1)].reshape((K+1, 3))
        except:
            logger.warning("Window optimization failed, using initial guess")
            p_opt = p_init
        
        return p_init, p_opt
    
    def _add_keyframe_for_loop_detection(self, img_path: str, pose: np.ndarray):
        """Add keyframe to loop closure detector."""
        import cv2
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, desc = self.orb_tracker.detect_and_compute(img)
            
            pose_4x4 = np.eye(4)
            pose_4x4[:3, 3] = pose
            
            self.loop_detector.add_keyframe(desc, pose_4x4)
    
    def _reconstruct_global_trajectory(self, trajectory_increments: list) -> np.ndarray:
        """Reconstruct global trajectory from incremental poses."""
        if not trajectory_increments:
            return np.array([]).reshape(0, 3)
        
        trajectory_increments = np.array(trajectory_increments)
        trajectory = np.vstack(([0, 0, 0], np.cumsum(trajectory_increments, axis=0)))
        
        logger.info(f"Reconstructed trajectory with {len(trajectory)} poses")
        return trajectory
    
    def _optimize_pose_graph(self, trajectory: np.ndarray) -> np.ndarray:
        """Perform pose graph optimization."""
        logger.info("Performing pose graph optimization...")
        
        # Add poses to optimizer
        for i, pose_pos in enumerate(trajectory):
            pose_4x4 = np.eye(4)
            pose_4x4[:3, 3] = pose_pos
            
            if i >= len(self.pose_optimizer.poses):
                self.pose_optimizer.add_pose(pose_4x4)
            else:
                self.pose_optimizer.poses[i] = pose_4x4
        
        # Optimize
        self.pose_optimizer.optimize(max_iterations=100)
        
        # Extract optimized trajectory
        optimized_trajectory = np.array([
            pose[:3, 3] for pose in self.pose_optimizer.poses
        ])
        
        logger.info(f"Pose graph optimization completed with "
                   f"{len(self.pose_optimizer.loop_edges)} loop closures")
        
        return optimized_trajectory
    
    def visualize_trajectory(self, save_path: Optional[str] = None):
        """Visualize the computed trajectory."""
        if len(self.trajectory) == 0:
            logger.warning("No trajectory to visualize")
            return
        
        if self.plotter:
            self.plotter.plot_trajectory(self.trajectory, save_path=save_path)
    
    def save_results(self, output_path: str):
        """Save SLAM results."""
        import pickle
        
        results = {
            'trajectory': self.trajectory,
            'keyframe_poses': self.keyframe_poses,
            'processing_times': self.processing_times,
            'config': self.config,
            'loop_closures': len(self.pose_optimizer.loop_edges) if self.pose_optimizer else 0,
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not self.processing_times:
            return {}
        
        return {
            'total_frames': len(self.trajectory),
            'avg_processing_time_ms': np.mean(self.processing_times) * 1000,
            'total_processing_time_s': np.sum(self.processing_times),
            'loop_closures_detected': len(self.pose_optimizer.loop_edges) if self.pose_optimizer else 0,
            'trajectory_length_m': np.sum(np.linalg.norm(np.diff(self.trajectory, axis=0), axis=1)) if len(self.trajectory) > 1 else 0,
        }
"""
Sliding window optimization for visual-inertial odometry.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SlidingWindowOptimizer:
    """
    Sliding window optimizer for visual-inertial odometry.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize sliding window optimizer.
        
        Args:
            window_size: Number of poses in the sliding window
        """
        self.window_size = window_size
        self.poses = []
        self.imu_constraints = []
        self.visual_constraints = []
        
        logger.debug(f"Initialized sliding window optimizer with window size {window_size}")
    
    def add_imu_constraint(self, from_idx: int, to_idx: int,
                          delta_p: np.ndarray, delta_R: np.ndarray,
                          information: np.ndarray):
        """
        Add IMU preintegration constraint.
        
        Args:
            from_idx: Source pose index
            to_idx: Target pose index  
            delta_p: Preintegrated position
            delta_R: Preintegrated rotation
            information: Information matrix
        """
        self.imu_constraints.append({
            'from': from_idx,
            'to': to_idx,
            'delta_p': delta_p.copy(),
            'delta_R': delta_R.copy(),
            'information': information.copy()
        })
    
    def add_visual_constraint(self, from_idx: int, to_idx: int,
                             direction: np.ndarray, weight: float = 1.0):
        """
        Add visual direction constraint.
        
        Args:
            from_idx: Source pose index
            to_idx: Target pose index
            direction: Unit direction vector
            weight: Constraint weight
        """
        self.visual_constraints.append({
            'from': from_idx,
            'to': to_idx,
            'direction': direction / (np.linalg.norm(direction) + 1e-8),
            'weight': weight
        })
    
    def optimize(self, initial_poses: np.ndarray,
                max_iterations: int = 100) -> Tuple[np.ndarray, bool]:
        """
        Optimize poses in sliding window.
        
        Args:
            initial_poses: Initial pose estimates (N, 3)
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized poses and success flag
        """
        if len(initial_poses) == 0:
            return initial_poses, False
        
        # Convert poses to parameter vector
        theta_init = np.zeros(len(initial_poses))
        x0 = np.hstack([initial_poses.flatten(), theta_init])
        
        # Define residual function
        def residual_function(x):
            return self._compute_residuals(x)
        
        try:
            # Optimize using least squares
            result = least_squares(
                residual_function, x0,
                max_nfev=max_iterations * len(x0),
                verbose=0
            )
            
            # Extract optimized poses
            n_poses = len(initial_poses)
            p_opt = result.x[:3*n_poses].reshape((n_poses, 3))
            
            logger.debug(f"Sliding window optimization completed. Cost: {result.cost:.6f}")
            return p_opt, result.success
            
        except Exception as e:
            logger.error(f"Sliding window optimization failed: {e}")
            return initial_poses, False
    
    def _compute_residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute residuals for all constraints."""
        n_poses = (len(x)) // 4  # positions + angles
        p = x[:3*n_poses].reshape((n_poses, 3))
        theta = x[3*n_poses:]
        
        residuals = []
        
        # IMU constraints
        for constraint in self.imu_constraints:
            i, j = constraint['from'], constraint['to']
            if i >= n_poses or j >= n_poses:
                continue
                
            # Position residual
            dp_expected = constraint['delta_p']
            dp_actual = p[j] - p[i]
            pos_residual = dp_actual - dp_expected
            residuals.extend(pos_residual.tolist())
            
            # Rotation residual (yaw only for simplicity)
            dR = constraint['delta_R']
            dyaw_expected = np.arctan2(dR[1, 0], dR[0, 0])
            dyaw_actual = theta[j] - theta[i]
            dyaw_actual = self._normalize_angle(dyaw_actual)
            
            rot_residual = dyaw_actual - dyaw_expected
            residuals.append(rot_residual)
        
        # Visual constraints
        for constraint in self.visual_constraints:
            i, j = constraint['from'], constraint['to']
            if i >= n_poses or j >= n_poses:
                continue
                
            direction_expected = constraint['direction']
            dp_actual = p[j] - p[i]
            direction_actual = dp_actual / (np.linalg.norm(dp_actual) + 1e-8)
            
            direction_residual = direction_actual - direction_expected
            weight = constraint['weight']
            residuals.extend((weight * direction_residual).tolist())
        
        return np.array(residuals)
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]."""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def clear_constraints(self):
        """Clear all constraints."""
        self.imu_constraints.clear()
        self.visual_constraints.clear()
        logger.debug("Cleared all sliding window constraints")
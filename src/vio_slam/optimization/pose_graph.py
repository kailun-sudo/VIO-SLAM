"""
Pose graph optimization for loop closure.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PoseGraphOptimizer:
    """
    Pose graph optimization for loop closure correction.
    """
    
    def __init__(self):
        """Initialize pose graph optimizer."""
        self.poses = []  # List of 4x4 pose matrices
        self.odometry_edges = []  # Sequential odometry constraints
        self.loop_edges = []  # Loop closure constraints
        
        logger.debug("Initialized pose graph optimizer")
    
    def add_pose(self, pose: np.ndarray):
        """
        Add new pose to graph.
        
        Args:
            pose: 4x4 pose matrix
        """
        self.poses.append(pose.copy())
    
    def add_odometry_edge(self, from_idx: int, to_idx: int, 
                         relative_pose: np.ndarray, information: np.ndarray):
        """
        Add odometry constraint between consecutive poses.
        
        Args:
            from_idx: Source pose index
            to_idx: Target pose index
            relative_pose: 4x4 relative transformation
            information: 6x6 information matrix
        """
        self.odometry_edges.append({
            'from': from_idx,
            'to': to_idx,
            'relative_pose': relative_pose.copy(),
            'information': information.copy()
        })
    
    def add_loop_edge(self, from_idx: int, to_idx: int,
                      relative_pose: np.ndarray, information: np.ndarray):
        """
        Add loop closure constraint.
        
        Args:
            from_idx: Source pose index
            to_idx: Target pose index
            relative_pose: 4x4 relative transformation
            information: 6x6 information matrix
        """
        self.loop_edges.append({
            'from': from_idx,
            'to': to_idx,
            'relative_pose': relative_pose.copy(),
            'information': information.copy()
        })
        
        logger.info(f"Added loop closure edge: {from_idx} -> {to_idx}")
    
    def optimize(self, max_iterations: int = 50) -> bool:
        """
        Optimize pose graph using Gauss-Newton method.
        
        Args:
            max_iterations: Maximum optimization iterations
            
        Returns:
            True if optimization succeeded
        """
        if len(self.poses) < 2:
            logger.warning("Not enough poses for optimization")
            return False
        
        logger.info(f"Starting pose graph optimization with {len(self.poses)} poses, "
                   f"{len(self.odometry_edges)} odometry edges, {len(self.loop_edges)} loop edges")
        
        # Convert poses to parameter vector (x, y, theta for 2D)
        params = self._poses_to_params()
        
        # Define residual function
        def residual_function(p):
            return self._compute_residuals(p)
        
        try:
            # Optimize using least squares
            result = least_squares(
                residual_function, params, 
                max_nfev=max_iterations * len(params),
                verbose=0
            )
            
            # Update poses with optimized parameters
            self._params_to_poses(result.x)
            
            logger.info(f"Pose graph optimization completed. Cost: {result.cost:.6f}")
            return result.success
            
        except Exception as e:
            logger.error(f"Pose graph optimization failed: {e}")
            return False
    
    def _poses_to_params(self) -> np.ndarray:
        """Convert poses to optimization parameters."""
        params = []
        for pose in self.poses:
            x, y = pose[0, 3], pose[1, 3]
            theta = np.arctan2(pose[1, 0], pose[0, 0])
            params.extend([x, y, theta])
        return np.array(params)
    
    def _params_to_poses(self, params: np.ndarray):
        """Convert optimization parameters back to poses."""
        params_2d = params.reshape(-1, 3)
        
        for i, (x, y, theta) in enumerate(params_2d):
            self.poses[i] = np.array([
                [np.cos(theta), -np.sin(theta), 0, x],
                [np.sin(theta), np.cos(theta), 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
    
    def _compute_residuals(self, params: np.ndarray) -> np.ndarray:
        """Compute residuals for all edges."""
        poses_2d = params.reshape(-1, 3)  # (N, 3) [x, y, theta]
        residuals = []
        
        # Odometry residuals
        for edge in self.odometry_edges:
            i, j = edge['from'], edge['to']
            if i >= len(poses_2d) or j >= len(poses_2d):
                continue
                
            rel_pose = edge['relative_pose']
            
            # Expected relative transformation
            dx_exp = rel_pose[0, 3]
            dy_exp = rel_pose[1, 3]
            dtheta_exp = np.arctan2(rel_pose[1, 0], rel_pose[0, 0])
            
            # Actual relative transformation
            dx_act = poses_2d[j, 0] - poses_2d[i, 0]
            dy_act = poses_2d[j, 1] - poses_2d[i, 1]
            dtheta_act = poses_2d[j, 2] - poses_2d[i, 2]
            
            # Normalize angle
            dtheta_act = self._normalize_angle(dtheta_act)
            dtheta_exp = self._normalize_angle(dtheta_exp)
            
            # Add residuals
            residuals.extend([
                dx_act - dx_exp,
                dy_act - dy_exp,
                dtheta_act - dtheta_exp
            ])
        
        # Loop closure residuals (with higher weight)
        loop_weight = 10.0
        for edge in self.loop_edges:
            i, j = edge['from'], edge['to']
            if i >= len(poses_2d) or j >= len(poses_2d):
                continue
                
            rel_pose = edge['relative_pose']
            
            # Expected relative transformation
            dx_exp = rel_pose[0, 3]
            dy_exp = rel_pose[1, 3]
            dtheta_exp = np.arctan2(rel_pose[1, 0], rel_pose[0, 0])
            
            # Actual relative transformation
            dx_act = poses_2d[j, 0] - poses_2d[i, 0]
            dy_act = poses_2d[j, 1] - poses_2d[i, 1]
            dtheta_act = poses_2d[j, 2] - poses_2d[i, 2]
            
            # Normalize angles
            dtheta_act = self._normalize_angle(dtheta_act)
            dtheta_exp = self._normalize_angle(dtheta_exp)
            
            # Add weighted residuals
            residuals.extend([
                loop_weight * (dx_act - dx_exp),
                loop_weight * (dy_act - dy_exp),
                loop_weight * (dtheta_act - dtheta_exp)
            ])
        
        return np.array(residuals)
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]."""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def get_trajectory(self) -> np.ndarray:
        """
        Get trajectory as array of positions.
        
        Returns:
            Array of shape (N, 3) with [x, y, z] positions
        """
        if not self.poses:
            return np.array([]).reshape(0, 3)
        
        trajectory = np.array([pose[:3, 3] for pose in self.poses])
        return trajectory
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'num_poses': len(self.poses),
            'num_odometry_edges': len(self.odometry_edges),
            'num_loop_edges': len(self.loop_edges),
            'total_edges': len(self.odometry_edges) + len(self.loop_edges)
        }
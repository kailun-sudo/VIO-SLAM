"""
IMU preintegration with bias estimation and flexible transform.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class IMUPreintegrator:
    """
    IMU preintegration with bias estimation and noise modeling.
    
    This class implements IMU preintegration between keyframes, which is essential
    for efficient visual-inertial optimization by avoiding the need to include
    high-frequency IMU measurements directly in the optimization.
    """
    
    def __init__(self, gravity: np.ndarray = np.array([0, 0, -9.81])):
        """
        Initialize IMU preintegrator.
        
        Args:
            gravity: Gravity vector in world frame [m/s²]
        """
        self.gravity = gravity
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.reset()
        
        logger.debug("Initialized IMU preintegrator")
        
    def reset(self):
        """Reset preintegration states."""
        self.delta_t = 0.0
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        
        # Covariance and Jacobians for uncertainty propagation
        self.P_delta = np.zeros((9, 9))  # Covariance of [δp, δv, δθ]
        self.J_bias_gyro = np.zeros((9, 3))  # Jacobian w.r.t. gyro bias
        self.J_bias_acc = np.zeros((9, 3))   # Jacobian w.r.t. accel bias
        
    def update_bias(self, gyro_residual: np.ndarray, accel_residual: np.ndarray,
                   learning_rate: float = 0.001):
        """
        Update IMU bias estimates based on residuals.
        
        Args:
            gyro_residual: Gyroscope residual [rad/s]
            accel_residual: Accelerometer residual [m/s²]
            learning_rate: Bias learning rate
        """
        self.gyro_bias += learning_rate * gyro_residual
        self.accel_bias += learning_rate * accel_residual
        
        logger.debug(f"Updated biases: gyro={self.gyro_bias}, accel={self.accel_bias}")
    
    def integrate(self, dt_arr: np.ndarray, omega_arr: np.ndarray, 
                 acc_arr: np.ndarray, noise_gyro: float = 1e-4,
                 noise_acc: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preintegrate IMU measurements with bias correction and uncertainty propagation.
        
        Args:
            dt_arr: Array of time intervals [s]
            omega_arr: Array of angular velocities [rad/s]
            acc_arr: Array of accelerations [m/s²]
            noise_gyro: Gyroscope noise density
            noise_acc: Accelerometer noise density
            
        Returns:
            delta_p: Preintegrated position
            delta_v: Preintegrated velocity  
            delta_R: Preintegrated rotation matrix
        """
        self.reset()
        
        if len(dt_arr) == 0:
            return self.delta_p.copy(), self.delta_v.copy(), self.delta_R.copy()
        
        # Noise covariance matrices
        Q_gyro = np.eye(3) * noise_gyro**2
        Q_acc = np.eye(3) * noise_acc**2
        
        for i, (dt, w, a) in enumerate(zip(dt_arr, omega_arr, acc_arr)):
            if dt <= 0:
                continue
                
            # Apply bias correction
            w_corrected = w - self.gyro_bias
            a_corrected = a - self.accel_bias
            
            # Store previous states for Jacobian computation
            R_prev = self.delta_R.copy()
            v_prev = self.delta_v.copy()
            
            # Update rotation using midpoint method
            theta = w_corrected * dt
            theta_norm = np.linalg.norm(theta)
            
            if theta_norm > 1e-8:
                dR = Rotation.from_rotvec(theta).as_matrix()
            else:
                # Small angle approximation
                dR = np.eye(3) + self._skew_symmetric(theta)
            
            self.delta_R = self.delta_R @ dR
            
            # Update velocity and position
            a_world = R_prev @ a_corrected + self.gravity
            self.delta_v += a_world * dt
            self.delta_p += v_prev * dt + 0.5 * a_world * dt**2
            
            # Update covariance and Jacobians
            self._update_covariance_and_jacobians(
                dt, w_corrected, a_corrected, R_prev, Q_gyro, Q_acc
            )
            
            self.delta_t += dt
            
        logger.debug(f"Preintegrated over {len(dt_arr)} IMU samples, dt={self.delta_t:.3f}s")
        return self.delta_p.copy(), self.delta_v.copy(), self.delta_R.copy()
    
    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """
        Convert vector to skew-symmetric matrix.
        
        Args:
            v: 3D vector
            
        Returns:
            3x3 skew-symmetric matrix
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]], 
            [-v[1], v[0], 0]
        ])
    
    def _update_covariance_and_jacobians(self, dt: float, w: np.ndarray, a: np.ndarray,
                                        R_prev: np.ndarray, Q_gyro: np.ndarray, 
                                        Q_acc: np.ndarray):
        """
        Update covariance matrix and bias Jacobians.
        
        This method propagates uncertainty through the preintegration process
        to maintain proper uncertainty estimates for optimization.
        """
        # State transition matrix F (9x9)
        F = np.eye(9)
        
        # Position w.r.t. velocity
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Position w.r.t. rotation  
        F[0:3, 6:9] = -0.5 * R_prev @ self._skew_symmetric(a) * dt**2
        
        # Velocity w.r.t. rotation
        F[3:6, 6:9] = -R_prev @ self._skew_symmetric(a) * dt
        
        # Noise matrix G (9x6)
        G = np.zeros((9, 6))
        G[3:6, 0:3] = -R_prev * dt  # Velocity w.r.t. accel noise
        G[6:9, 3:6] = -np.eye(3) * dt  # Rotation w.r.t. gyro noise
        G[0:3, 0:3] = -0.5 * R_prev * dt**2  # Position w.r.t. accel noise
        
        # Process noise covariance
        Q = np.block([
            [Q_acc, np.zeros((3, 3))],
            [np.zeros((3, 3)), Q_gyro]
        ])
        
        # Propagate covariance
        self.P_delta = F @ self.P_delta @ F.T + G @ Q @ G.T
        
        # Update bias Jacobians
        self.J_bias_acc = F @ self.J_bias_acc
        self.J_bias_acc[0:3, :] += -0.5 * R_prev * dt**2
        self.J_bias_acc[3:6, :] += -R_prev * dt
        
        self.J_bias_gyro = F @ self.J_bias_gyro  
        self.J_bias_gyro[6:9, :] += -np.eye(3) * dt
    
    def get_measurement_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get preintegrated measurements for optimization.
        
        Returns:
            z: Measurement vector [δp, δv, δθ] (9,)
            P: Measurement covariance (9x9)
            H_bias: Jacobian w.r.t. biases (9x6)
        """
        # Convert rotation to axis-angle representation
        delta_theta = Rotation.from_matrix(self.delta_R).as_rotvec()
        
        # Measurement vector
        z = np.concatenate([self.delta_p, self.delta_v, delta_theta])
        
        # Bias Jacobian [accel_bias, gyro_bias]
        H_bias = np.hstack([self.J_bias_acc, self.J_bias_gyro])
        
        return z, self.P_delta, H_bias
    
    def correct_preintegration(self, delta_bias_gyro: np.ndarray, 
                              delta_bias_acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Correct preintegrated measurements for bias changes.
        
        Args:
            delta_bias_gyro: Change in gyro bias
            delta_bias_acc: Change in accelerometer bias
            
        Returns:
            Corrected preintegrated measurements
        """
        delta_bias = np.concatenate([delta_bias_acc, delta_bias_gyro])
        H_bias = np.hstack([self.J_bias_acc, self.J_bias_gyro])
        
        correction = H_bias @ delta_bias
        
        # Apply corrections
        corrected_p = self.delta_p - correction[0:3]
        corrected_v = self.delta_v - correction[3:6]
        
        # Correct rotation
        delta_theta_correction = correction[6:9]
        if np.linalg.norm(delta_theta_correction) > 1e-8:
            dR_correction = Rotation.from_rotvec(-delta_theta_correction).as_matrix()
            corrected_R = self.delta_R @ dR_correction
        else:
            corrected_R = self.delta_R
            
        return corrected_p, corrected_v, corrected_R
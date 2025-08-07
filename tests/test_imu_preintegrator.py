"""
Tests for IMU preintegrator module.
"""

import pytest
import numpy as np

from vio_slam.optimization.imu_preintegrator import IMUPreintegrator


class TestIMUPreintegrator:
    """Test IMU preintegration functionality."""
    
    @pytest.fixture
    def preintegrator(self):
        """Create IMU preintegrator instance."""
        return IMUPreintegrator()
    
    def test_initialization(self, preintegrator):
        """Test preintegrator initialization."""
        assert preintegrator.delta_t == 0.0
        assert np.allclose(preintegrator.delta_R, np.eye(3))
        assert np.allclose(preintegrator.delta_v, np.zeros(3))
        assert np.allclose(preintegrator.delta_p, np.zeros(3))
        assert np.allclose(preintegrator.gyro_bias, np.zeros(3))
        assert np.allclose(preintegrator.accel_bias, np.zeros(3))
    
    def test_reset(self, preintegrator):
        """Test reset functionality."""
        # Modify state
        preintegrator.delta_t = 1.0
        preintegrator.delta_R = np.random.rand(3, 3)
        preintegrator.delta_v = np.random.rand(3)
        preintegrator.delta_p = np.random.rand(3)
        
        # Reset
        preintegrator.reset()
        
        # Check reset state
        assert preintegrator.delta_t == 0.0
        assert np.allclose(preintegrator.delta_R, np.eye(3))
        assert np.allclose(preintegrator.delta_v, np.zeros(3))
        assert np.allclose(preintegrator.delta_p, np.zeros(3))
    
    def test_bias_update(self, preintegrator):
        """Test bias update functionality."""
        gyro_residual = np.array([0.01, -0.02, 0.005])
        accel_residual = np.array([0.1, 0.05, -0.08])
        learning_rate = 0.01
        
        preintegrator.update_bias(gyro_residual, accel_residual, learning_rate)
        
        expected_gyro_bias = learning_rate * gyro_residual
        expected_accel_bias = learning_rate * accel_residual
        
        np.testing.assert_allclose(preintegrator.gyro_bias, expected_gyro_bias)
        np.testing.assert_allclose(preintegrator.accel_bias, expected_accel_bias)
    
    def test_integrate_empty_arrays(self, preintegrator):
        """Test integration with empty arrays."""
        dt_arr = np.array([])
        omega_arr = np.array([]).reshape(0, 3)
        acc_arr = np.array([]).reshape(0, 3)
        
        dp, dv, dR = preintegrator.integrate(dt_arr, omega_arr, acc_arr)
        
        assert np.allclose(dp, np.zeros(3))
        assert np.allclose(dv, np.zeros(3))
        assert np.allclose(dR, np.eye(3))
    
    def test_integrate_single_measurement(self, preintegrator):
        """Test integration with single measurement."""
        dt = 0.01  # 10ms
        omega = np.array([0.1, 0.0, 0.0])  # rotation around x-axis
        acc = np.array([0.0, 0.0, 9.81])   # acceleration upward (cancels gravity)
        
        dt_arr = np.array([dt])
        omega_arr = np.array([omega])
        acc_arr = np.array([acc])
        
        dp, dv, dR = preintegrator.integrate(dt_arr, omega_arr, acc_arr)
        
        # Should have some rotation and no net acceleration (gravity cancellation)
        assert not np.allclose(dR, np.eye(3))  # Some rotation occurred
        assert np.linalg.norm(dv) < 0.01  # Very small velocity (gravity cancellation)
        assert np.linalg.norm(dp) < 0.001  # Very small displacement
    
    def test_integrate_multiple_measurements(self, preintegrator):
        """Test integration with multiple measurements."""
        n_samples = 10
        dt = 0.01
        
        # Constant angular velocity and acceleration
        omega = np.array([0.0, 0.0, 0.1])  # yaw rotation
        acc = np.array([1.0, 0.0, 9.81])   # forward acceleration + gravity cancellation
        
        dt_arr = np.full(n_samples, dt)
        omega_arr = np.tile(omega, (n_samples, 1))
        acc_arr = np.tile(acc, (n_samples, 1))
        
        dp, dv, dR = preintegrator.integrate(dt_arr, omega_arr, acc_arr)
        
        # Check that integration accumulated over time
        total_time = n_samples * dt
        assert preintegrator.delta_t == total_time
        
        # Should have significant rotation after multiple steps
        rotation_angle = np.linalg.norm(omega) * total_time
        assert rotation_angle > 0.05
        
        # Should have forward velocity
        assert dv[0] > 0  # Forward velocity component
    
    def test_integrate_with_bias(self, preintegrator):
        """Test integration with bias correction."""
        # Set biases
        preintegrator.gyro_bias = np.array([0.01, 0.0, 0.0])
        preintegrator.accel_bias = np.array([0.1, 0.0, 0.0])
        
        dt = 0.01
        omega = np.array([0.02, 0.0, 0.0])  # Raw gyro measurement
        acc = np.array([0.2, 0.0, 9.81])    # Raw accel measurement
        
        dt_arr = np.array([dt])
        omega_arr = np.array([omega])
        acc_arr = np.array([acc])
        
        dp1, dv1, dR1 = preintegrator.integrate(dt_arr, omega_arr, acc_arr)
        
        # Reset and integrate without bias
        preintegrator.gyro_bias = np.zeros(3)
        preintegrator.accel_bias = np.zeros(3)
        
        # Use bias-corrected measurements directly
        omega_corrected = omega - np.array([0.01, 0.0, 0.0])
        acc_corrected = acc - np.array([0.1, 0.0, 0.0])
        
        omega_arr_corrected = np.array([omega_corrected])
        acc_arr_corrected = np.array([acc_corrected])
        
        dp2, dv2, dR2 = preintegrator.integrate(dt_arr, omega_arr_corrected, acc_arr_corrected)
        
        # Results should be similar (accounting for numerical differences)
        np.testing.assert_allclose(dp1, dp2, atol=1e-6)
        np.testing.assert_allclose(dv1, dv2, atol=1e-6)
        np.testing.assert_allclose(dR1, dR2, atol=1e-6)
    
    def test_skew_symmetric(self, preintegrator):
        """Test skew-symmetric matrix generation."""
        v = np.array([1, 2, 3])
        skew = preintegrator._skew_symmetric(v)
        
        expected = np.array([
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0]
        ])
        
        np.testing.assert_array_equal(skew, expected)
    
    def test_skew_symmetric_antisymmetric(self, preintegrator):
        """Test that skew-symmetric matrix is antisymmetric."""
        v = np.random.rand(3)
        skew = preintegrator._skew_symmetric(v)
        
        # Should satisfy: S^T = -S
        np.testing.assert_allclose(skew.T, -skew)
    
    def test_get_measurement_model(self, preintegrator):
        """Test measurement model extraction."""
        # Perform some integration first
        dt_arr = np.array([0.01, 0.01])
        omega_arr = np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
        acc_arr = np.array([[1.0, 0.0, 9.81], [1.0, 0.0, 9.81]])
        
        preintegrator.integrate(dt_arr, omega_arr, acc_arr)
        
        z, P, H_bias = preintegrator.get_measurement_model()
        
        # Check dimensions
        assert z.shape == (9,)  # [δp, δv, δθ]
        assert P.shape == (9, 9)  # Covariance matrix
        assert H_bias.shape == (9, 6)  # Jacobian w.r.t. [accel_bias, gyro_bias]
        
        # Measurement should contain position, velocity, and rotation
        assert len(z) == 9
        
        # Covariance should be positive semi-definite
        eigenvals = np.linalg.eigvals(P)
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
    
    def test_correct_preintegration(self, preintegrator):
        """Test preintegration correction for bias changes."""
        # Perform initial integration
        dt_arr = np.array([0.01])
        omega_arr = np.array([[0.1, 0.0, 0.0]])
        acc_arr = np.array([[1.0, 0.0, 9.81]])
        
        preintegrator.integrate(dt_arr, omega_arr, acc_arr)
        
        # Store original results
        dp_orig = preintegrator.delta_p.copy()
        dv_orig = preintegrator.delta_v.copy()
        dR_orig = preintegrator.delta_R.copy()
        
        # Apply bias correction
        delta_bias_gyro = np.array([0.01, 0.0, 0.0])
        delta_bias_acc = np.array([0.1, 0.0, 0.0])
        
        dp_corr, dv_corr, dR_corr = preintegrator.correct_preintegration(
            delta_bias_gyro, delta_bias_acc
        )
        
        # Corrected results should be different from original
        assert not np.allclose(dp_corr, dp_orig)
        assert not np.allclose(dv_corr, dv_orig)
        assert not np.allclose(dR_corr, dR_orig)


if __name__ == "__main__":
    pytest.main([__file__])
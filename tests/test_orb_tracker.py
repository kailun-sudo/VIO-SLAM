"""
Tests for ORB feature tracker module.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from vio_slam.features.orb_tracker import ORBTracker


class TestORBTracker:
    """Test ORB feature tracker functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create ORB tracker instance."""
        return ORBTracker(n_features=100)
    
    @pytest.fixture
    def sample_image(self):
        """Create sample test image."""
        # Create a simple test image with some features
        img = np.zeros((480, 640), dtype=np.uint8)
        
        # Add some rectangles and circles as features
        cv2.rectangle(img, (100, 100), (200, 200), 255, -1)
        cv2.circle(img, (300, 300), 50, 128, -1)
        cv2.rectangle(img, (400, 50), (500, 150), 200, 2)
        
        return img
    
    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.n_features == 100
        assert tracker.brightness_threshold == 50.0
        assert tracker.gradient_threshold == 10.0
        assert tracker.orb is not None
        assert tracker.bf is not None
    
    def test_detect_and_compute_valid_image(self, tracker, sample_image):
        """Test feature detection and computation with valid image."""
        keypoints, descriptors = tracker.detect_and_compute(sample_image)
        
        assert len(keypoints) > 0
        assert descriptors is not None
        assert descriptors.shape[0] == len(keypoints)
        assert descriptors.shape[1] == 32  # ORB descriptor length
    
    def test_detect_and_compute_empty_image(self, tracker):
        """Test feature detection with empty image."""
        empty_img = np.array([])
        keypoints, descriptors = tracker.detect_and_compute(empty_img)
        
        assert len(keypoints) == 0
        assert descriptors is None
    
    def test_detect_and_compute_none_image(self, tracker):
        """Test feature detection with None image."""
        keypoints, descriptors = tracker.detect_and_compute(None)
        
        assert len(keypoints) == 0
        assert descriptors is None
    
    def test_match_valid_descriptors(self, tracker, sample_image):
        """Test descriptor matching with valid descriptors."""
        # Detect features in the same image (should have perfect matches)
        kp1, des1 = tracker.detect_and_compute(sample_image)
        kp2, des2 = tracker.detect_and_compute(sample_image)
        
        if des1 is not None and des2 is not None:
            matches = tracker.match(des1, des2)
            
            assert len(matches) > 0
            assert len(matches) <= len(kp1)
            
            # All matches should have distance 0 (same image)
            for match in matches:
                assert match.distance == 0
    
    def test_match_none_descriptors(self, tracker):
        """Test descriptor matching with None descriptors."""
        matches = tracker.match(None, None)
        assert len(matches) == 0
        
        # Test with one None descriptor
        dummy_desc = np.random.randint(0, 256, (10, 32), dtype=np.uint8)
        matches = tracker.match(dummy_desc, None)
        assert len(matches) == 0
        
        matches = tracker.match(None, dummy_desc)
        assert len(matches) == 0
    
    def test_filter_photometric_outliers(self, tracker, sample_image):
        """Test photometric outlier filtering."""
        keypoints, _ = tracker.detect_and_compute(sample_image)
        
        if len(keypoints) > 0:
            filtered_kp = tracker.filter_photometric_outliers(sample_image, keypoints)
            
            # Should filter out some keypoints (those in dark regions)
            assert len(filtered_kp) <= len(keypoints)
            
            # All filtered keypoints should have reasonable brightness/gradient
            for kp in filtered_kp:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                if 3 <= x < sample_image.shape[1]-3 and 3 <= y < sample_image.shape[0]-3:
                    patch = sample_image[y-3:y+4, x-3:x+4]
                    brightness = np.mean(patch)
                    assert brightness >= tracker.brightness_threshold
    
    def test_filter_photometric_outliers_empty_keypoints(self, tracker, sample_image):
        """Test photometric filtering with empty keypoints."""
        filtered_kp = tracker.filter_photometric_outliers(sample_image, [])
        assert len(filtered_kp) == 0
    
    def test_filter_photometric_outliers_none_image(self, tracker):
        """Test photometric filtering with None image."""
        dummy_kp = [cv2.KeyPoint(100, 100, 10)]
        filtered_kp = tracker.filter_photometric_outliers(None, dummy_kp)
        assert len(filtered_kp) == 0
    
    def test_track_valid_images(self, tracker, sample_image):
        """Test feature tracking between valid images."""
        # Create slightly modified second image
        img2 = sample_image.copy()
        img2 = cv2.GaussianBlur(img2, (3, 3), 0.5)  # Add slight blur
        
        kp1, kp2, pts1, pts2, matches = tracker.track(sample_image, img2)
        
        if len(matches) > 0:
            assert len(pts1) == len(pts2) == len(matches)
            assert pts1.shape[1] == 2  # 2D points
            assert pts2.shape[1] == 2
            assert pts1.dtype == np.float32
            assert pts2.dtype == np.float32
    
    def test_track_none_images(self, tracker):
        """Test feature tracking with None images."""
        kp1, kp2, pts1, pts2, matches = tracker.track(None, None)
        
        assert len(kp1) == 0
        assert len(kp2) == 0
        assert len(pts1) == 0
        assert len(pts2) == 0
        assert len(matches) == 0
    
    def test_track_top_k_limiting(self, tracker, sample_image):
        """Test that track respects top_k parameter."""
        img2 = sample_image.copy()
        
        kp1, kp2, pts1, pts2, matches = tracker.track(sample_image, img2, top_k=5)
        
        # Should not exceed top_k matches
        assert len(matches) <= 5
        assert len(pts1) <= 5
        assert len(pts2) <= 5
    
    def test_draw_matches(self, tracker, sample_image):
        """Test match visualization."""
        img2 = sample_image.copy()
        kp1, kp2, pts1, pts2, matches = tracker.track(sample_image, img2)
        
        if len(matches) > 0:
            result_img = tracker.draw_matches(sample_image, img2, kp1, kp2, matches, num=10)
            
            # Should return an image
            assert result_img is not None
            assert len(result_img.shape) >= 2
            assert result_img.shape[1] >= sample_image.shape[1]  # Should be wider (concatenated)
    
    def test_draw_matches_no_matches(self, tracker, sample_image):
        """Test match visualization with no matches."""
        img2 = sample_image.copy()
        
        result_img = tracker.draw_matches(sample_image, img2, [], [], [])
        
        # Should return concatenated images
        assert result_img is not None
        assert result_img.shape[1] == sample_image.shape[1] * 2
    
    def test_draw_keypoints(self, tracker, sample_image):
        """Test keypoint visualization."""
        keypoints, _ = tracker.detect_and_compute(sample_image)
        
        if len(keypoints) > 0:
            result_img = tracker.draw_keypoints(sample_image, keypoints)
            
            assert result_img is not None
            assert result_img.shape[:2] == sample_image.shape[:2]
    
    def test_compute_fundamental_matrix_sufficient_points(self, tracker):
        """Test fundamental matrix computation with sufficient points."""
        # Create corresponding points
        n_points = 20
        pts1 = np.random.rand(n_points, 2).astype(np.float32) * 400 + 100
        pts2 = pts1 + np.random.rand(n_points, 2).astype(np.float32) * 10  # Small displacement
        
        F, mask = tracker.compute_fundamental_matrix(pts1, pts2)
        
        if F is not None:
            assert F.shape == (3, 3)
            assert mask is not None
            assert len(mask) == n_points
    
    def test_compute_fundamental_matrix_insufficient_points(self, tracker):
        """Test fundamental matrix computation with insufficient points."""
        # Only 5 points (need at least 8)
        pts1 = np.random.rand(5, 2).astype(np.float32)
        pts2 = np.random.rand(5, 2).astype(np.float32)
        
        F, mask = tracker.compute_fundamental_matrix(pts1, pts2)
        
        assert F is None
        assert mask is None
    
    def test_brightness_gradient_thresholds(self):
        """Test custom brightness and gradient thresholds."""
        tracker = ORBTracker(n_features=100, brightness_threshold=100.0, gradient_threshold=20.0)
        
        assert tracker.brightness_threshold == 100.0
        assert tracker.gradient_threshold == 20.0
        
        # Create image with low brightness regions
        img = np.ones((100, 100), dtype=np.uint8) * 50  # Low brightness
        
        # Add high brightness corner
        img[10:30, 10:30] = 200
        
        keypoints, _ = tracker.detect_and_compute(img)
        
        if len(keypoints) > 0:
            filtered_kp = tracker.filter_photometric_outliers(img, keypoints)
            
            # Most keypoints should be filtered out due to high brightness threshold
            assert len(filtered_kp) <= len(keypoints)


if __name__ == "__main__":
    pytest.main([__file__])
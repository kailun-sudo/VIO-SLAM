"""
ORB feature detector and tracker with photometric outlier detection.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class ORBTracker:
    """
    ORB feature detector and tracker with enhanced outlier filtering.
    
    This class provides robust feature detection and tracking using ORB features
    with additional photometric filtering to remove low-quality matches.
    """
    
    def __init__(self, n_features: int = 1000, brightness_threshold: float = 50.0,
                 gradient_threshold: float = 10.0):
        """
        Initialize ORB tracker.
        
        Args:
            n_features: Maximum number of features to detect
            brightness_threshold: Minimum brightness for reliable features
            gradient_threshold: Minimum gradient magnitude for reliable features
        """
        self.n_features = n_features
        self.brightness_threshold = brightness_threshold
        self.gradient_threshold = gradient_threshold
        
        # Initialize ORB detector and matcher
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        logger.info(f"Initialized ORB tracker with {n_features} features")

    def detect_and_compute(self, img: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect ORB keypoints and compute descriptors.
        
        Args:
            img: Input grayscale image
            
        Returns:
            keypoints: List of detected keypoints
            descriptors: Feature descriptors array
        """
        if img is None or img.size == 0:
            return [], None
            
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        return keypoints, descriptors

    def match(self, des1: np.ndarray, des2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match descriptors between two images.
        
        Args:
            des1: Descriptors from first image
            des2: Descriptors from second image
            
        Returns:
            List of matches sorted by distance
        """
        if des1 is None or des2 is None:
            return []
            
        matches = self.bf.match(des1, des2)
        return sorted(matches, key=lambda m: m.distance)

    def filter_photometric_outliers(self, img: np.ndarray, 
                                   keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        """
        Filter keypoints in low-light or low-gradient regions.
        
        This method removes keypoints that are likely to be unreliable due to
        poor photometric conditions (low brightness or low texture).
        
        Args:
            img: Input grayscale image
            keypoints: List of keypoints to filter
            
        Returns:
            List of filtered keypoints
        """
        if not keypoints or img is None:
            return []
            
        valid_kp = []
        h, w = img.shape
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Check bounds
            if x < 3 or y < 3 or x >= w-3 or y >= h-3:
                continue
                
            # Extract local patch
            patch = img[y-3:y+4, x-3:x+4]
            
            # Check local brightness
            brightness = np.mean(patch)
            if brightness < self.brightness_threshold:
                continue
                
            # Check gradient magnitude
            grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            
            if gradient_mag > self.gradient_threshold:
                valid_kp.append(kp)
                
        logger.debug(f"Filtered {len(keypoints)} -> {len(valid_kp)} keypoints")
        return valid_kp
    
    def track(self, img1: np.ndarray, img2: np.ndarray, 
              top_k: int = 200) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], 
                                        np.ndarray, np.ndarray, List[cv2.DMatch]]:
        """
        Track features between two consecutive images with outlier filtering.
        
        Args:
            img1: First image (grayscale)
            img2: Second image (grayscale)
            top_k: Maximum number of matches to return
            
        Returns:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            pts1: Matched point coordinates from first image
            pts2: Matched point coordinates from second image
            matches: List of matches
        """
        if img1 is None or img2 is None:
            return [], [], np.array([]), np.array([]), []
        
        # Detect features
        kp1, des1 = self.detect_and_compute(img1)
        kp2, des2 = self.detect_and_compute(img2)
        
        if not kp1 or not kp2 or des1 is None or des2 is None:
            return [], [], np.array([]), np.array([]), []
        
        # Filter photometric outliers
        kp1_filtered = self.filter_photometric_outliers(img1, kp1)
        kp2_filtered = self.filter_photometric_outliers(img2, kp2)
        
        if not kp1_filtered or not kp2_filtered:
            return [], [], np.array([]), np.array([]), []
        
        # Recompute descriptors for filtered keypoints
        _, des1_filtered = self.orb.compute(img1, kp1_filtered)
        _, des2_filtered = self.orb.compute(img2, kp2_filtered)
        
        if des1_filtered is None or des2_filtered is None:
            return [], [], np.array([]), np.array([]), []
            
        # Match features
        matches = self.match(des1_filtered, des2_filtered)[:top_k]
        
        if not matches:
            return kp1_filtered, kp2_filtered, np.array([]), np.array([]), []
        
        # Extract matched points
        pts1 = np.array([kp1_filtered[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2_filtered[m.trainIdx].pt for m in matches], dtype=np.float32)
        
        logger.debug(f"Tracked {len(matches)} features between images")
        return kp1_filtered, kp2_filtered, pts1, pts2, matches

    def draw_matches(self, img1: np.ndarray, img2: np.ndarray, 
                     kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
                     matches: List[cv2.DMatch], num: int = 50) -> np.ndarray:
        """
        Visualize feature matches between two images.
        
        Args:
            img1: First image
            img2: Second image
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of matches
            num: Number of matches to visualize
            
        Returns:
            Image with drawn matches
        """
        if not matches:
            # Return concatenated images if no matches
            return np.hstack([img1, img2])
            
        return cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:num], None,
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
        )
    
    def draw_keypoints(self, img: np.ndarray, keypoints: List[cv2.KeyPoint],
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw keypoints on an image.
        
        Args:
            img: Input image
            keypoints: List of keypoints to draw
            color: Color for drawing keypoints
            
        Returns:
            Image with drawn keypoints
        """
        return cv2.drawKeypoints(
            img, keypoints, None, color=color,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
    
    def compute_fundamental_matrix(self, pts1: np.ndarray, pts2: np.ndarray,
                                  method: int = cv2.FM_RANSAC,
                                  threshold: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute fundamental matrix from matched points.
        
        Args:
            pts1: Points from first image
            pts2: Points from second image
            method: Method for fundamental matrix estimation
            threshold: RANSAC threshold
            
        Returns:
            Fundamental matrix and inlier mask
        """
        if len(pts1) < 8 or len(pts2) < 8:
            return None, None
            
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, method=method, 
            ransacReprojThreshold=threshold, confidence=0.99
        )
        
        return F, mask
"""
Loop closure detection using Bag-of-Words.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LoopClosureDetector:
    """
    Place recognition and loop closure detection using BoW.
    """
    
    def __init__(self, vocabulary_size: int = 1000):
        """
        Initialize loop closure detector.
        
        Args:
            vocabulary_size: Size of the BoW vocabulary
        """
        self.vocabulary_size = vocabulary_size
        self.descriptors_db = []
        self.keyframe_poses = []
        self.bow_vectors = []
        self.vocabulary = None
        self.similarity_threshold = 0.7
        
        logger.debug(f"Initialized loop closure detector with vocab size {vocabulary_size}")
    
    def build_vocabulary(self, all_descriptors: List[np.ndarray]):
        """
        Build BoW vocabulary from training descriptors.
        
        Args:
            all_descriptors: List of descriptor arrays from training images
        """
        try:
            from sklearn.cluster import KMeans
            if len(all_descriptors) == 0:
                logger.warning("No descriptors provided for vocabulary building")
                return
            
            all_desc = np.vstack(all_descriptors)
            n_clusters = min(self.vocabulary_size, len(all_desc))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(all_desc)
            self.vocabulary = kmeans.cluster_centers_
            
            logger.info(f"Built vocabulary with {n_clusters} words from {len(all_desc)} descriptors")
            
        except ImportError:
            logger.warning("sklearn not available, using simple vocabulary")
            # Fallback to random vocabulary
            if len(all_descriptors) > 0:
                all_desc = np.vstack(all_descriptors)
                indices = np.random.choice(
                    len(all_desc), 
                    min(self.vocabulary_size, len(all_desc)), 
                    replace=False
                )
                self.vocabulary = all_desc[indices]
                logger.info(f"Built fallback vocabulary with {len(self.vocabulary)} words")
    
    def compute_bow_vector(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Compute BoW vector for given descriptors.
        
        Args:
            descriptors: Feature descriptors array
            
        Returns:
            Normalized BoW histogram vector
        """
        if self.vocabulary is None or descriptors is None:
            return np.zeros(self.vocabulary_size)
        
        # Find closest vocabulary words
        distances = np.linalg.norm(
            descriptors[:, np.newaxis] - self.vocabulary[np.newaxis, :], axis=2
        )
        closest_words = np.argmin(distances, axis=1)
        
        # Build histogram
        bow_vector = np.bincount(closest_words, minlength=self.vocabulary_size)
        norm = np.linalg.norm(bow_vector)
        
        if norm > 1e-8:
            bow_vector = bow_vector / norm
        
        return bow_vector
    
    def add_keyframe(self, descriptors: np.ndarray, pose: np.ndarray):
        """
        Add new keyframe to database.
        
        Args:
            descriptors: Feature descriptors from keyframe
            pose: 4x4 pose matrix of keyframe
        """
        if descriptors is not None:
            self.descriptors_db.append(descriptors)
            self.keyframe_poses.append(pose.copy())
            
            bow_vector = self.compute_bow_vector(descriptors)
            self.bow_vectors.append(bow_vector)
            
            logger.debug(f"Added keyframe {len(self.descriptors_db)-1} to database")
    
    def detect_loop_closure(self, current_descriptors: np.ndarray, 
                           current_pose: np.ndarray,
                           min_frame_gap: int = 30) -> List[Tuple[int, float]]:
        """
        Detect loop closure candidates.
        
        Args:
            current_descriptors: Descriptors from current frame
            current_pose: Current pose estimate
            min_frame_gap: Minimum frame gap to consider for loop closure
            
        Returns:
            List of (frame_index, similarity_score) tuples
        """
        if len(self.bow_vectors) < min_frame_gap:
            return []
        
        current_bow = self.compute_bow_vector(current_descriptors)
        similarities = []
        
        # Compare with keyframes (excluding recent ones)
        for i, bow_vec in enumerate(self.bow_vectors[:-min_frame_gap]):
            similarity = np.dot(current_bow, bow_vec)
            similarities.append((i, similarity))
        
        # Find candidates above threshold
        candidates = [
            (i, sim) for i, sim in similarities 
            if sim > self.similarity_threshold
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            logger.info(f"Detected {len(candidates)} loop closure candidates")
        
        return candidates[:3]  # Return top 3 candidates
"""
Stability detector for steady hand detection
"""

import numpy as np
from collections import deque


class StabilityDetector:
    """Detects if hand is stable for character recognition"""
    
    def __init__(self, max_frames=30, stability_threshold=0.02):
        """
        Initialize stability detector
        
        Args:
            max_frames (int): Maximum number of frames to consider for stability
            stability_threshold (float): Threshold for determining stability
        """
        self.max_frames = max_frames
        self.stability_threshold = stability_threshold
        self.keypoints_history = deque(maxlen=max_frames)
    
    def add_keypoints(self, keypoints):
        """Add keypoints to history for stability analysis"""
        self.keypoints_history.append(keypoints)
    
    def is_stable(self):
        """Check if hand is stable based on keypoints variance"""
        if len(self.keypoints_history) < self.max_frames:
            return False
        
        arr = np.array(self.keypoints_history)
        var = np.var(arr, axis=0)
        return np.mean(var) < self.stability_threshold
    
    def reset(self):
        """Reset the stability detector"""
        self.keypoints_history.clear()

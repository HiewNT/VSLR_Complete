#!/usr/bin/env python3
"""
MediaPipe utilities cho VSLR_Complete
Dựa trên VSLR_DauThanh/utils/mediapipe_utils.py
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List

class MediaPipeHandTracker:
    """Class xử lý hand tracking với MediaPipe"""
    
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=1,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        Khởi tạo MediaPipeHandTracker
        
        Args:
            static_image_mode (bool): Chế độ xử lý ảnh tĩnh
            max_num_hands (int): Số tay tối đa
            min_detection_confidence (float): Ngưỡng tin cậy phát hiện
            min_tracking_confidence (float): Ngưỡng tin cậy tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Khởi tạo MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Landmark indices cho các điểm quan trọng
        self.landmark_indices = list(range(21))  # 21 landmarks cho mỗi tay
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Trích xuất keypoints từ frame
        
        Args:
            frame (np.ndarray): Frame đầu vào
            
        Returns:
            Optional[np.ndarray]: Keypoints (63 features) hoặc None
        """
        # Chuyển đổi BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý với MediaPipe
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Lấy tay đầu tiên
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Trích xuất keypoints
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(keypoints, dtype=np.float32)
        
        return None
    
    def draw_landmarks(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Vẽ landmarks lên frame
        
        Args:
            frame (np.ndarray): Frame gốc
            keypoints (np.ndarray): Keypoints đã trích xuất
            
        Returns:
            np.ndarray: Frame với landmarks được vẽ
        """
        # Chuyển đổi BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý với MediaPipe
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Vẽ landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame
    
    def get_hand_bbox(self, keypoints: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Lấy bounding box của tay từ keypoints
        
        Args:
            keypoints (np.ndarray): Keypoints (63 features)
            frame_shape (Tuple[int, int]): Kích thước frame (height, width)
            
        Returns:
            Optional[Tuple[int, int, int, int]]: (x, y, w, h) hoặc None
        """
        if keypoints is None or len(keypoints) != 63:
            return None
        
        # Reshape keypoints thành (21, 3)
        landmarks = keypoints.reshape(21, 3)
        
        # Lấy tọa độ x, y (bỏ qua z)
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        # Tính bounding box
        x_min = int(np.min(x_coords) * frame_shape[1])
        y_min = int(np.min(y_coords) * frame_shape[0])
        x_max = int(np.max(x_coords) * frame_shape[1])
        y_max = int(np.max(y_coords) * frame_shape[0])
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa keypoints
        
        Args:
            keypoints (np.ndarray): Keypoints gốc
            
        Returns:
            np.ndarray: Keypoints đã chuẩn hóa
        """
        if keypoints is None:
            return np.zeros(63, dtype=np.float32)
        
        # Reshape thành (21, 3)
        landmarks = keypoints.reshape(21, 3)
        
        # Chuẩn hóa theo wrist (landmark 0)
        wrist = landmarks[0]
        normalized_landmarks = landmarks - wrist
        
        # Flatten lại
        return normalized_landmarks.flatten()
    
    def get_hand_orientation(self, keypoints: np.ndarray) -> str:
        """
        Xác định hướng của tay (trái/phải)
        
        Args:
            keypoints (np.ndarray): Keypoints (63 features)
            
        Returns:
            str: 'left' hoặc 'right'
        """
        if keypoints is None or len(keypoints) != 63:
            return 'unknown'
        
        # Reshape keypoints thành (21, 3)
        landmarks = keypoints.reshape(21, 3)
        
        # Sử dụng wrist và middle finger để xác định hướng
        wrist = landmarks[0]
        middle_finger = landmarks[9]
        
        # Nếu middle finger ở bên phải wrist thì là tay trái
        if middle_finger[0] > wrist[0]:
            return 'left'
        else:
            return 'right'
    
    def close(self):
        """Đóng MediaPipe hands"""
        if hasattr(self, 'hands'):
            self.hands.close()

class MediaPipePoseTracker:
    """Class xử lý pose tracking với MediaPipe"""
    
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Khởi tạo MediaPipePoseTracker
        
        Args:
            static_image_mode (bool): Chế độ xử lý ảnh tĩnh
            model_complexity (int): Độ phức tạp mô hình (0, 1, 2)
            smooth_landmarks (bool): Làm mượt landmarks
            min_detection_confidence (float): Ngưỡng tin cậy phát hiện
            min_tracking_confidence (float): Ngưỡng tin cậy tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Khởi tạo MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract_pose_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Trích xuất pose keypoints từ frame
        
        Args:
            frame (np.ndarray): Frame đầu vào
            
        Returns:
            Optional[np.ndarray]: Pose keypoints hoặc None
        """
        # Chuyển đổi BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý với MediaPipe
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Trích xuất keypoints
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            return np.array(keypoints, dtype=np.float32)
        
        return None
    
    def draw_pose_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Vẽ pose landmarks lên frame
        
        Args:
            frame (np.ndarray): Frame gốc
            
        Returns:
            np.ndarray: Frame với pose landmarks được vẽ
        """
        # Chuyển đổi BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý với MediaPipe
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Vẽ landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return frame
    
    def close(self):
        """Đóng MediaPipe pose"""
        if hasattr(self, 'pose'):
            self.pose.close()

def test_mediapipe_utils():
    """Test function cho MediaPipe utilities"""
    print("Testing MediaPipe utilities...")
    
    # Test hand tracker
    hand_tracker = MediaPipeHandTracker()
    
    # Test với frame giả
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    keypoints = hand_tracker.extract_keypoints(test_frame)
    
    if keypoints is not None:
        print(f"✅ Hand keypoints extracted: {keypoints.shape}")
    else:
        print("ℹ️ No hand detected in test frame")
    
    # Test normalization
    if keypoints is not None:
        normalized = hand_tracker.normalize_keypoints(keypoints)
        print(f"✅ Keypoints normalized: {normalized.shape}")
    
    hand_tracker.close()
    print("✅ MediaPipe utilities test completed")

if __name__ == "__main__":
    test_mediapipe_utils()

"""
Frame processor that coordinates character and tone recognition
"""

import cv2
import numpy as np
import torch
import math
import time
from collections import deque
from ..utils.config import (
    IMAGE_SIZE, TONE_LABELS, PREDICTION_HISTORY_SIZE, 
    MIN_CONFIDENCE_THRESHOLD, TONE_CONFIDENCE_THRESHOLD, TONE_FRAMES_COUNT
)
from ..utils.recognition_utils import (
    get_bounding_box, prepare_image_for_classification, is_hand_moving
)


class FrameProcessor:
    """Main frame processor that coordinates all recognition components"""
    
    def __init__(self, detector, character_recognizer, tone_recognizer, stability_detector, text_processor):
        """
        Initialize frame processor
        
        Args:
            detector: Hand detector instance
            character_recognizer: Character recognizer instance
            tone_recognizer: Tone recognizer instance
            stability_detector: Stability detector instance
            text_processor: Text processor instance
        """
        self.detector = detector
        self.character_recognizer = character_recognizer
        self.tone_recognizer = tone_recognizer
        self.stability_detector = stability_detector
        self.text_processor = text_processor
        
        # Timing and state management
        self.last_detection_time = time.time()
        self.hand_detected_time = None
        self.recognition_started = False
        
        # Character recognition
        self.prediction = deque(maxlen=PREDICTION_HISTORY_SIZE)
        
        # Tone recognition
        self.tone_collection = False
        self.tone_frames = []
        self.tone_start_time = None
        self.after_tone_cooldown = 0
        self.hand_positions = deque(maxlen=10)
        self.movement_threshold = 0.03
        self.static_timeout = 0.3
        self.tone_motion_detected_time = None
        self.after_tone_stable_cooldown = 0
        self.tone_just_processed = False
        
        # Motion detection
        self.motion_history = deque(maxlen=15)
        self.motion_energy_history = deque(maxlen=15)
        self.motion_threshold = 0.03
        self.motion_count_required = 7
        self.gesture_active = False
        self.gesture_start_frame = None
        self.gesture_end_frame = None

    def detect_tone_action(self, keypoints):
        """Detect start of tone action"""
        if self.tone_collection:
            return False
        self.tone_collection = True
        self.tone_frames = []
        self.tone_start_time = time.time() + 0.2  # Start collecting after 0.2s
        self.tone_collecting = False
        self.tone_first_kpts = keypoints
        return True

    def finalize_tone_recognition(self):
        """Finalize tone recognition process"""
        frames_needed = TONE_FRAMES_COUNT
        duration = time.time() - self.tone_start_time if self.tone_start_time else 0
        
        # Only recognize if total movement time >= 1.2s
        if (len(self.tone_frames) >= frames_needed or duration >= 1.5 or self.tone_collecting is False):
            if duration < 1.2:
                print(f"[INFO] Động tác quá ngắn ({duration:.2f}s), bỏ qua nhận diện dấu thanh.")
                self.reset_tone_state()
                self.after_tone_stable_cooldown = time.time() + 0.5
                self.hand_positions.clear()
                return
                
            try:
                while len(self.tone_frames) < frames_needed:
                    self.tone_frames.append(self.tone_frames[-1])
                    
                tone, confidence = self.tone_recognizer.predict(self.tone_frames[:frames_needed])
                if tone and confidence >= TONE_CONFIDENCE_THRESHOLD:
                    self.text_processor.apply_tone_to_word(tone)
                    print(f"[INFO] Dấu thanh được áp dụng: {tone}, confidence: {confidence:.2f}")
                    self.tone_just_processed = True
                    self.reset_tone_state()
                    self.after_tone_stable_cooldown = time.time() + 0.7
                    self.hand_positions.clear()
                else:
                    print(f"[INFO] Độ tin cậy thấp: {confidence:.2f}, cho phép nhận diện lại")
                    self.reset_tone_state()
                    self.after_tone_stable_cooldown = time.time() + 0.3
                    self.hand_positions.clear()
            except Exception as e:
                print(f"[ERROR] Lỗi khi chạy mô hình LSTM: {e}")
                self.reset_tone_state()
                self.after_tone_cooldown = time.time() + 0.3
                self.hand_positions.clear()

    def reset_tone_state(self):
        """Reset tone recognition state"""
        self.tone_collection = False
        self.tone_frames = []
        self.tone_start_time = None
        self.last_tone_frame_time = None
        self.stability_detector.reset()
        self.after_tone_cooldown = time.time() + 1

    def process_character_recognition(self, processed_image):
        """Process character recognition"""
        try:
            results, index, _, confidence = self.character_recognizer.predict(processed_image, draw=False)
            self.prediction.append(index.item())
            probabilities = torch.softmax(torch.tensor(results), dim=0).numpy()
            recent_predictions = list(self.prediction)[-PREDICTION_HISTORY_SIZE:]
            most_common = self.text_processor.most_common_value(recent_predictions)
            
            if most_common == index.item() and confidence > MIN_CONFIDENCE_THRESHOLD:
                from ..utils.config import CLASSES
                raw_character = CLASSES[index]
                if self.text_processor.process_character(raw_character):
                    self.last_detection_time = time.time()
                    self.tone_just_processed = False
                    self.after_tone_cooldown = time.time() + 0.3
                    return True
        except Exception as e:
            print(f"[ERROR] Lỗi trong process_character_recognition: {e}")
        return False

    def reset_hand_state(self):
        """Reset hand detection state"""
        self.stability_detector.reset()
        self.reset_tone_state()
        self.hand_positions.clear()
        self.hand_detected_time = None
        self.recognition_started = False
        self.text_processor.just_processed_character = False

    def compute_motion_energy(self, kpts):
        """Compute motion energy from keypoints"""
        if len(self.motion_history) == 0:
            return 0.0
        prev_kpts = self.motion_history[-1]
        diffs = np.linalg.norm(kpts - prev_kpts, axis=1)
        return np.mean(diffs)

    def update_motion_state(self, kpts):
        """Update motion state for gesture detection"""
        self.motion_history.append(kpts)
        if len(self.motion_history) < 2:
            self.motion_energy_history.append(0.0)
            return

        energy = self.compute_motion_energy(kpts)
        self.motion_energy_history.append(energy)

        # Check consecutive threshold crossings
        above = [e > self.motion_threshold for e in list(self.motion_energy_history)[-self.motion_count_required:]]
        below = [e < self.motion_threshold for e in list(self.motion_energy_history)[-self.motion_count_required:]]

        if not self.gesture_active and all(above) and len(above) == self.motion_count_required:
            self.gesture_active = True
            self.gesture_start_frame = len(self.motion_history)
            print("[INFO] Gesture START detected")
        elif self.gesture_active and all(below) and len(below) == self.motion_count_required:
            self.gesture_active = False
            self.gesture_end_frame = len(self.motion_history)
            print("[INFO] Gesture END detected")

    def process_frame(self, frame, no_hand_threshold=1):
        """
        Main frame processing function
        
        Args:
            frame: Input video frame
            no_hand_threshold: Time threshold for no hand detection
            
        Returns:
            Processed frame
        """
        try:
            current_time = time.time()
            hands, image = self.detector.find_hands(frame)
            frame_out = frame.copy()
            
            # Handle cooldown period
            if current_time < self.after_tone_stable_cooldown:
                if hands:
                    hand = hands[0]
                    if 'landmark' in hand:
                        pinky_xy = (hand['landmark'][20].x, hand['landmark'][20].y)
                        index_xy = (hand['landmark'][8].x, hand['landmark'][8].y)
                        avg_xy = ((pinky_xy[0] + index_xy[0]) / 2, (pinky_xy[1] + index_xy[1]) / 2)
                        self.hand_positions.append(avg_xy)
                return frame_out
            
            if hands:
                hand = hands[0]
                if self.hand_detected_time is None:
                    self.hand_detected_time = current_time
                    self.recognition_started = False
                    
                time_elapsed = current_time - self.hand_detected_time
                if not self.recognition_started and time_elapsed >= 0.3:
                    self.recognition_started = True
                    
                if 'landmark' in hand and self.recognition_started:
                    kpts = np.array([[lm.x, lm.y, lm.z] for lm in hand['landmark']])
                    if len(kpts) != 21:
                        print(f"[WARNING] Số lượng landmarks không đúng: {len(kpts)} thay vì 21")
                        kpts = np.pad(kpts, ((0, max(0, 21 - len(kpts))), (0, 0)), mode='constant')[:21]
                    
                    # Update motion state
                    self.update_motion_state(kpts)
                    
                    self.stability_detector.add_keypoints(kpts.flatten())
                    wrist = hand['landmark'][0]
                    
                    # Use pinky (20) and index (8) positions for tone movement detection
                    pinky_xy = (hand['landmark'][20].x, hand['landmark'][20].y)
                    index_xy = (hand['landmark'][8].x, hand['landmark'][8].y)
                    avg_xy = ((pinky_xy[0] + index_xy[0]) / 2, (pinky_xy[1] + index_xy[1]) / 2)
                    self.hand_positions.append(avg_xy)
                    
                    hand_is_moving = is_hand_moving(self.hand_positions, self.movement_threshold)
                    can_detect_tone = (not self.tone_collection and 
                                     not self.text_processor.just_processed_character and 
                                     current_time >= self.after_tone_cooldown and 
                                     not self.tone_just_processed)
                    
                    if can_detect_tone and hand_is_moving:
                        if self.tone_motion_detected_time is None:
                            self.tone_motion_detected_time = current_time
                        elif (current_time - self.tone_motion_detected_time) >= 0.2:
                            self.detect_tone_action(kpts)
                            self.tone_motion_detected_time = None
                    else:
                        self.tone_motion_detected_time = None
                    
                    if self.tone_collection:
                        # Start collecting after 0.2s from motion detection
                        if not self.tone_collecting and current_time >= self.tone_start_time:
                            self.tone_collecting = True
                            self.tone_frames.append(self.tone_first_kpts)
                            self.last_tone_frame_time = current_time
                            
                        if self.tone_collecting:
                            # Continue collecting if moving, stop if static
                            interval = 1.5 / TONE_FRAMES_COUNT
                            if (hand_is_moving and 
                                len(self.tone_frames) < TONE_FRAMES_COUNT and 
                                (current_time - self.last_tone_frame_time) >= interval):
                                self.tone_frames.append(kpts)
                                self.last_tone_frame_time = current_time
                            elif not hand_is_moving:
                                self.tone_collecting = False
                                print(f"[INFO] Tay tĩnh, dừng thu thập frame cho dấu thanh")
                            
                            # End when enough frames, timeout, or hand static
                            if (len(self.tone_frames) >= TONE_FRAMES_COUNT or 
                                (current_time - self.tone_start_time) >= 1.5 or 
                                not self.tone_collecting):
                                self.tone_collecting = False
                                self.finalize_tone_recognition()
                    else:
                        # Character recognition when stable and not collecting tones
                        if (time.time() >= self.after_tone_cooldown and 
                            self.stability_detector.is_stable() and 
                            not self.tone_collection):
                            bbox = get_bounding_box(hand['landmark'], image.shape)
                            processed_image = prepare_image_for_classification(image, bbox, IMAGE_SIZE)
                            if processed_image is not None:
                                if self.process_character_recognition(processed_image):
                                    self.text_processor.just_processed_character = True
                                else:
                                    self.text_processor.just_processed_character = False
            else:
                self.tone_motion_detected_time = None
                if self.hand_detected_time and (current_time - self.hand_detected_time) >= no_hand_threshold:
                    self.reset_hand_state()
                if (self.text_processor.current_word and 
                    (current_time - self.last_detection_time) >= no_hand_threshold):
                    self.text_processor.finalize_word()
                    
            return frame_out
        except Exception as e:
            print(f"[ERROR] Lỗi trong process_frame: {e}")
            return frame.copy()

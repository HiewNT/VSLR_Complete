"""
Hand tracking using MediaPipe
"""

import cv2
import mediapipe as mp
import time


class HandDetector:
    """Hand detection and tracking using MediaPipe"""
    
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        """
        Initialize hand detector
        
        Args:
            mode (bool): Static image mode
            max_hands (int): Maximum number of hands to detect
            detection_con (float): Detection confidence threshold
            track_con (float): Tracking confidence threshold
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img, draw=True):
        """
        Find hands in image
        
        Args:
            img: Input image
            draw (bool): Whether to draw landmarks
            
        Returns:
            tuple: (hands_list, processed_image)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        hands = []

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                # Get bounding box and landmarks
                x_list, y_list = [], []
                h, w, _ = img.shape
                for lm in hand_lms.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

                bbox = (min(x_list), min(y_list), max(x_list) - min(x_list), max(y_list) - min(y_list))
                hands.append({
                    "bbox": bbox,
                    "landmark": hand_lms.landmark
                })

                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return hands, img

    def find_position(self, img, hand_no=0, draw=True):
        """
        Get landmark positions for specified hand
        
        Args:
            img: Input image
            hand_no (int): Hand index
            draw (bool): Whether to draw landmarks
            
        Returns:
            list: List of landmark positions
        """
        lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                hand_lms = self.results.multi_hand_landmarks[hand_no]
                h, w, _ = img.shape
                for id, lm in enumerate(hand_lms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list


def main():
    """Test function for hand detection"""
    p_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Không thể truy cập camera.")
            break

        hands, img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        if lm_list:
            print("Thumb tip position:", lm_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

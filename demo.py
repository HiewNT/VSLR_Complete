"""
Demo script for VSLR_Complete Recognition Module
Simple command-line demo without GUI
"""

import cv2
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Recognition import (
    CharacterRecognizer, ToneRecognizer,
    TextProcessor, FrameProcessor
)
from Recognition.utils import HandDetector, StabilityDetector


def main():
    """Main demo function"""
    print("🤟 VSLR Complete - Demo Mode")
    print("=" * 50)
    
    try:
        # Initialize components
        print("[INFO] Đang khởi tạo các component...")
        detector = HandDetector(max_hands=1)
        character_recognizer = CharacterRecognizer()
        tone_recognizer = ToneRecognizer()
        stability_detector = StabilityDetector()
        text_processor = TextProcessor()
        
        # Create frame processor
        frame_processor = FrameProcessor(
            detector=detector,
            character_recognizer=character_recognizer,
            tone_recognizer=tone_recognizer,
            stability_detector=stability_detector,
            text_processor=text_processor
        )
        
        print("[INFO] Khởi tạo thành công!")
        print("[INFO] Nhấn 'q' để thoát, 'c' để xóa text, 's' để lưu text")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Không thể mở camera!")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("[INFO] Camera đã sẵn sàng!")
        print("\nHướng dẫn sử dụng:")
        print("- Giữ tay tĩnh để nhận dạng ký tự")
        print("- Di chuyển tay để nhận dạng dấu thanh")
        print("- Nhấn 'c' để xóa text")
        print("- Nhấn 's' để lưu text")
        print("- Nhấn 'q' để thoát")
        print("\n" + "=" * 50)
        
        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Không thể đọc frame từ camera!")
                break
            
            # Process frame
            processed_frame = frame_processor.process_frame(frame)
            
            # Get current text
            display_text = frame_processor.text_processor.get_display_text()
            current_word = frame_processor.text_processor.get_current_word()
            
            # Display information on frame
            cv2.putText(processed_frame, f"Text: {display_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Current: {current_word}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Status
            status = "TONE" if frame_processor.tone_collection else "CHAR"
            cv2.putText(processed_frame, f"Mode: {status}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Instructions
            cv2.putText(processed_frame, "Press 'q' to quit, 'c' to clear, 's' to save", 
                       (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("VSLR Complete Demo", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                frame_processor.text_processor.clear_text()
                print("[INFO] Đã xóa text")
            elif key == ord('s'):
                save_text_to_file(display_text)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Demo kết thúc!")
        
    except Exception as e:
        print(f"[ERROR] Lỗi trong demo: {e}")
        import traceback
        traceback.print_exc()


def save_text_to_file(text):
    """Save text to file"""
    if not text or text.strip() == "":
        print("[WARNING] Không có text để lưu!")
        return
    
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vslr_demo_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text.strip())
        
        print(f"[INFO] Đã lưu text vào file: {filename}")
        print(f"[INFO] Nội dung: {text.strip()}")
        
    except Exception as e:
        print(f"[ERROR] Không thể lưu file: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script thu thập dữ liệu dấu thanh cho VSLR_Complete
Dựa trên VSLR_DauThanh/collect_data.py
"""

import sys
import cv2
import os
import time
import numpy as np
import json
from pathlib import Path
from collections import deque

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from Recognition.character_recognition.hand_tracking import handDetector

class ToneDataCollector:
    """Class thu thập dữ liệu dấu thanh giống VSLR_DauThanh"""
    
    def __init__(self, data_dir="data/tones"):
        """Khởi tạo ToneDataCollector"""
        self.data_dir = Path(data_dir)
        self.keypoints_dir = self.data_dir / "keypoints"
        self.keypoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo hand detector
        self.detector = handDetector(maxHands=1)
        
        # Các tham số thu thập
        self.total_frames = 30  # Chính xác 30 frame
        self.fps = 20  # FPS của video
        
        # Các lớp dấu thanh
        self.classes = ['huyen', 'sac', 'hoi', 'nga', 'nang']
        self.class_names = {
            'huyen': 'huyen',
            'sac': 'sac', 
            'hoi': 'hoi',
            'nga': 'nga',
            'nang': 'nang'
        }
        
        # Trạng thái hiện tại
        self.current_class = 0  # Index của lớp hiện tại
        self.is_recording = False
        self.recording_frames = []
        self.frame_count = 0
        
        # Thống kê
        self.stats = self.get_data_statistics()
        
        print("=== HỆ THỐNG THU THẬP DỮ LIỆU DẤU THANH ===")
        print(f"Số frame: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print("Các lớp dấu thanh:")
        for i, (cls, name) in enumerate(self.class_names.items(), 1):
            print(f"  {i}. {cls}: {name}")
        print("\nĐIỀU KHIỂN:")
        print("  1-5: Chọn dấu thanh")
        print("  SPACE: Bắt đầu thu thập")
        print("  Q: Thoát")
    
    def get_next_sample_id(self, class_name: str) -> int:
        """Lấy ID mẫu tiếp theo cho lớp"""
        class_dir = self.keypoints_dir / class_name
        if not class_dir.exists():
            return 0
        
        existing_samples = [f for f in class_dir.glob('*.npy')]
        return len(existing_samples)
    
    def start_recording(self):
        """Bắt đầu thu thập video"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.recording_frames = []
        self.frame_count = 0
        self.sample_id = self.get_next_sample_id(self.classes[self.current_class])
        print(f"\n🎬 Bắt đầu thu thập {self.class_names[self.classes[self.current_class]]} (ID: {self.sample_id})")
    
    def stop_recording(self):
        """Dừng thu thập video"""
        if not self.is_recording:
            return
        self.is_recording = False
        if len(self.recording_frames) == self.total_frames:
            class_name = self.classes[self.current_class]
            # Sử dụng numpy array để giảm overhead khi lưu
            frames_array = np.stack(self.recording_frames)
            self.save_keypoints(frames_array, class_name, self.sample_id)
            self.stats[class_name] += 1
            self.stats['total'] += 1
            print(f"✅ Đã lưu {len(self.recording_frames)} frames cho {self.class_names[class_name]} (ID: {self.sample_id})")
        else:
            print(f"❌ Thu thập không thành công! Chỉ có {len(self.recording_frames)} frames")
        self.recording_frames.clear()
        self.frame_count = 0

    def save_keypoints(self, keypoints: np.ndarray, class_name: str, sample_id: int):
        """Lưu keypoints vào file giống VSLR_DauThanh"""
        if class_name not in self.classes:
            raise ValueError(f"Class {class_name} không hợp lệ. Các lớp hợp lệ: {self.classes}")
        
        class_dir = self.keypoints_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Lưu keypoints
        file_path = class_dir / f"sample_{sample_id:04d}.npy"
        np.save(file_path, keypoints.astype(np.float32))
        
        # Lưu metadata
        metadata = {
            'class': class_name,
            'sample_id': sample_id,
            'num_frames': len(keypoints),
            'keypoints_shape': keypoints.shape
        }
        
        metadata_path = class_dir / f"sample_{sample_id:04d}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def get_data_statistics(self) -> dict:
        """Lấy thống kê dữ liệu"""
        stats = {}
        
        for cls in self.classes:
            class_dir = self.keypoints_dir / cls
            if class_dir.exists():
                num_samples = len(list(class_dir.glob('*.npy')))
                stats[cls] = num_samples
            else:
                stats[cls] = 0
        
        stats['total'] = sum(stats.values())
        stats['classes'] = self.classes
        
        return stats

    def process_frame(self, frame):
        """Xử lý frame và trích xuất keypoints"""
        hands, frame = self.detector.findHands(frame, draw=True)
        
        # Sử dụng biến static cho frame trống để tránh tạo mới liên tục
        if not hasattr(self, '_empty_keypoints'):
            self._empty_keypoints = np.zeros(63)
        
        if hands:
            hand = hands[0]
            if 'landmark' in hand:
                # Trích xuất keypoints giống VSLR_DauThanh
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand['landmark']]).flatten()
                if self.is_recording:
                    self.recording_frames.append(keypoints)
                    self.frame_count += 1
        else:
            if self.is_recording:
                self.recording_frames.append(self._empty_keypoints)
                self.frame_count += 1
        
        return frame
    
    def draw_ui(self, frame):
        """Vẽ giao diện người dùng lên frame giống VSLR_DauThanh"""
        # Thông tin lớp hiện tại
        current_class_name = self.class_names[self.classes[self.current_class]]
        cv2.putText(frame, f"Dau thanh: {current_class_name}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Thống kê lớp hiện tại
        current_count = self.stats.get(self.classes[self.current_class], 0)
        cv2.putText(frame, f"Label: {current_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Trạng thái thu thập
        if self.is_recording:
            cv2.putText(frame, f"Thu thập: {self.frame_count}/{self.total_frames}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Vẽ progress bar
            progress = self.frame_count / self.total_frames
            bar_width = 200
            bar_height = 20
            bar_x, bar_y = 10, 150
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (128, 128, 128), 2)
            fill_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
            
            if self.frame_count >= self.total_frames:
                cv2.putText(frame, "Hoàn thành!", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Enter Space to begin:", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Danh sách các lớp
        y_offset = 220
        for i, (cls, name) in enumerate(self.class_names.items()):
            color = (0, 255, 0) if i == self.current_class else (128, 128, 128)
            count = self.stats.get(cls, 0)
            cv2.putText(frame, f"{i+1}. {name}: {count} mẫu", (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Tổng thống kê
        cv2.putText(frame, f"Total: {self.stats['total']} mau", (10, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def run(self):
        """Chạy hệ thống thu thập dữ liệu"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở camera!")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        print("Bắt đầu hiển thị camera...")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.process_frame(frame)
                self.draw_ui(frame)
                if self.is_recording and self.frame_count >= self.total_frames:
                    self.stop_recording()
                cv2.imshow('Thu thập dữ liệu dấu thanh', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if not self.is_recording:
                        self.start_recording()
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    class_idx = key - ord('1')
                    if 0 <= class_idx < len(self.classes):
                        self.current_class = class_idx
                        print(f"Đã chọn: {self.class_names[self.classes[self.current_class]]}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n=== THỐNG KÊ CUỐI CÙNG ===")
            for cls in self.classes:
                print(f"  {self.class_names[cls]}: {self.stats[cls]} mẫu")
            print(f"Tổng cộng: {self.stats['total']} mẫu")

def main():
    """Hàm chính"""
    import argparse
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu dấu thanh")
    parser.add_argument("--data-dir", default="data/tones", 
                       help="Thư mục lưu dữ liệu")
    
    args = parser.parse_args()
    
    # Tạo collector
    collector = ToneDataCollector(args.data_dir)
    
    # Thu thập dữ liệu
    collector.run()

if __name__ == "__main__":
    main()
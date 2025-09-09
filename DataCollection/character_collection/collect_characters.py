#!/usr/bin/env python3
"""
Module thu thập dữ liệu ký tự cho VSLR_Complete
Dựa trên VSLR_Pytorch/src/collect_data.py
"""

import cv2
import os
import numpy as np
import math
import time
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from Recognition.character_recognition.hand_tracking import handDetector

class CharacterDataCollector:
    """Class thu thập dữ liệu ký tự"""
    
    def __init__(self, data_dir="data/characters", img_size=300, offset=20):
        """
        Khởi tạo CharacterDataCollector
        
        Args:
            data_dir (str): Thư mục lưu dữ liệu
            img_size (int): Kích thước ảnh đầu ra
            offset (int): Khoảng cách viền xung quanh bàn tay
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.offset = offset
        
        # Các lớp ký tự tiếng Việt
        self.classes = [
            'A', 'B', 'C', 'D', 'DD', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 
            'Mu', 'Munguoc', 'N', 'O', 'P', 'Q', 'R', 'Rau', 'S', 'T', 
            'U', 'V', 'X', 'Y'
        ]
        
        # Tạo thư mục dữ liệu nếu chưa có
        self.setup_directories()
        
        # Khởi tạo hand detector
        self.detector = handDetector(maxHands=1)
        
        # Trạng thái hiện tại
        self.current_class = 0
        self.selected_class = None
        self.existing_images = 0
        
        print("=== HỆ THỐNG THU THẬP DỮ LIỆU KÝ TỰ ===")
        print(f"Thư mục dữ liệu: {self.data_dir}")
        print(f"Kích thước ảnh: {self.img_size}x{self.img_size}")
        print(f"Số lớp ký tự: {len(self.classes)}")
        print("\nCÁC LỚP KÝ TỰ:")
        for i, cls in enumerate(self.classes, 1):
            print(f"  {i:2d}. {cls}")
        print("\nĐIỀU KHIỂN:")
        print("  1-26: Chọn ký tự")
        print("  S: Thu thập ảnh")
        print("  Q: Thoát")
    
    def setup_directories(self):
        """Tạo thư mục dữ liệu cho từng lớp"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        for label in self.classes:
            class_dir = self.data_dir / label
            class_dir.mkdir(exist_ok=True)
    
    def select_class(self, class_name):
        """Chọn lớp ký tự để thu thập"""
        if class_name in self.classes:
            self.selected_class = class_name
            class_dir = self.data_dir / class_name
            self.existing_images = len(list(class_dir.glob("*.jpg")))
            print(f"\n✅ Đã chọn lớp: {class_name}")
            print(f"📊 Số ảnh hiện có: {self.existing_images}")
            return True
        else:
            print(f"❌ Lớp '{class_name}' không hợp lệ!")
            return False
    
    def collect_image(self, img):
        """Thu thập ảnh từ frame hiện tại"""
        if not self.selected_class:
            print("❌ Vui lòng chọn lớp ký tự trước!")
            return False
        
        hands, img = self.detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Cắt vùng chứa bàn tay + padding
            x1, y1 = max(x - self.offset, 0), max(y - self.offset, 0)
            x2, y2 = min(x + w + self.offset, img.shape[1]), min(y + h + self.offset, img.shape[0])
            img_crop = img[y1:y2, x1:x2]
            
            # Tạo ảnh trắng với kích thước cố định
            img_white = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
            
            # Xác định tỉ lệ để resize
            aspect_ratio = h / w
            if aspect_ratio > 1:  # Cao hơn rộng
                scale = self.img_size / h
                w_resized = math.ceil(scale * w)
                img_resized = cv2.resize(img_crop, (w_resized, self.img_size))
                w_gap = math.ceil((self.img_size - w_resized) / 2)
                img_white[:, w_gap:w_gap + w_resized] = img_resized
            else:  # Rộng hơn cao
                scale = self.img_size / w
                h_resized = math.ceil(scale * h)
                img_resized = cv2.resize(img_crop, (self.img_size, h_resized))
                h_gap = math.ceil((self.img_size - h_resized) / 2)
                img_white[h_gap:h_gap + h_resized, :] = img_resized
            
            # Lưu ảnh
            timestamp = time.time()
            filename = self.data_dir / self.selected_class / f"Image_{timestamp}.jpg"
            cv2.imwrite(str(filename), img_white)
            self.existing_images += 1
            
            print(f"✅ Đã lưu ảnh: {filename}")
            print(f"📊 Tổng số ảnh {self.selected_class}: {self.existing_images}")
            return True
        else:
            print("❌ Không phát hiện tay trong khung hình!")
            return False
    
    def draw_ui(self, img):
        """Vẽ giao diện người dùng lên frame"""
        # Thông tin lớp hiện tại
        if self.selected_class:
            cv2.putText(img, f"Class: {self.selected_class} - Images: {self.existing_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(img, "Chọn lớp ký tự (1-26)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Hướng dẫn
        cv2.putText(img, f"Press 's' to collect class '{self.selected_class or 'NONE'}'", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "Press 'q' to exit", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Hiển thị danh sách lớp
        y_offset = 120
        for i, cls in enumerate(self.classes):
            color = (0, 255, 0) if cls == self.selected_class else (128, 128, 128)
            cv2.putText(img, f"{i+1:2d}. {cls}", 
                       (10, y_offset + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def run(self):
        """Chạy hệ thống thu thập dữ liệu"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Không thể mở camera!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎥 Bắt đầu hiển thị camera...")
        
        try:
            while True:
                ret, img = cap.read()
                if not ret:
                    break
                
                # Xử lý frame
                hands, img = self.detector.findHands(img)
                
                # Vẽ giao diện
                self.draw_ui(img)
                
                # Hiển thị frame
                cv2.imshow("Thu thập dữ liệu ký tự", img)
                
                # Xử lý phím
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.collect_image(img)
                elif key >= ord('1') and key <= ord('9'):
                    # Chọn lớp 1-9
                    class_idx = key - ord('1')
                    if class_idx < len(self.classes):
                        self.select_class(self.classes[class_idx])
                elif key >= ord('a') and key <= ord('z'):
                    # Chọn lớp 10-26 (a=10, b=11, ..., z=26)
                    class_idx = key - ord('a') + 9
                    if class_idx < len(self.classes):
                        self.select_class(self.classes[class_idx])
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n=== THỐNG KÊ CUỐI CÙNG ===")
            total_images = 0
            for cls in self.classes:
                class_dir = self.data_dir / cls
                count = len(list(class_dir.glob("*.jpg")))
                total_images += count
                print(f"  {cls}: {count} ảnh")
            print(f"Tổng cộng: {total_images} ảnh")

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu ký tự cho VSLR")
    parser.add_argument("--data-dir", default="data/characters", 
                       help="Thư mục lưu dữ liệu")
    parser.add_argument("--img-size", type=int, default=300, 
                       help="Kích thước ảnh đầu ra")
    parser.add_argument("--offset", type=int, default=20, 
                       help="Khoảng cách viền xung quanh bàn tay")
    
    args = parser.parse_args()
    
    collector = CharacterDataCollector(
        data_dir=args.data_dir,
        img_size=args.img_size,
        offset=args.offset
    )
    collector.run()

if __name__ == "__main__":
    main()

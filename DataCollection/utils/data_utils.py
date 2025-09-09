#!/usr/bin/env python3
"""
Data utilities cho VSLR_Complete
Dựa trên VSLR_DauThanh/utils/data_utils.py
"""

import os
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2

class DataProcessor:
    """Class xử lý dữ liệu cho VSLR"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Khởi tạo DataProcessor
        
        Args:
            data_dir (str): Thư mục dữ liệu chính
        """
        self.data_dir = Path(data_dir)
        self.keypoints_dir = self.data_dir / "keypoints"
        self.raw_videos_dir = self.data_dir / "raw_videos"
        
        # Các lớp dấu thanh
        self.classes = ['hoi', 'huyen', 'nang', 'nga', 'sac']
        self.class_names = {
            'hoi': 'Hỏi',
            'huyen': 'Huyền', 
            'nang': 'Nặng',
            'nga': 'Ngã',
            'sac': 'Sắc'
        }
        
        # Tạo thư mục nếu chưa có
        self.setup_directories()
    
    def setup_directories(self):
        """Tạo các thư mục cần thiết"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.keypoints_dir.mkdir(parents=True, exist_ok=True)
        self.raw_videos_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo thư mục cho từng lớp
        for class_name in self.classes:
            class_dir = self.keypoints_dir / class_name
            class_dir.mkdir(exist_ok=True)
    
    def save_keypoints(self, keypoints: np.ndarray, class_name: str, sample_id: int):
        """
        Lưu keypoints vào file
        
        Args:
            keypoints (np.ndarray): Keypoints data
            class_name (str): Tên lớp
            sample_id (int): ID mẫu
        """
        if class_name not in self.classes:
            raise ValueError(f"Invalid class name: {class_name}")
        
        class_dir = self.keypoints_dir / class_name
        
        # Lưu dưới dạng numpy array
        npy_file = class_dir / f"{sample_id:04d}.npy"
        np.save(npy_file, keypoints)
        
        # Lưu metadata dưới dạng JSON
        metadata = {
            'class_name': class_name,
            'sample_id': sample_id,
            'shape': keypoints.shape,
            'dtype': str(keypoints.dtype),
            'timestamp': self._get_timestamp()
        }
        
        json_file = class_dir / f"{sample_id:04d}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def load_keypoints(self, class_name: str, sample_id: int) -> Optional[np.ndarray]:
        """
        Tải keypoints từ file
        
        Args:
            class_name (str): Tên lớp
            sample_id (int): ID mẫu
            
        Returns:
            Optional[np.ndarray]: Keypoints data hoặc None
        """
        if class_name not in self.classes:
            return None
        
        class_dir = self.keypoints_dir / class_name
        npy_file = class_dir / f"{sample_id:04d}.npy"
        
        if npy_file.exists():
            return np.load(npy_file)
        return None
    
    def load_class_data(self, class_name: str) -> List[np.ndarray]:
        """
        Tải tất cả dữ liệu của một lớp
        
        Args:
            class_name (str): Tên lớp
            
        Returns:
            List[np.ndarray]: Danh sách keypoints
        """
        if class_name not in self.classes:
            return []
        
        class_dir = self.keypoints_dir / class_name
        keypoints_list = []
        
        for npy_file in sorted(class_dir.glob("*.npy")):
            keypoints = np.load(npy_file)
            keypoints_list.append(keypoints)
        
        return keypoints_list
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tải tất cả dữ liệu
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) - features và labels
        """
        X_list = []
        y_list = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_data = self.load_class_data(class_name)
            
            for keypoints in class_data:
                X_list.append(keypoints)
                y_list.append(class_idx)
        
        if not X_list:
            return np.array([]), np.array([])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def get_data_statistics(self) -> Dict[str, int]:
        """
        Lấy thống kê dữ liệu
        
        Returns:
            Dict[str, int]: Thống kê số lượng mẫu
        """
        stats = {}
        total = 0
        
        for class_name in self.classes:
            class_dir = self.keypoints_dir / class_name
            count = len(list(class_dir.glob("*.npy")))
            stats[class_name] = count
            total += count
        
        stats['total'] = total
        return stats
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Kiểm tra tính hợp lệ của dữ liệu
        
        Returns:
            Dict[str, Any]: Kết quả validation
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        stats = self.get_data_statistics()
        validation_result['statistics'] = stats
        
        # Kiểm tra từng lớp
        for class_name in self.classes:
            class_dir = self.keypoints_dir / class_name
            npy_files = list(class_dir.glob("*.npy"))
            json_files = list(class_dir.glob("*.json"))
            
            # Kiểm tra số lượng file
            if len(npy_files) != len(json_files):
                validation_result['warnings'].append(
                    f"Class {class_name}: Số lượng .npy và .json files không khớp"
                )
            
            # Kiểm tra từng file
            for npy_file in npy_files:
                try:
                    keypoints = np.load(npy_file)
                    
                    # Kiểm tra shape
                    if keypoints.shape != (30, 63):
                        validation_result['errors'].append(
                            f"File {npy_file}: Shape không đúng {keypoints.shape}, expected (30, 63)"
                        )
                        validation_result['valid'] = False
                    
                    # Kiểm tra NaN values
                    if np.isnan(keypoints).any():
                        validation_result['errors'].append(
                            f"File {npy_file}: Chứa NaN values"
                        )
                        validation_result['valid'] = False
                    
                except Exception as e:
                    validation_result['errors'].append(
                        f"File {npy_file}: Lỗi khi tải {str(e)}"
                    )
                    validation_result['valid'] = False
        
        return validation_result
    
    def clean_data(self, remove_invalid: bool = True) -> Dict[str, int]:
        """
        Dọn dẹp dữ liệu
        
        Args:
            remove_invalid (bool): Có xóa file không hợp lệ không
            
        Returns:
            Dict[str, int]: Số lượng file đã xóa
        """
        removed_count = {}
        
        for class_name in self.classes:
            class_dir = self.keypoints_dir / class_name
            removed_count[class_name] = 0
            
            for npy_file in class_dir.glob("*.npy"):
                try:
                    keypoints = np.load(npy_file)
                    
                    # Kiểm tra tính hợp lệ
                    is_valid = (
                        keypoints.shape == (30, 63) and
                        not np.isnan(keypoints).any()
                    )
                    
                    if not is_valid and remove_invalid:
                        # Xóa file không hợp lệ
                        npy_file.unlink()
                        
                        # Xóa file JSON tương ứng
                        json_file = npy_file.with_suffix('.json')
                        if json_file.exists():
                            json_file.unlink()
                        
                        removed_count[class_name] += 1
                        print(f"Removed invalid file: {npy_file}")
                
                except Exception as e:
                    if remove_invalid:
                        npy_file.unlink()
                        json_file = npy_file.with_suffix('.json')
                        if json_file.exists():
                            json_file.unlink()
                        removed_count[class_name] += 1
                        print(f"Removed corrupted file: {npy_file} - {str(e)}")
        
        return removed_count
    
    def export_data(self, output_file: str, format: str = 'numpy'):
        """
        Xuất dữ liệu ra file
        
        Args:
            output_file (str): File đầu ra
            format (str): Định dạng xuất ('numpy', 'pickle', 'json')
        """
        X, y = self.load_all_data()
        
        if len(X) == 0:
            print("Không có dữ liệu để xuất!")
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'numpy':
            np.savez(output_path, X=X, y=y)
        elif format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump({'X': X, 'y': y, 'classes': self.classes}, f)
        elif format == 'json':
            data = {
                'X': X.tolist(),
                'y': y.tolist(),
                'classes': self.classes,
                'class_names': self.class_names
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data exported to {output_path}")
    
    def _get_timestamp(self) -> str:
        """Lấy timestamp hiện tại"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_class_distribution(self) -> Dict[str, float]:
        """
        Lấy phân phối dữ liệu theo lớp
        
        Returns:
            Dict[str, float]: Tỷ lệ phần trăm của từng lớp
        """
        stats = self.get_data_statistics()
        total = stats['total']
        
        if total == 0:
            return {}
        
        distribution = {}
        for class_name in self.classes:
            count = stats[class_name]
            distribution[class_name] = (count / total) * 100
        
        return distribution

def test_data_utils():
    """Test function cho DataProcessor"""
    print("Testing DataProcessor...")
    
    # Tạo DataProcessor
    processor = DataProcessor("test_data")
    
    # Test tạo dữ liệu giả
    test_keypoints = np.random.rand(30, 63).astype(np.float32)
    
    # Test lưu và tải dữ liệu
    processor.save_keypoints(test_keypoints, 'hoi', 0)
    loaded_keypoints = processor.load_keypoints('hoi', 0)
    
    if loaded_keypoints is not None:
        print("✅ Keypoints saved and loaded successfully")
    else:
        print("❌ Failed to load keypoints")
    
    # Test thống kê
    stats = processor.get_data_statistics()
    print(f"✅ Data statistics: {stats}")
    
    # Test validation
    validation = processor.validate_data()
    print(f"✅ Data validation: {validation['valid']}")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_data", ignore_errors=True)
    print("✅ DataProcessor test completed")

if __name__ == "__main__":
    test_data_utils()

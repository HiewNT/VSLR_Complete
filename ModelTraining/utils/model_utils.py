#!/usr/bin/env python3
"""
Model utilities cho VSLR_Complete
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from pathlib import Path

class ModelUtils:
    """Class chứa các utility functions cho models"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """
        Đếm số parameters của model
        
        Args:
            model (nn.Module): PyTorch model
            
        Returns:
            int: Số parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size(model: nn.Module) -> str:
        """
        Lấy kích thước model
        
        Args:
            model (nn.Module): PyTorch model
            
        Returns:
            str: Kích thước model (MB)
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return f"{size_all_mb:.2f} MB"
    
    @staticmethod
    def save_model_info(model: nn.Module, 
                       model_path: str, 
                       training_info: Dict[str, Any] = None):
        """
        Lưu thông tin model
        
        Args:
            model (nn.Module): Model
            model_path (str): Đường dẫn lưu model
            training_info (Dict): Thông tin training
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Lưu model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_info': {
                'num_parameters': ModelUtils.count_parameters(model),
                'model_size': ModelUtils.get_model_size(model),
                'model_class': model.__class__.__name__
            },
            'training_info': training_info or {}
        }, model_path)
    
    @staticmethod
    def load_model_info(model_path: str) -> Dict[str, Any]:
        """
        Tải thông tin model
        
        Args:
            model_path (str): Đường dẫn model
            
        Returns:
            Dict[str, Any]: Thông tin model
        """
        checkpoint = torch.load(model_path, map_location='cpu')
        return checkpoint.get('model_info', {})
    
    @staticmethod
    def create_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> str:
        """
        Tạo summary của model
        
        Args:
            model (nn.Module): Model
            input_size (Tuple): Kích thước input
            
        Returns:
            str: Model summary
        """
        summary = []
        summary.append(f"Model: {model.__class__.__name__}")
        summary.append(f"Input size: {input_size}")
        summary.append(f"Parameters: {ModelUtils.count_parameters(model):,}")
        summary.append(f"Model size: {ModelUtils.get_model_size(model)}")
        
        return "\n".join(summary)
    
    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: List[str]):
        """
        Đóng băng các layers
        
        Args:
            model (nn.Module): Model
            layer_names (List[str]): Tên các layers cần đóng băng
        """
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                print(f"Frozen layer: {name}")
    
    @staticmethod
    def unfreeze_layers(model: nn.Module, layer_names: List[str]):
        """
        Mở băng các layers
        
        Args:
            model (nn.Module): Model
            layer_names (List[str]): Tên các layers cần mở băng
        """
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                print(f"Unfrozen layer: {name}")

class TensorFlowModelUtils:
    """Class chứa các utility functions cho TensorFlow models"""
    
    @staticmethod
    def save_model_info(model, model_path: str, training_info: Dict[str, Any] = None):
        """
        Lưu thông tin model TensorFlow
        
        Args:
            model: TensorFlow model
            model_path (str): Đường dẫn lưu model
            training_info (Dict): Thông tin training
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Lưu model
        model.save(model_path)
        
        # Lưu thông tin bổ sung
        info_path = model_path.parent / f"{model_path.stem}_info.json"
        model_info = {
            'num_parameters': model.count_params(),
            'model_class': model.__class__.__name__,
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'training_info': training_info or {}
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_model_info(model_path: str) -> Dict[str, Any]:
        """
        Tải thông tin model TensorFlow
        
        Args:
            model_path (str): Đường dẫn model
            
        Returns:
            Dict[str, Any]: Thông tin model
        """
        info_path = Path(model_path).parent / f"{Path(model_path).stem}_info.json"
        
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    @staticmethod
    def create_model_summary(model) -> str:
        """
        Tạo summary của model TensorFlow
        
        Args:
            model: TensorFlow model
            
        Returns:
            str: Model summary
        """
        summary = []
        summary.append(f"Model: {model.__class__.__name__}")
        summary.append(f"Parameters: {model.count_params():,}")
        summary.append(f"Input shape: {model.input_shape}")
        summary.append(f"Output shape: {model.output_shape}")
        
        return "\n".join(summary)

class ModelComparison:
    """Class so sánh các models"""
    
    def __init__(self):
        self.results = {}
    
    def add_model_result(self, model_name: str, 
                        accuracy: float, 
                        loss: float, 
                        inference_time: float,
                        model_size: str,
                        num_parameters: int):
        """
        Thêm kết quả model
        
        Args:
            model_name (str): Tên model
            accuracy (float): Độ chính xác
            loss (float): Loss
            inference_time (float): Thời gian inference
            model_size (str): Kích thước model
            num_parameters (int): Số parameters
        """
        self.results[model_name] = {
            'accuracy': accuracy,
            'loss': loss,
            'inference_time': inference_time,
            'model_size': model_size,
            'num_parameters': num_parameters
        }
    
    def get_best_model(self, metric: str = 'accuracy') -> str:
        """
        Lấy model tốt nhất theo metric
        
        Args:
            metric (str): Metric để so sánh
            
        Returns:
            str: Tên model tốt nhất
        """
        if not self.results:
            return None
        
        if metric == 'accuracy':
            return max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        elif metric == 'loss':
            return min(self.results.keys(), key=lambda x: self.results[x]['loss'])
        elif metric == 'inference_time':
            return min(self.results.keys(), key=lambda x: self.results[x]['inference_time'])
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def print_comparison(self):
        """In bảng so sánh"""
        if not self.results:
            print("No results to compare!")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Loss':<10} {'Inference':<12} {'Size':<10} {'Params':<10}")
        print("-"*80)
        
        for model_name, result in self.results.items():
            print(f"{model_name:<20} "
                  f"{result['accuracy']:<10.4f} "
                  f"{result['loss']:<10.4f} "
                  f"{result['inference_time']:<12.4f} "
                  f"{result['model_size']:<10} "
                  f"{result['num_parameters']:<10,}")
        
        print("="*80)
        
        # Best models
        best_acc = self.get_best_model('accuracy')
        best_loss = self.get_best_model('loss')
        best_speed = self.get_best_model('inference_time')
        
        print(f"\nBest Accuracy: {best_acc} ({self.results[best_acc]['accuracy']:.4f})")
        print(f"Best Loss: {best_loss} ({self.results[best_loss]['loss']:.4f})")
        print(f"Fastest: {best_speed} ({self.results[best_speed]['inference_time']:.4f}s)")

class ModelValidator:
    """Class kiểm tra tính hợp lệ của model"""
    
    @staticmethod
    def validate_pytorch_model(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Kiểm tra PyTorch model
        
        Args:
            model (nn.Module): Model
            input_size (Tuple): Kích thước input
            
        Returns:
            Dict[str, Any]: Kết quả validation
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Test forward pass
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, *input_size)
                output = model(dummy_input)
                
                result['info']['input_shape'] = input_size
                result['info']['output_shape'] = output.shape
                result['info']['num_parameters'] = ModelUtils.count_parameters(model)
                result['info']['model_size'] = ModelUtils.get_model_size(model)
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Forward pass failed: {str(e)}")
        
        # Kiểm tra parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if trainable_params == 0:
            result['warnings'].append("No trainable parameters found!")
        
        if total_params > 100_000_000:  # 100M parameters
            result['warnings'].append("Model has very large number of parameters")
        
        return result
    
    @staticmethod
    def validate_tensorflow_model(model, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Kiểm tra TensorFlow model
        
        Args:
            model: TensorFlow model
            input_shape (Tuple): Kích thước input
            
        Returns:
            Dict[str, Any]: Kết quả validation
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Test predict
            dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
            output = model.predict(dummy_input, verbose=0)
            
            result['info']['input_shape'] = input_shape
            result['info']['output_shape'] = output.shape
            result['info']['num_parameters'] = model.count_params()
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Predict failed: {str(e)}")
        
        # Kiểm tra parameters
        if model.count_params() > 100_000_000:  # 100M parameters
            result['warnings'].append("Model has very large number of parameters")
        
        return result

def test_model_utils():
    """Test function cho ModelUtils"""
    print("Testing ModelUtils...")
    
    # Test PyTorch model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = TestModel()
    
    # Test utilities
    param_count = ModelUtils.count_parameters(model)
    model_size = ModelUtils.get_model_size(model)
    
    print(f"✅ PyTorch model - Parameters: {param_count}, Size: {model_size}")
    
    # Test validation
    validator = ModelValidator()
    validation_result = validator.validate_pytorch_model(model, (10,))
    
    if validation_result['valid']:
        print("✅ Model validation passed")
    else:
        print(f"❌ Model validation failed: {validation_result['errors']}")
    
    print("✅ ModelUtils test completed")

if __name__ == "__main__":
    test_model_utils()

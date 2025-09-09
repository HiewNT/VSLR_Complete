#!/usr/bin/env python3
"""
Training utilities cho VSLR_Complete
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class TrainingLogger:
    """Class ghi log quá trình training"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Khởi tạo TrainingLogger
        
        Args:
            log_dir (str): Thư mục lưu log
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'timestamp': []
        }
        
        self.start_time = time.time()
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float, lr: float = None):
        """
        Ghi log một epoch
        
        Args:
            epoch (int): Số epoch
            train_loss (float): Training loss
            train_acc (float): Training accuracy
            val_loss (float): Validation loss
            val_acc (float): Validation accuracy
            lr (float): Learning rate
        """
        self.logs['epochs'].append(epoch)
        self.logs['train_loss'].append(train_loss)
        self.logs['train_acc'].append(train_acc)
        self.logs['val_loss'].append(val_loss)
        self.logs['val_acc'].append(val_acc)
        self.logs['learning_rate'].append(lr or 0)
        self.logs['timestamp'].append(time.time())
    
    def save_logs(self, filename: str = None):
        """
        Lưu logs ra file
        
        Args:
            filename (str): Tên file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.json"
        
        log_path = self.log_dir / filename
        
        # Thêm thông tin tổng quan
        log_data = {
            'training_time': time.time() - self.start_time,
            'total_epochs': len(self.logs['epochs']),
            'best_val_acc': max(self.logs['val_acc']) if self.logs['val_acc'] else 0,
            'best_val_loss': min(self.logs['val_loss']) if self.logs['val_loss'] else float('inf'),
            'logs': self.logs
        }
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"Logs saved to {log_path}")
    
    def plot_training_curves(self, save_path: str = None):
        """
        Vẽ biểu đồ training curves
        
        Args:
            save_path (str): Đường dẫn lưu biểu đồ
        """
        if not self.logs['epochs']:
            print("No logs to plot!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = self.logs['epochs']
        
        # Loss plot
        ax1.plot(epochs, self.logs['train_loss'], label='Train Loss', color='blue')
        ax1.plot(epochs, self.logs['val_loss'], label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.logs['train_acc'], label='Train Acc', color='blue')
        ax2.plot(epochs, self.logs['val_acc'], label='Val Acc', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()

class MetricsCalculator:
    """Class tính toán các metrics"""
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Tính accuracy
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            float: Accuracy
        """
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def calculate_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                                    class_names: List[str] = None) -> Dict[str, Any]:
        """
        Tính precision, recall, f1-score
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str]): Tên các lớp
            
        Returns:
            Dict[str, Any]: Metrics
        """
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        return report
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str] = None, 
                            save_path: str = None):
        """
        Vẽ confusion matrix
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str]): Tên các lớp
            save_path (str): Đường dẫn lưu biểu đồ
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()

class EarlyStopping:
    """Early stopping cho PyTorch"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, 
                 restore_best_weights: bool = True, mode: str = 'min'):
        """
        Khởi tạo EarlyStopping
        
        Args:
            patience (int): Số epochs chờ đợi
            min_delta (float): Độ chênh lệch tối thiểu
            restore_best_weights (bool): Có khôi phục weights tốt nhất không
            mode (str): 'min' hoặc 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Kiểm tra early stopping
        
        Args:
            val_score (float): Validation score
            model (nn.Module): Model
            
        Returns:
            bool: Có dừng training không
        """
        current_score = val_score
        
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        elif self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.wait = 0
            self.save_checkpoint(model)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Lưu checkpoint"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

class LearningRateScheduler:
    """Learning rate scheduler cho PyTorch"""
    
    def __init__(self, optimizer, mode: str = 'min', factor: float = 0.5, 
                 patience: int = 10, min_lr: float = 1e-7, verbose: bool = True):
        """
        Khởi tạo LearningRateScheduler
        
        Args:
            optimizer: Optimizer
            mode (str): 'min' hoặc 'max'
            factor (float): Hệ số giảm learning rate
            patience (int): Số epochs chờ đợi
            min_lr (float): Learning rate tối thiểu
            verbose (bool): Có in thông báo không
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.wait = 0
        self.best_score = None
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
    
    def step(self, val_score: float):
        """
        Cập nhật learning rate
        
        Args:
            val_score (float): Validation score
        """
        if self.best_score is None:
            self.best_score = val_score
        elif self.monitor_op(val_score, self.best_score):
            self.best_score = val_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._reduce_lr()
                self.wait = 0
    
    def _reduce_lr(self):
        """Giảm learning rate"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose:
                print(f'Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}')

class ModelCheckpoint:
    """Model checkpoint cho PyTorch"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', 
                 mode: str = 'min', save_best_only: bool = True, verbose: bool = True):
        """
        Khởi tạo ModelCheckpoint
        
        Args:
            filepath (str): Đường dẫn lưu model
            monitor (str): Metric để monitor
            mode (str): 'min' hoặc 'max'
            save_best_only (bool): Chỉ lưu model tốt nhất
            verbose (bool): Có in thông báo không
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best_score = None
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
    
    def __call__(self, val_score: float, model: nn.Module, optimizer, epoch: int):
        """
        Lưu checkpoint
        
        Args:
            val_score (float): Validation score
            model (nn.Module): Model
            optimizer: Optimizer
            epoch (int): Số epoch
        """
        if self.best_score is None or self.monitor_op(val_score, self.best_score):
            self.best_score = val_score
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': val_score
            }
            
            torch.save(checkpoint, self.filepath)
            
            if self.verbose:
                print(f'Model saved to {self.filepath} (val_score: {val_score:.4f})')

class TrainingProgressBar:
    """Progress bar cho training"""
    
    def __init__(self, total: int, desc: str = "Training"):
        """
        Khởi tạo TrainingProgressBar
        
        Args:
            total (int): Tổng số steps
            desc (str): Mô tả
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1, loss: float = None, acc: float = None):
        """
        Cập nhật progress bar
        
        Args:
            n (int): Số steps
            loss (float): Loss
            acc (float): Accuracy
        """
        self.current += n
        
        # Tính phần trăm
        percent = (self.current / self.total) * 100
        
        # Tính thời gian
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
        else:
            eta = 0
        
        # Tạo progress bar
        bar_length = 50
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Tạo thông báo
        message = f'\r{self.desc}: |{bar}| {percent:.1f}% ({self.current}/{self.total})'
        
        if loss is not None:
            message += f' Loss: {loss:.4f}'
        if acc is not None:
            message += f' Acc: {acc:.2f}%'
        
        message += f' ETA: {eta:.0f}s'
        
        print(message, end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete

def test_training_utils():
    """Test function cho TrainingUtils"""
    print("Testing TrainingUtils...")
    
    # Test TrainingLogger
    logger = TrainingLogger("test_logs")
    
    # Simulate training
    for epoch in range(5):
        train_loss = 1.0 - epoch * 0.1
        train_acc = epoch * 20
        val_loss = 1.2 - epoch * 0.08
        val_acc = epoch * 18
        lr = 0.001 * (0.9 ** epoch)
        
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr)
    
    logger.save_logs("test_training.json")
    print("✅ TrainingLogger test completed")
    
    # Test MetricsCalculator
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    
    accuracy = MetricsCalculator.calculate_accuracy(y_true, y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    
    # Test EarlyStopping
    model = torch.nn.Linear(10, 2)
    early_stopping = EarlyStopping(patience=3)
    
    for epoch in range(10):
        val_loss = 1.0 - epoch * 0.05  # Decreasing loss
        should_stop = early_stopping(val_loss, model)
        
        if should_stop:
            print(f"✅ Early stopping triggered at epoch {epoch}")
            break
    
    print("✅ TrainingUtils test completed")

if __name__ == "__main__":
    test_training_utils()

#!/usr/bin/env python3
"""
Module huấn luyện mô hình ký tự cho VSLR_Complete
Dựa trên VSLR_Pytorch/src/model.py và train.ipynb
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

class CharacterDataset(Dataset):
    """Dataset cho ký tự"""
    
    def __init__(self, data_dir: str, transform=None):
        """
        Khởi tạo CharacterDataset
        
        Args:
            data_dir (str): Thư mục dữ liệu
            transform: Transformations
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Các lớp ký tự
        self.classes = [
            'A', 'B', 'C', 'D', 'DD', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 
            'Mu', 'Munguoc', 'N', 'O', 'P', 'Q', 'R', 'Rau', 'S', 'T', 
            'U', 'V', 'X', 'Y'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Tải dữ liệu
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Tải danh sách mẫu"""
        samples = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            for img_file in class_dir.glob("*.jpg"):
                samples.append((str(img_file), class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Tải ảnh
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CharacterCNN(nn.Module):
    """CNN model cho nhận dạng ký tự"""
    
    def __init__(self, num_classes: int = 26):
        """
        Khởi tạo CharacterCNN
        
        Args:
            num_classes (int): Số lớp
        """
        super(CharacterCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CharacterTrainer:
    """Class huấn luyện mô hình ký tự"""
    
    def __init__(self, 
                 data_dir: str,
                 model_dir: str = "trained_models",
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 num_epochs: int = 100,
                 device: str = None):
        """
        Khởi tạo CharacterTrainer
        
        Args:
            data_dir (str): Thư mục dữ liệu
            model_dir (str): Thư mục lưu mô hình
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            num_epochs (int): Số epochs
            device (str): Device (cuda/cpu)
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Tạo thư mục model
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Model và optimizer
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def prepare_data(self, val_split: float = 0.2):
        """Chuẩn bị dữ liệu"""
        print("Preparing data...")
        
        # Tạo dataset
        full_dataset = CharacterDataset(self.data_dir, transform=self.train_transform)
        
        # Chia train/val
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Tạo val dataset với val transform
        val_dataset.dataset.transform = self.val_transform
        
        # Tạo dataloaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    def create_model(self, num_classes: int = 26):
        """Tạo mô hình"""
        print("Creating model...")
        
        self.model = CharacterCNN(num_classes=num_classes)
        self.model = self.model.to(self.device)
        
        # Optimizer và scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # In thông tin mô hình
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Huấn luyện một epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validation một epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Huấn luyện mô hình"""
        print("Starting training...")
        
        best_val_acc = 0.0
        best_model_path = self.model_dir / "character_model_best.pt"
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Lưu history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Lưu best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, best_model_path)
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Lưu final model
        final_model_path = self.model_dir / "character_model_final.pt"
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }, final_model_path)
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    def evaluate(self, model_path: str = None):
        """Đánh giá mô hình"""
        if model_path is None:
            model_path = self.model_dir / "character_model_best.pt"
        
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Classification report
        class_names = [
            'A', 'B', 'C', 'D', 'DD', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 
            'Mu', 'Munguoc', 'N', 'O', 'P', 'Q', 'R', 'Rau', 'S', 'T', 
            'U', 'V', 'X', 'Y'
        ]
        
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.model_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """Vẽ biểu đồ training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description="Train character recognition model")
    parser.add_argument("--data-dir", default="data/characters", 
                       help="Data directory")
    parser.add_argument("--model-dir", default="trained_models", 
                       help="Model directory")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of epochs")
    parser.add_argument("--device", default=None, 
                       help="Device (cuda/cpu)")
    parser.add_argument("--evaluate", action="store_true", 
                       help="Evaluate model only")
    parser.add_argument("--model-path", default=None, 
                       help="Model path for evaluation")
    
    args = parser.parse_args()
    
    # Tạo trainer
    trainer = CharacterTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        device=args.device
    )
    
    if args.evaluate:
        # Chỉ đánh giá
        trainer.prepare_data()
        trainer.create_model()
        trainer.evaluate(args.model_path)
    else:
        # Huấn luyện
        trainer.prepare_data()
        trainer.create_model()
        trainer.train()
        trainer.evaluate()
        trainer.plot_training_history()

if __name__ == "__main__":
    main()

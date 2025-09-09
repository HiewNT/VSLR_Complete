#!/usr/bin/env python3
"""
Module huấn luyện mô hình dấu thanh cho VSLR_Complete
Dựa trên VSLR_DauThanh/train_model.ipynb
"""

import os
import sys
import argparse
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

class ToneDataProcessor:
    """Class xử lý dữ liệu dấu thanh"""
    
    def __init__(self, data_dir: str):
        """
        Khởi tạo ToneDataProcessor
        
        Args:
            data_dir (str): Thư mục dữ liệu
        """
        self.data_dir = Path(data_dir)
        self.keypoints_dir = self.data_dir / "keypoints"
        
        # Các lớp dấu thanh
        self.classes = ['hoi', 'huyen', 'nang', 'nga', 'sac']
        self.class_names = {
            'hoi': 'Hỏi',
            'huyen': 'Huyền', 
            'nang': 'Nặng',
            'nga': 'Ngã',
            'sac': 'Sắc'
        }
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tải dữ liệu từ thư mục keypoints
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) - features và labels
        """
        X_list = []
        y_list = []
        
        print("Loading tone data...")
        
        for class_name in self.classes:
            class_dir = self.keypoints_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} not found")
                continue
            
            class_idx = self.label_encoder.transform([class_name])[0]
            npy_files = list(class_dir.glob("*.npy"))
            
            print(f"Loading {len(npy_files)} samples for class {class_name}")
            
            for npy_file in npy_files:
                try:
                    keypoints = np.load(npy_file)
                    
                    # Kiểm tra shape
                    if keypoints.shape == (30, 63):
                        X_list.append(keypoints)
                        y_list.append(class_idx)
                    else:
                        print(f"Warning: Invalid shape {keypoints.shape} in {npy_file}")
                
                except Exception as e:
                    print(f"Error loading {npy_file}: {e}")
        
        if not X_list:
            raise ValueError("No valid data found!")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Loaded {len(X)} samples with shape {X.shape}")
        return X, y
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray, 
                       test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Tiền xử lý dữ liệu
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            test_size (float): Tỷ lệ test set
            random_state (int): Random seed
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test, y_train_cat, y_test_cat)
        """
        print("Preprocessing data...")
        
        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # One-hot encoding
        y_train_cat = to_categorical(y_train, num_classes=len(self.classes))
        y_test_cat = to_categorical(y_test, num_classes=len(self.classes))
        
        print(f"Train set: {X_train.shape}, {y_train_cat.shape}")
        print(f"Test set: {X_test.shape}, {y_test_cat.shape}")
        
        return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat

class ToneLSTMModel:
    """LSTM model cho nhận dạng dấu thanh"""
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int = 5):
        """
        Khởi tạo ToneLSTMModel
        
        Args:
            input_shape (Tuple[int, int]): Shape đầu vào (sequence_length, features)
            num_classes (int): Số lớp
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def create_model(self) -> Model:
        """Tạo mô hình LSTM"""
        print("Creating LSTM model...")
        
        # Input layer
        inputs = Input(shape=self.input_shape, name='input_layer')
        
        # LSTM layers
        x = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
        x = BatchNormalization()(x)
        
        x = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
        x = BatchNormalization()(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='output_layer')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_model_summary(self):
        """In thông tin mô hình"""
        if self.model is None:
            print("Model not created yet!")
            return
        
        self.model.summary()
        
        # Tính số parameters
        total_params = self.model.count_params()
        print(f"\nTotal parameters: {total_params:,}")

class ToneMLPModel:
    """MLP model cho nhận dạng dấu thanh"""
    
    def __init__(self, input_shape: int, num_classes: int = 5):
        """
        Khởi tạo ToneMLPModel
        
        Args:
            input_shape (int): Số features đầu vào (flattened)
            num_classes (int): Số lớp
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def create_model(self) -> Model:
        """Tạo mô hình MLP"""
        print("Creating MLP model...")
        
        # Input layer
        inputs = Input(shape=(self.input_shape,), name='input_layer')
        
        # Dense layers
        x = Dense(512, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='output_layer')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

class ToneTrainer:
    """Class huấn luyện mô hình dấu thanh"""
    
    def __init__(self, 
                 data_dir: str,
                 model_dir: str = "trained_models",
                 batch_size: int = 32,
                 epochs: int = 100):
        """
        Khởi tạo ToneTrainer
        
        Args:
            data_dir (str): Thư mục dữ liệu
            model_dir (str): Thư mục lưu mô hình
            batch_size (int): Batch size
            epochs (int): Số epochs
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Tạo thư mục model
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Data processor
        self.data_processor = ToneDataProcessor(str(self.data_dir))
        
        # Models
        self.lstm_model = None
        self.mlp_model = None
        
        # Training history
        self.lstm_history = None
        self.mlp_history = None
    
    def prepare_data(self):
        """Chuẩn bị dữ liệu"""
        print("Preparing data...")
        
        # Tải dữ liệu
        X, y = self.data_processor.load_data()
        
        # Tiền xử lý
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_train_cat, self.y_test_cat = \
            self.data_processor.preprocess_data(X, y)
        
        # Tạo MLP data (flatten)
        self.X_train_flat = self.X_train.reshape(self.X_train.shape[0], -1)
        self.X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
        
        print(f"LSTM input shape: {self.X_train.shape}")
        print(f"MLP input shape: {self.X_train_flat.shape}")
    
    def train_lstm(self):
        """Huấn luyện mô hình LSTM"""
        print("\n" + "="*50)
        print("Training LSTM Model")
        print("="*50)
        
        # Tạo mô hình
        self.lstm_model = ToneLSTMModel(
            input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
            num_classes=len(self.data_processor.classes)
        )
        model = self.lstm_model.create_model()
        self.lstm_model.get_model_summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.model_dir / 'lstm_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Training
        print("Starting LSTM training...")
        start_time = time.time()
        
        self.lstm_history = model.fit(
            self.X_train, self.y_train_cat,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_test, self.y_test_cat),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"LSTM training completed in {training_time:.2f} seconds")
        
        # Lưu final model
        model.save(self.model_dir / 'lstm_model_final.h5')
        print("LSTM model saved!")
    
    def train_mlp(self):
        """Huấn luyện mô hình MLP"""
        print("\n" + "="*50)
        print("Training MLP Model")
        print("="*50)
        
        # Tạo mô hình
        self.mlp_model = ToneMLPModel(
            input_shape=self.X_train_flat.shape[1],
            num_classes=len(self.data_processor.classes)
        )
        model = self.mlp_model.create_model()
        self.mlp_model.get_model_summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.model_dir / 'mlp_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Training
        print("Starting MLP training...")
        start_time = time.time()
        
        self.mlp_history = model.fit(
            self.X_train_flat, self.y_train_cat,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_test_flat, self.y_test_cat),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"MLP training completed in {training_time:.2f} seconds")
        
        # Lưu final model
        model.save(self.model_dir / 'mlp_model_final.h5')
        print("MLP model saved!")
    
    def evaluate_models(self):
        """Đánh giá các mô hình"""
        print("\n" + "="*50)
        print("Evaluating Models")
        print("="*50)
        
        # Đánh giá LSTM
        if self.lstm_model is not None:
            print("\nLSTM Model Evaluation:")
            lstm_loss, lstm_acc = self.lstm_model.model.evaluate(
                self.X_test, self.y_test_cat, verbose=0
            )
            print(f"LSTM - Loss: {lstm_loss:.4f}, Accuracy: {lstm_acc:.4f}")
            
            # Predictions
            lstm_pred = self.lstm_model.model.predict(self.X_test)
            lstm_pred_classes = np.argmax(lstm_pred, axis=1)
            
            # Classification report
            print("\nLSTM Classification Report:")
            print(classification_report(
                self.y_test, lstm_pred_classes, 
                target_names=self.data_processor.class_names.values()
            ))
        
        # Đánh giá MLP
        if self.mlp_model is not None:
            print("\nMLP Model Evaluation:")
            mlp_loss, mlp_acc = self.mlp_model.model.evaluate(
                self.X_test_flat, self.y_test_cat, verbose=0
            )
            print(f"MLP - Loss: {mlp_loss:.4f}, Accuracy: {mlp_acc:.4f}")
            
            # Predictions
            mlp_pred = self.mlp_model.model.predict(self.X_test_flat)
            mlp_pred_classes = np.argmax(mlp_pred, axis=1)
            
            # Classification report
            print("\nMLP Classification Report:")
            print(classification_report(
                self.y_test, mlp_pred_classes, 
                target_names=self.data_processor.class_names.values()
            ))
    
    def plot_training_history(self):
        """Vẽ biểu đồ training history"""
        if self.lstm_history is None and self.mlp_history is None:
            print("No training history to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # LSTM plots
        if self.lstm_history is not None:
            # Loss
            axes[0, 0].plot(self.lstm_history.history['loss'], label='Train Loss')
            axes[0, 0].plot(self.lstm_history.history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('LSTM Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy
            axes[0, 1].plot(self.lstm_history.history['accuracy'], label='Train Acc')
            axes[0, 1].plot(self.lstm_history.history['val_accuracy'], label='Val Acc')
            axes[0, 1].set_title('LSTM Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # MLP plots
        if self.mlp_history is not None:
            # Loss
            axes[1, 0].plot(self.mlp_history.history['loss'], label='Train Loss')
            axes[1, 0].plot(self.mlp_history.history['val_loss'], label='Val Loss')
            axes[1, 0].set_title('MLP Model Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Accuracy
            axes[1, 1].plot(self.mlp_history.history['accuracy'], label='Train Acc')
            axes[1, 1].plot(self.mlp_history.history['val_accuracy'], label='Val Acc')
            axes[1, 1].set_title('MLP Model Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'tone_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_label_encoder(self):
        """Lưu label encoder"""
        encoder_path = self.model_dir / 'lstm_model_label_encoder.pkl'
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.data_processor.label_encoder, f)
        print(f"Label encoder saved to {encoder_path}")
    
    def train(self):
        """Huấn luyện tất cả mô hình"""
        print("Starting tone model training...")
        
        # Chuẩn bị dữ liệu
        self.prepare_data()
        
        # Huấn luyện LSTM
        self.train_lstm()
        
        # Huấn luyện MLP
        self.train_mlp()
        
        # Đánh giá
        self.evaluate_models()
        
        # Vẽ biểu đồ
        self.plot_training_history()
        
        # Lưu label encoder
        self.save_label_encoder()
        
        print("\nTone model training completed!")

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description="Train tone recognition models")
    parser.add_argument("--data-dir", default="data/tones", 
                       help="Data directory")
    parser.add_argument("--model-dir", default="trained_models", 
                       help="Model directory")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of epochs")
    parser.add_argument("--model-type", choices=['lstm', 'mlp', 'both'], default='both',
                       help="Model type to train")
    
    args = parser.parse_args()
    
    # Tạo trainer
    trainer = ToneTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Chuẩn bị dữ liệu
    trainer.prepare_data()
    
    # Huấn luyện theo loại mô hình
    if args.model_type in ['lstm', 'both']:
        trainer.train_lstm()
    
    if args.model_type in ['mlp', 'both']:
        trainer.train_mlp()
    
    # Đánh giá và vẽ biểu đồ
    trainer.evaluate_models()
    trainer.plot_training_history()
    trainer.save_label_encoder()

if __name__ == "__main__":
    main()

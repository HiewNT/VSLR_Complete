"""
Tone recognition using LSTM model
"""

import os
import numpy as np
import tensorflow as tf
import pickle
from ..utils.config import TONE_FRAMES_COUNT, TONE_CONFIDENCE_THRESHOLD


class ToneRecognizer:
    """Tone recognition using trained LSTM model"""
    
    def __init__(self, model_path=None):
        """
        Initialize ToneRecognizer with LSTM model
        
        Args:
            model_path (str): Path to LSTM model. If None, uses default path
        """
        self.model = None
        self.sequence_length = TONE_FRAMES_COUNT  # 30 frames
        self.prediction_threshold = TONE_CONFIDENCE_THRESHOLD  # 0.8
        
        # Auto-select model path if not provided
        if model_path is None:
            self.model_path = "trained_models/lstm_model_final.h5"
        else:
            self.model_path = model_path
            
        self.label_encoder = None
        self.classes = []  # List of labels after decoding from label_encoder
        self.current_prediction = None
        self.current_confidence = 0.0

        # Load model and label encoder
        self.load_model()

    def load_model(self):
        """Load LSTM model and label encoder from specified path"""
        if not os.path.exists(self.model_path):
            print(f"[ERROR] Không tìm thấy mô hình tại {self.model_path}!")
            return

        try:
            print(f"[INFO] Đang tải mô hình LSTM từ: {self.model_path}")
            
            # Try loading model with custom_objects to handle compatibility issues
            try:
                self.model = tf.keras.models.load_model(self.model_path)
            except Exception as e:
                print(f"[WARNING] Lỗi khi tải mô hình thông thường: {e}")
                print("[INFO] Thử tải với custom_objects...")
                
                # Create custom_objects to handle batch_shape errors
                def custom_input_layer(*args, **kwargs):
                    # Remove batch_shape if present
                    if 'batch_shape' in kwargs:
                        del kwargs['batch_shape']
                    return tf.keras.layers.InputLayer(*args, **kwargs)
                
                custom_objects = {
                    'InputLayer': custom_input_layer
                }
                
                try:
                    self.model = tf.keras.models.load_model(
                        self.model_path, 
                        custom_objects=custom_objects,
                        compile=False
                    )
                    print("[INFO] Tải mô hình thành công với custom_objects!")
                except Exception as e2:
                    print(f"[ERROR] Vẫn không thể tải mô hình: {e2}")
                    # Try loading with compile=False
                    try:
                        self.model = tf.keras.models.load_model(
                            self.model_path, 
                            compile=False
                        )
                        print("[INFO] Tải mô hình thành công với compile=False!")
                    except Exception as e3:
                        print(f"[ERROR] Không thể tải mô hình: {e3}")
                        self.model = None
                        return
            
            # Print detailed model information
            if self.model is not None:
                print(f"[INFO] Mô hình LSTM đã tải thành công!")
                print(f"[INFO] Input shape: {self.model.input_shape}")
                print(f"[INFO] Output shape: {self.model.output_shape}")
            
            # Load label encoder
            encoder_path = self.model_path.replace("_final.h5", "_label_encoder.pkl")
            
            if os.path.exists(encoder_path):
                with open(encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                    self.classes = list(self.label_encoder.classes_)
                    print(f"[INFO] Label encoder đã tải: {self.classes}")
            else:
                print(f"[WARNING] Không tìm thấy label encoder tại {encoder_path}")
                self.label_encoder = None
                self.classes = []

            print(f"[INFO] Số lớp dự đoán: {len(self.classes)}")
        except Exception as e:
            print(f"[ERROR] Lỗi khi tải mô hình hoặc label encoder: {e}")
            self.model = None

    def preprocess_keypoints(self, keypoints_sequence):
        """Normalize keypoints sequence for LSTM prediction"""
        keypoints_array = np.array(keypoints_sequence)
        
        # Ensure we have enough frames
        if keypoints_array.shape[0] < self.sequence_length:
            padding = np.zeros((self.sequence_length - keypoints_array.shape[0], keypoints_array.shape[1]))
            keypoints_array = np.vstack([keypoints_array, padding])
        else:
            keypoints_array = keypoints_array[:self.sequence_length]
        
        # Reshape for LSTM: (batch, sequence, features)
        return keypoints_array.reshape(1, self.sequence_length, -1)

    def predict(self, keypoints_sequence):
        """
        Predict tone from keypoints sequence
        
        Args:
            keypoints_sequence: List of keypoints arrays
            
        Returns:
            tuple: (predicted_tone, confidence)
        """
        if self.model is None:
            print(f"[WARNING] Không thể dự đoán: Mô hình LSTM chưa được tải.")
            return None, 0.0

        if len(keypoints_sequence) != self.sequence_length:
            print(f"[WARNING] Số frame không đúng: {len(keypoints_sequence)}/{self.sequence_length}")
            return None, 0.0

        try:
            X = self.preprocess_keypoints(keypoints_sequence)
            
            predictions = self.model.predict(X, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]

            if self.label_encoder:
                predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
            else:
                predicted_label = str(predicted_idx)  # fallback if label_encoder missing

            self.current_prediction = predicted_label
            self.current_confidence = confidence
            
            return self.current_prediction, self.current_confidence
        except Exception as e:
            print(f"[ERROR] Lỗi khi dự đoán dấu thanh với mô hình LSTM: {e}")
            return None, 0.0
    
    def is_model_loaded(self):
        """Check if model is successfully loaded"""
        return self.model is not None
    
    def get_available_tones(self):
        """Get list of available tone classes"""
        return self.classes.copy() if self.classes else []

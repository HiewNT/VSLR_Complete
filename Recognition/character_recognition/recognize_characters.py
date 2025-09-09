"""
Character recognition using ResNet50 model
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50
from ..utils.config import CLASSES


class CharacterRecognizer:
    """Character recognition using trained ResNet50 model"""
    
    def __init__(self, model_path="trained_models/character_model_best.pt"):
        """
        Initialize character recognizer
        
        Args:
            model_path (str): Path to trained character model
        """
        self.model_path = model_path
        self.model = resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASSES))
        
        # Load model on CPU
        if self.model_path is not None and os.path.isfile(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint["model_params"])
            print(f"[INFO] Đã tải mô hình ký tự từ: {self.model_path}")
        else:
            print(f"[ERROR] Không tìm thấy mô hình ký tự tại: {self.model_path}")
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model.to(torch.device("cpu"))
        self.model.eval()

    def predict(self, image, draw=True):
        """
        Predict character from image
        
        Args:
            image: Input image
            draw (bool): Whether to draw prediction on image
            
        Returns:
            tuple: (raw_results, prediction_index, output_image, confidence)
        """
        device = torch.device("cpu")
        # Save original image for drawing
        image_to_draw = image.copy() if draw else None
        
        # Preprocessing
        image = cv2.resize(image, (224, 224))
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image = image[None, :, :, :]
        image = torch.from_numpy(image).float().to(device)

        # Prediction
        with torch.no_grad():
            results = self.model(image)
            probabilities = torch.softmax(results, dim=1).numpy()[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]

        if draw and image_to_draw is not None:
            cv2.putText(image_to_draw, f"{CLASSES[prediction]} ({confidence:.2f})", 
                       (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            return list(results[0]), prediction, image_to_draw, confidence

        return list(results[0]), prediction, image, confidence
    
    def get_character_name(self, prediction_index):
        """Get character name from prediction index"""
        if 0 <= prediction_index < len(CLASSES):
            return CLASSES[prediction_index]
        return None

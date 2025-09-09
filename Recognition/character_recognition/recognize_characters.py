"""
Character recognition using ResNet50 model
"""

import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet50
from PIL import Image
from ..utils.config import CLASSES


class CharacterResNet50(nn.Module):
    """ResNet50 model cho nhận dạng ký tự - tương tự như trong training"""
    
    def __init__(self, num_classes: int = 26, pretrained: bool = False):
        """
        Khởi tạo CharacterResNet50
        
        Args:
            num_classes (int): Số lớp
            pretrained (bool): Sử dụng pretrained weights
        """
        super(CharacterResNet50, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = resnet50(pretrained=pretrained)
        
        # Freeze early layers for transfer learning
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze last few layers
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            for param in self.backbone.avgpool.parameters():
                param.requires_grad = True
        
        # Replace the final classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class CharacterRecognizer:
    """Character recognition using trained ResNet50 model"""
    
    def __init__(self, model_path="trained_models/character_model_epoch_1.pt"):
        """
        Initialize character recognizer
        
        Args:
            model_path (str): Path to trained character model
        """
        self.model_path = model_path
        self.device = torch.device("cpu")
        
        # Tạo mô hình với cấu trúc giống như training
        self.model = CharacterResNet50(num_classes=len(CLASSES), pretrained=False)
        
        # Transform cho preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model on CPU
        if self.model_path is not None and os.path.isfile(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"[INFO] Đã tải mô hình ký tự từ: {self.model_path}")
                print(f"[INFO] Best validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
                print(f"[INFO] Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
            except Exception as e:
                print(f"[ERROR] Lỗi khi tải mô hình: {e}")
                raise
        else:
            print(f"[ERROR] Không tìm thấy mô hình ký tự tại: {self.model_path}")
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image, draw=True):
        """
        Predict character from image
        
        Args:
            image: Input image (OpenCV format - BGR)
            draw (bool): Whether to draw prediction on image
            
        Returns:
            tuple: (raw_results, prediction_index, output_image, confidence)
        """
        # Save original image for drawing
        image_to_draw = image.copy() if draw else None
        
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transforms
        image_tensor = self.transform(pil_image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)

        # Prediction
        with torch.no_grad():
            results = self.model(image_tensor)
            probabilities = torch.softmax(results, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]

        if draw and image_to_draw is not None:
            cv2.putText(image_to_draw, f"{CLASSES[prediction]} ({confidence:.2f})", 
                       (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            return list(results[0].cpu()), prediction, image_to_draw, confidence

        return list(results[0].cpu()), prediction, image, confidence
    
    def get_character_name(self, prediction_index):
        """Get character name from prediction index"""
        if 0 <= prediction_index < len(CLASSES):
            return CLASSES[prediction_index]
        return None

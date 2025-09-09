"""
Recognition utilities and helper functions
"""

import cv2
import numpy as np
from collections import Counter


def most_common_value(sequence):
    """Get the most common value from a sequence"""
    counter = Counter(sequence)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None


def special_characters_prediction(sentence, character):
    """Predict special characters based on context"""
    if sentence:
        character = 'AW' if (sentence[-1] in ['A', 'AW'] and character == 'Munguoc') else character
        character = 'AA' if (sentence[-1] in ['A', 'AA'] and character == 'Mu') else character
        character = 'EE' if (sentence[-1] in ['E', 'EE'] and character == 'Mu') else character
        character = 'UW' if (sentence[-1] in ['U', 'UW'] and character == 'Rau') else character
        character = 'OW' if (sentence[-1] in ['O', 'OW'] and character == 'Rau') else character
        character = 'OO' if (sentence[-1] in ['O', 'OO'] and character == 'Mu') else character
    return character


def get_bounding_box(landmarks, shape):
    """Get bounding box from landmarks"""
    if not landmarks:
        return None
    h, w, _ = shape
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs) * w, max(xs) * w
    y_min, y_max = min(ys) * h, max(ys) * h
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)


def prepare_image_for_classification(image, bbox, image_size=300):
    """Prepare image for character classification"""
    if bbox is None:
        return None
    x, y, w, h = bbox
    h_img, w_img, _ = image.shape
    x1, y1 = max(0, x - 20), max(0, y - 20)
    x2, y2 = min(w_img, x + w + 20), min(h_img, y + h + 20)
    img_crop = image[y1:y2, x1:x2]
    if img_crop.size == 0:
        return None
    
    img_white = np.ones((image_size, image_size, 3), np.uint8) * 255
    aspect_ratio = h / w
    if aspect_ratio > 1:
        k = image_size / h
        w_cal = int(round(k * w))
        img_resize = cv2.resize(img_crop, (w_cal, image_size))
        img_resize = img_resize[:, :image_size]
        w_gap = (image_size - img_resize.shape[1]) // 2
        img_white[:, w_gap:w_gap + img_resize.shape[1]] = img_resize
    else:
        k = image_size / w
        h_cal = int(round(k * h))
        img_resize = cv2.resize(img_crop, (image_size, h_cal))
        img_resize = img_resize[:image_size, :]
        h_gap = (image_size - img_resize.shape[0]) // 2
        img_white[h_gap:h_gap + img_resize.shape[0], :] = img_resize
    return img_white


def is_hand_moving(hand_positions, threshold=0.03):
    """Check if hand is moving based on position history"""
    if len(hand_positions) < 2:
        return False
    arr = np.array(hand_positions)
    diffs = np.diff(arr, axis=0)
    total_move = np.sum(np.linalg.norm(diffs, axis=1))
    return total_move > threshold

# Recognition Module - VSLR Complete

Module nhận dạng ngôn ngữ ký hiệu tiếng Việt hoàn chỉnh, được tổ chức theo cấu trúc tương tự như DataCollection và ModelTraining.

## 📁 Cấu trúc Module

```
Recognition/
├── __init__.py                    # Main module exports
├── character_recognition/         # Nhận dạng ký tự
│   ├── __init__.py
│   └── recognize_characters.py    # CharacterRecognizer class
├── tone_recognition/              # Nhận dạng dấu thanh
│   ├── __init__.py
│   └── recognize_tones.py         # ToneRecognizer class
├── text_processing/               # Xử lý văn bản
│   ├── __init__.py
│   └── process_text.py            # TextProcessor class
├── frame_processing/              # Xử lý frame chính
│   ├── __init__.py
│   └── process_frames.py          # FrameProcessor class
├── utils/                         # Utilities và helper functions
│   ├── __init__.py
│   ├── recognition_utils.py       # Common utilities
│   ├── hand_tracking.py           # HandDetector class
│   ├── stability_detector.py      # StabilityDetector class
│   └── config.py                  # Configuration constants
└── README.md                      # Documentation
```

## 🧩 Các Submodule

### 1. Character Recognition (`character_recognition/`)
- **File chính**: `recognize_characters.py`
- **Class**: `CharacterRecognizer`
- **Chức năng**: Nhận dạng 26 ký tự cơ bản tiếng Việt
- **Mô hình**: ResNet50 fine-tuned
- **Input**: Ảnh tay tĩnh (224x224)
- **Output**: Ký tự được nhận dạng + độ tin cậy

### 2. Tone Recognition (`tone_recognition/`)
- **File chính**: `recognize_tones.py`
- **Class**: `ToneRecognizer`
- **Chức năng**: Nhận dạng 5 dấu thanh (huyền, sắc, hỏi, ngã, nặng)
- **Mô hình**: LSTM
- **Input**: Chuỗi 30 frame keypoints chuyển động
- **Output**: Dấu thanh được nhận dạng + độ tin cậy

### 3. Text Processing (`text_processing/`)
- **File chính**: `process_text.py`
- **Class**: `TextProcessor`
- **Chức năng**: 
  - Kết hợp ký tự và dấu thanh thành chữ hoàn chỉnh
  - Xử lý ký tự đặc biệt (Â, Ă, Ê, Ô, Ơ, Ư)
  - Quản lý từ và câu
  - Cache hiệu suất

### 4. Frame Processing (`frame_processing/`)
- **File chính**: `process_frames.py`
- **Class**: `FrameProcessor`
- **Chức năng**: Điều phối toàn bộ quá trình nhận dạng
- **Logic**:
  - Phát hiện tay tĩnh → Nhận dạng ký tự
  - Phát hiện chuyển động → Nhận dạng dấu thanh
  - Kết hợp kết quả thành văn bản hoàn chỉnh

### 5. Utils (`utils/`)
- **`recognition_utils.py`**: Utility functions chung
- **`hand_tracking.py`**: HandDetector class (MediaPipe)
- **`stability_detector.py`**: StabilityDetector class
- **`config.py`**: Configuration constants và mappings

## 🚀 Cách sử dụng

### Import cơ bản:
```python
from Recognition import (
    CharacterRecognizer, ToneRecognizer,
    TextProcessor, FrameProcessor
)
from Recognition.utils import HandDetector, StabilityDetector
```

### Sử dụng đầy đủ:
```python
# Khởi tạo các component
detector = HandDetector(max_hands=1)
character_recognizer = CharacterRecognizer()
tone_recognizer = ToneRecognizer()
stability_detector = StabilityDetector()
text_processor = TextProcessor()

# Tạo frame processor
frame_processor = FrameProcessor(
    detector=detector,
    character_recognizer=character_recognizer,
    tone_recognizer=tone_recognizer,
    stability_detector=stability_detector,
    text_processor=text_processor
)

# Xử lý frame
processed_frame = frame_processor.process_frame(frame)

# Lấy kết quả
display_text = text_processor.get_display_text()
```

## 🔄 Quy trình nhận dạng

1. **Phát hiện tay**: MediaPipe phát hiện landmarks của bàn tay
2. **Phân biệt trạng thái**:
   - **Tay tĩnh**: StabilityDetector xác nhận → CharacterRecognizer
   - **Tay chuyển động**: Thu thập 30 frame → ToneRecognizer
3. **Xử lý ký tự**: ResNet50 phân loại ký tự
4. **Xử lý dấu thanh**: LSTM phân loại dấu thanh
5. **Kết hợp**: TextProcessor kết hợp ký tự + dấu thanh
6. **Hiển thị**: Cập nhật giao diện với kết quả

## ⚙️ Cấu hình

### Constants quan trọng (trong `utils/config.py`):
```python
IMAGE_SIZE = 300                   # Kích thước ảnh cho classification
TONE_FRAMES_COUNT = 30             # Số frame cần thiết cho dấu thanh
MIN_CONFIDENCE_THRESHOLD = 0.98    # Ngưỡng tin cậy cho ký tự
TONE_CONFIDENCE_THRESHOLD = 0.8    # Ngưỡng tin cậy cho dấu thanh
```

### Classes được hỗ trợ:
- **Ký tự**: A, B, C, D, DD, E, G, H, I, K, L, M, Mu, Munguoc, N, O, P, Q, R, Rau, S, T, U, V, X, Y
- **Dấu thanh**: huyen, sac, hoi, nga, nang

## 🎯 Logic phân biệt ký tự vs dấu thanh

```python
# Phát hiện chuyển động
hand_is_moving = is_hand_moving(hand_positions)

# Nhận dạng ký tự khi tay tĩnh
if not hand_is_moving and stability_detector.is_stable():
    character = character_recognizer.predict(image)
    
# Nhận dạng dấu thanh khi tay chuyển động
elif hand_is_moving and not tone_collection:
    tone = tone_recognizer.predict(keypoints_sequence)
```

## 🔗 Kết hợp ký tự và dấu thanh

```python
# Thêm ký tự
text_processor.process_character("A")

# Áp dụng dấu thanh
text_processor.apply_tone_to_word("sac")  # A + sắc = Á

# Kết quả: "Á"
```

## 📊 Hiệu suất

- **FPS**: ~20-30 FPS trên CPU
- **Độ chính xác ký tự**: >95% (với tay ổn định)
- **Độ chính xác dấu thanh**: >85% (với chuyển động rõ ràng)
- **Độ trễ**: <100ms

## 🛠️ Yêu cầu mô hình

### Character Model:
- **File**: `trained_models/character_model_best.pt`
- **Kiến trúc**: ResNet50 với 26 output classes
- **Input**: 224x224x3 RGB image

### Tone Model:
- **File**: `trained_models/lstm_model_final.h5`
- **Label Encoder**: `trained_models/lstm_model_label_encoder.pkl`
- **Kiến trúc**: LSTM với 5 output classes
- **Input**: 30 frames x 63 features (21 landmarks x 3 coordinates)

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **Import errors**: Đảm bảo cấu trúc thư mục đúng
2. **Model not found**: Kiểm tra đường dẫn mô hình
3. **Camera issues**: Kiểm tra quyền truy cập camera
4. **Performance issues**: Giảm IMAGE_SIZE hoặc TONE_FRAMES_COUNT

## 📝 Lưu ý

- Module được thiết kế theo cấu trúc modular, dễ mở rộng
- Mỗi submodule có thể hoạt động độc lập
- Utils được chia sẻ giữa các submodule
- Configuration tập trung trong `utils/config.py`

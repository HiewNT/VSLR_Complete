# VSLR Complete - Vietnamese Sign Language Recognition

Hệ thống nhận dạng ngôn ngữ ký hiệu tiếng Việt hoàn chỉnh, bao gồm nhận dạng ký tự và dấu thanh.

## 🚀 Tính năng chính

- **Nhận dạng ký tự**: 26 ký tự cơ bản tiếng Việt (A-Z, Đ, Â, Ă, Ê, Ô, Ơ, Ư)
- **Nhận dạng dấu thanh**: 5 dấu thanh (huyền, sắc, hỏi, ngã, nặng)
- **Kết hợp thông minh**: Tự động kết hợp ký tự và dấu thanh thành chữ hoàn chỉnh
- **Giao diện hiện đại**: PyQt5 với thiết kế responsive
- **Xử lý real-time**: Nhận dạng trực tiếp từ camera
- **Lưu trữ văn bản**: Xuất kết quả ra file text

## 📁 Cấu trúc dự án

```
VSLR_Complete/
├── DataCollection/          # Thu thập dữ liệu
│   ├── character_collection/
│   ├── tone_collection/
│   └── utils/
├── ModelTraining/           # Huấn luyện mô hình
│   ├── character_training/
│   ├── tone_training/
│   └── utils/
├── Recognition/             # Module nhận dạng (MỚI)
│   ├── character_recognition/    # Nhận dạng ký tự
│   │   ├── __init__.py
│   │   └── recognize_characters.py
│   ├── tone_recognition/         # Nhận dạng dấu thanh
│   │   ├── __init__.py
│   │   └── recognize_tones.py
│   ├── text_processing/          # Xử lý văn bản
│   │   ├── __init__.py
│   │   └── process_text.py
│   ├── frame_processing/         # Xử lý frame chính
│   │   ├── __init__.py
│   │   └── process_frames.py
│   ├── utils/                    # Utilities và helpers
│   │   ├── __init__.py
│   │   ├── recognition_utils.py
│   │   ├── hand_tracking.py
│   │   ├── stability_detector.py
│   │   └── config.py
│   └── README.md
├── trained_models/          # Mô hình đã huấn luyện
├── app.py                  # Ứng dụng chính (PyQt5)
├── demo.py                 # Demo command-line
├── test_structure.py       # Test cấu trúc module
└── requirements.txt
```

## 🛠️ Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị mô hình

Đảm bảo có các file mô hình trong thư mục `trained_models/`:
- `character_model_best.pt` - Mô hình nhận dạng ký tự
- `lstm_model_final.h5` - Mô hình nhận dạng dấu thanh
- `lstm_model_label_encoder.pkl` - Label encoder cho dấu thanh

## 🎯 Cách sử dụng

### 1. Ứng dụng GUI (Khuyến nghị)

```bash
python app.py
```

**Giao diện chính:**
- **Camera panel**: Hiển thị video real-time với landmarks
- **Info panel**: Thông tin trạng thái, FPS, ký tự hiện tại, dấu thanh
- **Text display**: Văn bản được nhận dạng
- **Controls**: Bắt đầu/dừng, xóa, lưu file

### 2. Demo command-line

```bash
python demo.py
```

**Phím tắt:**
- `q`: Thoát
- `c`: Xóa text
- `s`: Lưu text ra file

### 3. Test cấu trúc module

```bash
python test_structure.py
```

**Kiểm tra:**
- Import các module
- Khởi tạo components
- Cấu trúc tổng thể

## 🧠 Cách hoạt động

### Logic nhận dạng:

1. **Phát hiện tay**: MediaPipe phát hiện landmarks của bàn tay
2. **Phân biệt trạng thái**:
   - **Tay tĩnh** → Nhận dạng ký tự (ResNet50)
   - **Tay chuyển động** → Nhận dạng dấu thanh (LSTM)
3. **Kết hợp**: TextProcessor kết hợp ký tự + dấu thanh = chữ hoàn chỉnh

### Ví dụ:

```
Ký tự: "A" + Dấu thanh: "sắc" = "Á"
Ký tự: "O" + Dấu thanh: "huyền" = "Ò"
```

### Ký tự đặc biệt:

- **Mu**: Tạo Â, Ê, Ô (A+Mu=Â, E+Mu=Ê, O+Mu=Ô)
- **Munguoc**: Tạo Ă (A+Munguoc=Ă)
- **Rau**: Tạo Ơ, Ư (O+Rau=Ơ, U+Rau=Ư)

## 📊 Hiệu suất

- **FPS**: ~20-30 FPS trên CPU
- **Độ chính xác ký tự**: >95% (với tay ổn định)
- **Độ chính xác dấu thanh**: >85% (với chuyển động rõ ràng)
- **Độ trễ**: <100ms

## ⚙️ Cấu hình

### Thresholds (có thể điều chỉnh trong `Recognition/config.py`):

```python
MIN_CONFIDENCE_THRESHOLD = 0.98    # Ngưỡng tin cậy ký tự
TONE_CONFIDENCE_THRESHOLD = 0.8    # Ngưỡng tin cậy dấu thanh
TONE_FRAMES_COUNT = 30             # Số frame cho dấu thanh
IMAGE_SIZE = 300                   # Kích thước ảnh
```

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **"Không tìm thấy mô hình"**
   - Kiểm tra file mô hình trong `trained_models/`
   - Đảm bảo đường dẫn đúng

2. **"Không thể mở camera"**
   - Kiểm tra camera có được sử dụng bởi ứng dụng khác
   - Thử thay đổi camera index (0, 1, 2...)

3. **Nhận dạng không chính xác**
   - Đảm bảo ánh sáng đủ
   - Giữ tay ổn định khi nhận dạng ký tự
   - Chuyển động rõ ràng khi nhận dạng dấu thanh

4. **Lỗi TensorFlow**
   - Cài đặt lại TensorFlow: `pip install tensorflow==2.13.0`
   - Hoặc sử dụng CPU-only: `pip install tensorflow-cpu==2.13.0`

## 📝 API Reference

### Recognition Module

```python
from Recognition import (
    CharacterRecognizer, ToneRecognizer,
    TextProcessor, FrameProcessor
)
from Recognition.utils import HandDetector, StabilityDetector

# Khởi tạo
detector = HandDetector(max_hands=1)
character_recognizer = CharacterRecognizer()
tone_recognizer = ToneRecognizer()
stability_detector = StabilityDetector()
text_processor = TextProcessor()

# Xử lý frame
frame_processor = FrameProcessor(
    detector=detector,
    character_recognizer=character_recognizer,
    tone_recognizer=tone_recognizer,
    stability_detector=stability_detector,
    text_processor=text_processor
)

# Sử dụng
processed_frame = frame_processor.process_frame(frame)
display_text = text_processor.get_display_text()
```

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

## 👥 Tác giả

- **Phát triển**: VSLR Team
- **Mô hình**: ResNet50 + LSTM
- **Framework**: PyTorch + TensorFlow + MediaPipe

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng tạo issue trên GitHub hoặc liên hệ team phát triển.

---

**Lưu ý**: Hệ thống được thiết kế cho ngôn ngữ ký hiệu tiếng Việt. Để sử dụng với ngôn ngữ khác, cần huấn luyện lại mô hình với dữ liệu tương ứng.

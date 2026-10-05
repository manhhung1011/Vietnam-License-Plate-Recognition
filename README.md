# 🇻🇳 Vietnam License Plate Recognition

## 📌 Giới thiệu

Hệ thống nhận dạng biển số xe Việt Nam được xây dựng bằng **Python**, sử dụng **YOLOv8** để phát hiện biển số và **CRNN + CTC** để nhận dạng ký tự. **OpenCV** được sử dụng để xử lý ảnh/video và hiển thị kết quả, mô hình được huấn luyện bằng **PyTorch**.

## 🔄 Pipeline hệ thống

```text
Ảnh / Video đầu vào
        ↓
      YOLOv8
  Phát hiện biển số
        ↓
    Crop biển số
        ↓
   Tiền xử lý ảnh
        ↓
     CRNN + CTC
   Nhận dạng ký tự
        ↓
    Kết quả biển số
```

## ⚙️ Clone & Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/manhhung1011/Vietnam-License-Plate-Recognition.git
cd Vietnam-License-Plate-Recognition
```

### 2. Tạo môi trường ảo

```bash
python -m venv .venv
```

### 3. Kích hoạt môi trường ảo

Trên Windows:

```bash
.venv\Scripts\activate
```

### 4. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## ▶️ Chạy thử hệ thống

Sau khi cài đặt đầy đủ thư viện và chuẩn bị các model trong thư mục `models/`, chạy:

```bash
python main.py
```

Hệ thống sẽ thực hiện:

```text
Input Image / Video
        ↓
YOLOv8 Detection
        ↓
License Plate Crop
        ↓
CRNN + CTC OCR
        ↓
License Plate Result
```

## 🧠 Công nghệ sử dụng

- **Python**
- **YOLOv8** – phát hiện biển số
- **CRNN + CTC** – nhận dạng ký tự
- **PyTorch** – huấn luyện và inference
- **OpenCV** – xử lý ảnh/video
- **Roboflow** – gán nhãn và quản lý dữ liệu

## 📊 Kết quả

Mô hình OCR đạt khoảng **94.37% Character Accuracy** trong quá trình đánh giá.

Hệ thống hướng tới khả năng nhận dạng biển số xe Việt Nam từ hình ảnh và video thông qua pipeline **Object Detection → Image Processing → OCR**.

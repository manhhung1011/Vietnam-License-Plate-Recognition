# 🇻🇳 Vietnam License Plate Recognition

## 📌 Giới thiệu

Hệ thống nhận dạng biển số xe Việt Nam được xây dựng bằng **Python**, sử dụng **YOLOv8** cho bài toán phát hiện biển số và **CRNN + CTC** cho nhận dạng ký tự. **OpenCV** được sử dụng để xử lý ảnh/video và hiển thị kết quả, mô hình được huấn luyện bằng **PyTorch**.

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
⚙️ Clone & Cài đặt
1. Clone repository
git clone https://github.com/manhhung1011/Vietnam-License-Plate-Recognition.git
cd Vietnam-License-Plate-Recognition
2. Tạo môi trường ảo
python -m venv .venv
3. Kích hoạt môi trường ảo

Trên Windows:

.venv\Scripts\activate
4. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
▶️ Chạy thử hệ thống

Sau khi cài đặt đầy đủ thư viện và chuẩn bị các model trong thư mục models/, chạy:

python main.py

Pipeline khi chạy:

Input Image / Video
        ↓
YOLOv8 Detection
        ↓
License Plate Crop
        ↓
CRNN + CTC OCR
        ↓
License Plate Result

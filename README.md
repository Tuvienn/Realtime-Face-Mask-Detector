# 😷 Real-time Face Mask Detection System

> **Production-Ready Deep Learning Pipeline** cho bài toán nhận diện đeo khẩu trang thời gian thực qua webcam, xây dựng bằng **TensorFlow / Keras**, **OpenCV DNN**, và **Clean Modular Architecture**.

---

## 🌟 Tính Năng Nổi Bật (Key Features)

- **Phát hiện khuôn mặt chính xác:** Sử dụng mô hình **Caffe SSD (res10_300x300)** của OpenCV cho khả năng detect khuôn mặt vượt trội so với Haar Cascade.
- **Hỗ trợ Transfer Learning:** Tích hợp mô hình **MobileNetV2** (pretrained trên ImageNet) giúp tăng độ chính xác và giảm latency so với Custom CNN.
- **Cấu trúc mã nguồn chuẩn (Clean Architecture):** Đóng gói module `src/` với mô hình Factory Pattern, phân tách rõ ràng trách nhiệm giữa Data, Model và Inference Engine.
- **Inference thời gian thực tối ưu:** Trực tiếp tính toán tensor qua webcam, hiển thị khung nhận diện (Mask/No Mask) kèm confidence score và FPS.
- **Quy trình kiểm thử & Đánh giá toàn diện:** Tích hợp Evaluation Pipeline tự động vẽ **Confusion Matrix**, **F1-Score**, **Accuracy Report** và hỗ trợ **TensorBoard / EarlyStopping**.
- **Containerized với Docker:** Đóng gói ứng dụng với `Dockerfile` và `docker-compose.yml` sẵn sàng cho triển khai cloud.

---

## 📁 Cấu Trúc Thư Mục Repository (Project Structure)

```
face-mask-detector/
├── src/                        # Core Python Package (Modular Logic)
│   ├── __init__.py
│   ├── model.py                # Model Factory (Custom CNN & MobileNetV2)
│   ├── dataset.py              # Data Generator & Preprocessing
│   └── detector.py             # Pipeline nhận diện khuôn mặt + khẩu trang
│
├── configs/                    # Quản lý cấu hình tập trung
│   └── config.yaml             # Hyperparameters, Paths & Labels
│
├── models/                     # Chứa các file mô hình pretrained
│   ├── deploy.prototxt         # Caffe Face Detector Architecture
│   └── res10_300x300_ssd_...   # Caffe Face Detector Weights
│
├── scripts/                    # Các script tiện ích
│   ├── split_dataset.py        # Tách tập train / val / test chuẩn tỉ lệ
│   ├── evaluate.py             # Đánh giá model & xuất báo cáo/Confusion Matrix
│   └── check_env.py            # Kiểm tra môi trường GPU / TensorFlow
│
├── tests/                      # Unit Tests tự động
│   └── test_model.py           # Unit tests kiểm tra shape & loss logic
│
├── data/                       # Dataset tập dữ liệu
│   ├── train/                  # Dataset tập huấn luyện
│   │   ├── with_mask/
│   │   └── without_mask/
│   └── test/                   # Dataset tập đánh giá/kiểm thử
│       ├── with_mask/
│       └── without_mask/
│
├── train.py                    # Entrypoint huấn luyện mô hình (CLI)
├── test.py                     # Entrypoint nhận diện thời gian thực qua Webcam
├── requirements.txt            # Quản lý dependencies
├── Dockerfile                  # Cấu hình Docker Container
├── docker-compose.yml          # Container Orchestration
├── working_rule.md             # Quy tắc làm việc & tiêu chuẩn code
└── README.md                   # Tài liệu hướng dẫn sử dụng
```

---

## 🛠️ Hướng Dẫn Cài Đặt (Installation)

### 1. Khởi tạo môi trường ảo (Virtual Environment)

```bash
# Clone repository
git clone <repository_url>
cd face-mask-detector

# Tạo và kích hoạt môi trường ảo Python (Python 3.9 - 3.11 khuyến nghị)
python3 -m venv .venv
source .venv/bin/activate    # Trên macOS / Linux
# .venv\Scripts\activate     # Trên Windows
```

### 2. Cài đặt các thư viện phụ thuộc

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Sử Dụng (Usage Guide)

### 1. Tách Dataset chuẩn (Train / Val / Test Split)

Sử dụng script để chia tự động tỉ lệ 70% Train, 15% Validation, 15% Test:
```bash
python scripts/split_dataset.py --source ./train --output ./data --train 0.7 --val 0.15 --test 0.15
```

### 2. Huấn Luyện Mô Hình (Training)

Bạn có thể lựa chọn 1 trong 2 kiến trúc qua tham số `--arch`:

- **Option A: MobileNetV2 (Transfer Learning - Khuyên dùng)**
  ```bash
  python train.py --arch mobilenetv2 --epochs 30 --batch-size 32
  ```

- **Option B: Custom CNN Baseline**
  ```bash
  python train.py --arch custom_cnn --epochs 30 --batch-size 32
  ```

*Quá trình training sẽ tự động:*
- Theo dõi `val_loss` và lưu mô hình tốt nhất vào `best_model.keras`.
- Tự động dừng sớm (**EarlyStopping**) sau 5 epoch nếu `val_loss` không giảm thêm.
- Ghi log theo dõi trực quan vào thư mục `./logs/fit/`.

### 3. Theo dõi Training với TensorBoard

```bash
tensorboard --logdir ./logs/fit
```
Mở trình duyệt truy cập `http://localhost:6006` để xem biểu đồ Loss & Accuracy theo epoch.

### 4. Đánh Giá Mô Hình (Evaluation)

Chạy script đánh giá để xuất báo cáo chi tiết và biểu đồ Confusion Matrix:
```bash
python scripts/evaluate.py --model ./best_model.keras --test-dir ./test --output ./outputs
```
*Kết quả sẽ được lưu tại thư mục `./outputs/`:*
- `eval_report.txt`: Chi tiết Precision, Recall, F1-Score từng lớp.
- `confusion_matrix.png`: Biểu đồ ma trận nhầm lẫn.

### 5. Chạy Nhận Diện Thời Gian Thực (Real-time Webcam Inference)

```bash
python test.py
```
- Phím **`Q`** hoặc **`ESC`**: Thoát chương trình.
- Hiển thị FPS thời gian thực cùng ô vuông xanh (Đeo khẩu trang) / đỏ (Không đeo khẩu trang).

---

## 🧪 Chạy Kiểm Thử Tự Động (Unit Testing)

Chạy bộ test tự động để đảm bảo các module hoạt động chính xác:
```bash
PYTHONPATH=. python -m unittest discover -s tests
```

---

## 🐳 Triển Khai Với Docker (Docker Deployment)

Xây dựng và khởi chạy container:
```bash
# Biên dịch image Docker
docker build -t face-mask-detector .

# Hoặc khởi chạy qua Docker Compose
docker compose up
```

---

## 📐 Kiến Trúc Kỹ Thuật (Technical Specifications)

- **Classification Loss Function:** `categorical_crossentropy` kết hợp với Softmax Output (2 classes).
- **Data Augmentation:** Random Rotation (40°), Width/Height Shift (0.2), Shear (0.2), Zoom (0.2), Horizontal Flip.
- **Inference Optimization:** Sử dụng trực tiếp `model(batch, training=False)` loại bỏ overhead của `model.predict()` trên từng frame ảnh.

---

<div align="center">

<h3>📬 Liên hệ</h3>

| | |
|---|---|
| 👩🏻‍💻 **Tác giả** | Nguyễn Thị Tú Viên |
| 🏫 **Trường** | Ho Chi Minh University of Transport|
| 📧 **Email** | [nguyenthituvien2005@gmail.com](mailto:nguyenthituvien2005@gmail.com) |
| 📞 **SĐT** | [0335 637 198](tel:+84335637198) |

<br>
<i>💖 Cảm ơn mọi người đã quan tâm và theo dõi dự án này. Chúc các bạn một ngày tốt lành!</i>


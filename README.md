# 🍌 Banana Disease Detection & Diagnosis Demo

Hệ thống AI nhận diện và chẩn đoán bệnh trên quả chuối sử dụng kiến trúc mạng nơ-ron **DenseNet-121** (huấn luyện tại Kaggle Notebook: [nguynthanhthit/banana-fruit-disease-efficientnet-b0](https://www.kaggle.com/code/nguynthanhthit/banana-fruit-disease-efficientnet-b0) - **Version 24**).

## 🎯 Hiệu Năng Mô Hình (Version 24)
- **Độ chính xác kiểm thử (Test Accuracy)**: **66.67%**
- **Chỉ số Macro F1-Score**: **62.36%**
- **Độ phân giải đầu vào**: 384x384 pixels
- **Phần cứng suy luận**: Hỗ trợ tự động GPU CUDA hoặc CPU

## 🍌 4 Phân Loại Bệnh Quả Chuối
1. **Bệnh rỉ sắt / đốm gỉ quả chuối** (`black_pitting_banana_rust`)
2. **Bệnh thối cổ / thối cuống nải chuối** (`crown_rot`)
3. **Bệnh thán thư / nấm đốm vỏ chuối** (`fungal_disease`)
4. **Chuối khỏe mạnh** (`healthy_banana`)

## 🚀 Hướng Dẫn Chạy Cục Bộ (Local)
```bash
cd D:\Cty_Synexan\Banana_detect\Demo
pip install -r requirements.txt
python app.py
```
Truy cập trình duyệt tại địa chỉ: `http://localhost:7860`

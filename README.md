# 🌟 Dự báo tải mạng quy mô quốc gia

## 📌 Mô tả đề tài

Dự án sử dụng mô hình LSTM, XGBoost, TFT rồi so sánh giữa các mô hình để dự báo hoạt động viễn thông như SMS, cuộc gọi và lưu lượng Internet trong từng "Square ID" tại các khung giờ trong tuần, dựa trên dữ liệu thời gian thực.

## 📂 Cấu trúc thư mục chung: `TIMESERIES_NHOM5/`
- data :  Dữ liệu tải mạng
- DEMO :  demo ứng dụng
- CodeModels :  Xây dựng và huấn luyện mô hình
- models  :  Thư mục chứa model đã huấn luyện (.pth)
- README.md : Tài liệu mô tả dự án


## 🧱 Cấu trúc thư mục DEMO `demo/`

- `app.py`: Chương trình chính để chạy dự báo  
- `lstm_model.py`: Định nghĩa mô hình LSTM  
- `predictor.py`: Hàm dự đoán  
- `train.py`: Huấn luyện mô hình cho các Square ID  
- `filtered_data.parquet`: Dữ liệu đầu vào đã lọc  
- `batch.py`: Lấy những square id đã huấn luyện và lọc ra  
- `README.md`: Tài liệu mô tả dự án

---

### 🌐 Giao diện ứng dụng triển khai:
👉 [Ứng dụng Hugging Face](https://huggingface.co/spaces/Kwann5002/Timeseries)



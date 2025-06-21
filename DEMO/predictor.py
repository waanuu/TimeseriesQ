import torch
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta
from lstm_model import LSTMModel

def predict(square_id, hour, day_of_week, parquet_file='DEMO/filtered_data.parquet', model_dir='models', input_len=432, pred_len=144):
    features = ['SMS-in activity', 'SMS-out activity', 'Call-in activity', 'Call-out activity', 'Internet traffic activity']
    days = ['thứ hai', 'thứ ba', 'thứ tư', 'thứ năm', 'thứ sáu', 'thứ bảy', 'chủ nhật']

    try:
        day_idx = days.index(day_of_week.lower())
    except ValueError:
        return f"Ngày không hợp lệ: {day_of_week}"

    if not (0 <= hour <= 23):
        return f"Giờ không hợp lệ: {hour}"

    model_path = f"{model_dir}/lstm_square_{square_id}.pth"
    scaler_path = f"{model_dir}/scaler_square_{square_id}.pkl"

    try:
        model = LSTMModel(input_dim=len(features), pred_len=pred_len)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        return f"Không tìm thấy model/scaler cho Square ID {square_id}: {e}"

    df = pd.read_parquet(parquet_file)
    df['Time Interval'] = pd.to_datetime(df['Time Interval'], unit='ms')
    group = df[df['Square id'] == square_id].copy()
    group['Hour'] = group['Time Interval'].dt.hour
    group['Day'] = group['Time Interval'].dt.dayofweek
    group = group.sort_values('Time Interval')

    match = group[(group['Day'] == day_idx) & (group['Hour'] == hour)]
    if match.empty:
        return f"Không có dữ liệu cho {day_of_week} lúc {hour}:00"

    latest_time = match['Time Interval'].max()
    input_data = group[group['Time Interval'] <= latest_time].tail(input_len)

    if len(input_data) < input_len:
        return f"Không đủ dữ liệu: {len(input_data)} (cần {input_len})"

    data = scaler.transform(input_data[features].values)
    x = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        y_pred = model(x).numpy().squeeze()
    y_pred = scaler.inverse_transform(y_pred)
    y_pred = np.clip(np.round(y_pred, 2), 0, None)

    pred_times = [latest_time + timedelta(minutes=10 * i) for i in range(1, pred_len + 1)]
    result = pd.DataFrame(y_pred, columns=features)
    result = result.applymap(lambda x: f"{max(0, round(x, 2)):.2f}")
    result.insert(0, "Thời gian", [t.strftime("%H:%M") for t in pred_times])
    return result
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Kiểm tra phiên bản PyTorch
print(f"Phiên bản PyTorch: {torch.__version__}")

# Lớp TimeSeriesDataset để tạo các chuỗi dữ liệu thời gian
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_len=432, pred_len=144):
        self.input_len = input_len
        self.pred_len = pred_len
        self.sequences = self.create_sequences(data)
    
    def create_sequences(self, data):
        sequences = []
        total_len = self.input_len + self.pred_len
        for i in range(len(data) - total_len + 1):
            input_seq = data[i:i + self.input_len]
            target_seq = data[i + self.input_len:i + total_len]
            sequences.append((input_seq, target_seq))
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Lớp mô hình PatchTST
class PatchTST(nn.Module):
    def __init__(self, input_dim, patch_len=16, stride=8, d_model=64, num_layers=3, pred_len=144):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.linear = nn.Linear(patch_len * input_dim, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=num_layers
        )
        self.head = nn.Linear(d_model, input_dim * pred_len)
    
    def forward(self, x):  # x: [B, T, C]
        B, T, C = x.shape
        patches = x.unfold(1, self.patch_len, self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(B, -1, self.patch_len * C)
        x = self.linear(patches)
        x = self.encoder(x)
        x = self.head(x[:, -1])
        return x.reshape(B, self.pred_len, C)

# Hàm trực quan hóa hiệu suất mô hình
def visualize_model_performance(square_id, batch_folder, model_dir="models", input_len=432, pred_len=144, output_dir="visualizations"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị được sử dụng cho trực quan hóa: {device}")
    features = ['SMS-in activity', 'SMS-out activity', 'Call-in activity', 'Call-out activity', 'Internet traffic activity']
    
    model_path = os.path.join(model_dir, f"patchtst_square_{square_id}.pth")
    scaler_path = os.path.join(model_dir, f"scaler_square_{square_id}.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return f"Không tìm thấy mô hình hoặc scaler cho Square id {square_id}"
    
    model = PatchTST(input_dim=len(features), pred_len=pred_len).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    group = []
    for batch_file in sorted(os.listdir(batch_folder)):
        if batch_file.endswith(".parquet"):
            batch_path = os.path.join(batch_folder, batch_file)
            df = pd.read_parquet(batch_path)
            df['Time Interval'] = pd.to_datetime(df['Time Interval'], unit='ms')
            group.append(df[df['Square id'] == square_id])
    
    if not group:
        return f"Không tìm thấy Square id {square_id} trong bất kỳ batch nào"
    
    group = pd.concat(group, ignore_index=True)
    group = group.sort_values('Time Interval')
    
    data = group[features].values
    data_scaled = scaler.transform(data)
    
    dataset = TimeSeriesDataset(data_scaled, input_len, pred_len)
    if len(dataset) < 10:
        return f"Không đủ dữ liệu cho Square id {square_id}: chỉ có {len(dataset)} chuỗi"
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            y_pred_np = y_pred.cpu().numpy()
            y_true_np = y.cpu().numpy()
            y_pred_np = scaler.inverse_transform(y_pred_np.reshape(-1, len(features))).reshape(y_pred_np.shape)
            y_true_np = scaler.inverse_transform(y_true_np.reshape(-1, len(features))).reshape(y_true_np.shape)
            y_true_all.append(y_true_np)
            y_pred_all.append(y_pred_np)
    
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    
    mse_scores = []
    mae_scores = []
    for i, feature in enumerate(features):
        mse = mean_squared_error(y_true_all[:, :, i].flatten(), y_pred_all[:, :, i].flatten())
        mae = mean_absolute_error(y_true_all[:, :, i].flatten(), y_pred_all[:, :, i].flatten())
        mse_scores.append(mse)
        mae_scores.append(mae)
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features):
        plt.subplot(3, 2, i + 1)
        plt.plot(y_true_all[:, :, i].flatten()[:288], label='Thực tế', color='blue')
        plt.plot(y_pred_all[:, :, i].flatten()[:288], label='Dự đoán', color='orange', linestyle='--')
        plt.title(f'{feature}\nMSE: {mse_scores[i]:.4f}, MAE: {mae_scores[i]:.4f}')
        plt.xlabel('Mẫu thời gian (10 phút/mẫu)')
        plt.ylabel('Giá trị')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"performance_square_{square_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    plt.close()
    
    return f"Biểu đồ hiệu suất đã được lưu vào {plot_path}\n" + "\n".join(
        f"{feature}: MSE={mse_scores[i]:.4f}, MAE={mae_scores[i]:.4f}" for i, feature in enumerate(features)
    )

def main():
    batch_folder = "/kaggle/input/temporal-fusion-transformer"
    model_dir = "models"
    
    num_models = len([f for f in os.listdir(model_dir) if f.endswith('.pth')])
    print(f"Số lượng mô hình hiện có: {num_models}")
    
    while True:
        print("\nNhập Square id để trực quan hóa hiệu suất mô hình (hoặc 'thoát' để kết thúc):")
        square_id = input("Square id: ")
        
        if square_id.lower() == 'thoát':
            break
        
        try:
            square_id = int(square_id)
        except ValueError:
            print("Square id phải là số nguyên")
            continue
        
        result = visualize_model_performance(square_id, batch_folder, model_dir)
        print(result)

if __name__ == "__main__":
    main()
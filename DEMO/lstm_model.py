import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, pred_len=144):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, input_dim * pred_len)
        self.pred_len = pred_len
        self.input_dim = input_dim

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # lấy bước thời gian cuối
        out = self.linear(out)
        return out.view(-1, self.pred_len, self.input_dim)
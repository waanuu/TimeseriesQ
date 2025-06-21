import os
import re
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')  # Để in Unicode ra terminal

# Hàm lấy danh sách các Square ID đã huấn luyện từ thư mục models/
def get_trained_square_ids(model_dir='models'):
    trained_ids = []
    for f in os.listdir(model_dir):
        match = re.match(r'lstm_square_(\d+)\.pth', f)
        if match:
            trained_ids.append(int(match.group(1)))
    return set(trained_ids)

# Lọc dữ liệu từ 21 batch chỉ giữ các square id đã huấn luyện
def filter_batches_by_square_ids(batch_folder='all_batches', output_file='DEMO/filtered_data.parquet', model_dir='models'):
    trained_square_ids = get_trained_square_ids(model_dir)
    print("Đã huấn luyện:", trained_square_ids)

    all_dfs = []
    for file in sorted(os.listdir(batch_folder)):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(batch_folder, file))
            filtered_df = df[df['Square id'].isin(trained_square_ids)]
            all_dfs.append(filtered_df)

    combined_df = pd.concat(all_dfs)
    combined_df.to_parquet(output_file)
    print(f"Đã lưu dữ liệu lọc tại: {output_file}")

# Gọi hàm chính
filter_batches_by_square_ids(batch_folder='E:/TIMESERIES_NHOM5/data', output_file='DEMO/filtered_data.parquet', model_dir='models')
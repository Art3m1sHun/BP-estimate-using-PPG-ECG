import wfdb
import pandas as pd
import numpy as np
import os
import scipy.io

# =================================================================
# === CẤU HÌNH HỆ THỐNG ===
# =================================================================

# 1. Đường dẫn thư mục chứa các file .dat và .hea
WFDB_DATA_DIR = r"D:\Python\data-ppg-ecg\MATLAB_preprocessing\data\data_MIMIC" # CẬP NHẬT
# 2. Đường dẫn thư mục chứa các file CSV
CSV_FOLDER_DIR = r"D:\Python\data-ppg-ecg\MATLAB_preprocessing\data\data_MIMIC\bidmc_csv" # CẬP NHẬT
# 3. Đường dẫn file .mat đầu ra
OUTPUT_MAT_PATH = r"D:\Python\data-ppg-ecg\MATLAB_preprocessing\data\combined_raw_data.mat"

# 4. Danh sách các record WFDB cần xử lý
# Ví dụ: bidmc01 đến bidmc35
RECORD_IDS = [f'bidmc{i:02d}' for i in range(1, 53)] 

# 5. Cấu hình tên kênh WFDB
WFDB_CHANNEL_NAMES = ['PLETH,', 'ABP,', 'II,'] 

# =================================================================
# === 1. XỬ LÝ VÀ GỘP DỮ LIỆU WFDB (.dat, .hea) ===
# =================================================================

import wfdb
import numpy as np
import os

# CẤU HÌNH CẦN CẬP NHẬT
WFDB_CHANNEL_NAMES = ['PLETH,', 'ABP,', 'II,'] 

def load_and_combine_wfdb_data(data_dir, record_ids, channel_names):
    """Đọc và gộp tín hiệu từ các record WFDB, chỉ xử lý record có đủ kênh."""
    
    all_ppg, all_abp, all_ecg = [], [], []
    
    for record_name in record_ids:
        print(f"Đang đọc record WFDB: {record_name}")
        try:
            # 1. Đọc header để kiểm tra kênh
            record = wfdb.rdrecord(os.path.join(data_dir, record_name), smooth_frames=True)
            sig_names = record.sig_name
            
            # 2. KIỂM TRA SỰ TỒN TẠI CỦA TẤT CẢ CÁC KÊNH BẮT BUỘC
            missing_channels = [name for name in channel_names if name not in sig_names]
            
            if missing_channels:
                print(f"  -> BỎ QUA: Thiếu các kênh: {', '.join(missing_channels)}. (Cần có ABP để tính nhãn).")
                continue # Bỏ qua record này
                
            data_raw = record.p_signal
            
            # 3. Trích xuất dữ liệu (Chỉ chạy khi tất cả các kênh đều có)
            indices = [sig_names.index(name) for name in channel_names]
            
            ppg_raw = data_raw[:, indices[channel_names.index('PLETH,')]]
            abp_raw = data_raw[:, indices[channel_names.index('ABP,')]]
            ecg_raw = data_raw[:, indices[channel_names.index('II,')]]
            
            # Gộp vào danh sách chung
            all_ppg.append(ppg_raw)
            all_abp.append(abp_raw)
            all_ecg.append(ecg_raw)
            print(f"  -> Thành công: Đã thêm {len(ppg_raw)} mẫu.")
            
        except Exception as e:
            print(f"  -> Lỗi không xác định khi xử lý WFDB {record_name}: {e}. Bỏ qua.")
            
    # Gộp tất cả các mảng NumPy lại
    if all_ppg:
        combined_ppg = np.concatenate(all_ppg, axis=0).astype(np.float32)
        combined_abp = np.concatenate(all_abp, axis=0).astype(np.float32)
        combined_ecg = np.concatenate(all_ecg, axis=0).astype(np.float32)
        print(f"✅ Hoàn tất WFDB: Tổng {len(combined_ppg)} mẫu từ {len(all_ppg)} records hợp lệ.")
        return combined_ppg, combined_abp, combined_ecg
    else:
        print("❌ KHÔNG CÓ RECORD WFDB NÀO CÓ ĐỦ CÁC KÊNH BẮT BUỘC.")
        return None, None, None

# Tích hợp hàm này vào script chính của bạn.

# =================================================================
# === 2. XỬ LÝ VÀ GỘP DỮ LIỆU CSV (Tín hiệu bổ sung) ===
# =================================================================

# SỬA HÀM XỬ LÝ CSV
import pandas as pd

# CẤU HÌNH NHÃN CSV CẦN TÌM
CSV_LABEL_COLUMNS = ['PLETH,', 'ABP,', 'II,']  # Giả định các cột này tồn tại trong Numerics.csv

def load_csv_labels(folder_dir, record_ids, label_cols):
    """Đọc và gộp các cột nhãn (HR, SpO2, vv.) từ file _Numerics.csv."""
    
    all_labels_combined = {col: [] for col in label_cols}
    
    for record_id in record_ids:
        # Tên file Numerics/Signals tương ứng với record WFDB
        # GIẢ ĐỊNH: Nhãn nằm trong file Numerics.csv
        filename = f'bidmc_{record_id}_Numerics.csv' 
        filepath = os.path.join(folder_dir, filename)
        
        if not os.path.exists(filepath):
            # Nếu file không tồn tại, tạo mảng rỗng tương ứng (hoặc bạn có thể bỏ qua record đó)
            print(f"  -> Bỏ qua nhãn CSV: File {filename} không tồn tại.")
            # Vì file WFDB có thể vẫn được xử lý, chúng ta sẽ bỏ qua bước này 
            # để tránh làm rối code gộp nếu bạn muốn xử lý chúng riêng.
            continue
            
        try:
            df = pd.read_csv(filepath) 
            
            # Kiểm tra và trích xuất các cột nhãn
            for col in label_cols:
                if col in df.columns:
                    all_labels_combined[col].append(df[col].values)
                else:
                    print(f"  -> Cảnh báo: Cột nhãn '{col}' không có trong {filename}.")
                    # Đưa vào mảng NaN để duy trì độ dài nếu cần
                    # all_labels_combined[col].append(np.full_like(df.iloc[:, 0].values, np.nan)) 
                    
        except Exception as e:
            print(f"  -> Lỗi khi đọc nhãn CSV {filename}: {e}.")
            
    # Gộp tất cả các mảng NumPy lại
    final_labels = {}
    for col, data_list in all_labels_combined.items():
        if data_list:
             final_labels[f'{col}_combined'] = np.concatenate(data_list, axis=0).astype(np.float32)
             
    print(f"✅ Hoàn tất CSV: Đã gộp các nhãn: {list(final_labels.keys())}")
    return final_labels

# =================================================================
# === 3. CHẠY CHÍNH VÀ LƯU FILE .MAT ===
# =================================================================

if __name__ == '__main__':
    
    # 1. Đọc và gộp dữ liệu WFDB (PPG, ABP, ECG)
    combined_ppg_wfdb, combined_abp_wfdb, combined_ecg_wfdb = load_and_combine_wfdb_data(
        WFDB_DATA_DIR, RECORD_IDS, WFDB_CHANNEL_NAMES
    )
    
    # 2. Đọc và gộp NHÃN CSV (HR, SpO2, v.v.)
    # LƯU Ý: Hàm này trả về một dictionary các mảng nhãn
    csv_labels_dict = load_csv_labels(CSV_FOLDER_DIR, RECORD_IDS, CSV_LABEL_COLUMNS)

    # 3. Kiểm tra tính nhất quán và lưu file .MAT
    
    if combined_ppg_wfdb is not None:
        
        # Bắt đầu với các tín hiệu WFDB đã gộp
        mat_data = {
            'PPG': combined_ppg_wfdb,
            'ABP': combined_abp_wfdb,
            'ECG': combined_ecg_wfdb,
        }
        
        # Thêm các nhãn CSV đã gộp vào dictionary
        mat_data.update(csv_labels_dict)

        # Lưu vào file .MAT
        scipy.io.savemat(OUTPUT_MAT_PATH, mat_data)
        
        print("-" * 50)
        print(f"✅ ĐÃ LƯU THÀNH CÔNG VÀO FILE .MAT: {OUTPUT_MAT_PATH}")
        print(f"Các biến được lưu: {list(mat_data.keys())}")

    else:
        print("❌ KHÔNG CÓ DỮ LIỆU TÍN HIỆU WFDB HỢP LỆ. KHÔNG LƯU FILE .MAT.")
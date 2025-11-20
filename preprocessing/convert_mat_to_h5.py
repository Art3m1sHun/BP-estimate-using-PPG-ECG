import h5py
import os
import scipy.io  # Cần import scipy.io để đọc file .mat cũ
import numpy as np


mat_dir = r"D:\Python\data-ppg-ecg\MATLAB_preprocessing\data"
mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]

for mat_file in mat_files:
    mat_path = os.path.join(mat_dir, mat_file)
    h5_path = mat_path.replace('.mat', '.h5')
    print(f"Chuyển {mat_file} -> {os.path.basename(h5_path)} ...")
    
    try:
        # 1. ĐỌC FILE .MAT DÙNG scipy.io
        mat_contents = scipy.io.loadmat(mat_path)
        
        # 2. GHI NỘI DUNG VÀO FILE .H5 DÙNG h5py
        with h5py.File(h5_path, 'w') as dst:
            for key, value in mat_contents.items():
                # Bỏ qua các metadata của MATLAB (thường bắt đầu bằng '__')
                if key.startswith('__'):
                    continue
                
                # Kiểm tra và xử lý dữ liệu (chuyển ma trận 2D sang vector 1D nếu cần)
                if isinstance(value, np.ndarray):
                    # Nếu là mảng 2D (Nx1 hoặc 1xN), chuyển thành 1D (vector) để lưu trong H5 nếu muốn
                    if value.ndim == 2 and (value.shape[0] == 1 or value.shape[1] == 1):
                        value = value.flatten()
                        
                    # Lưu dataset vào file H5
                    dst.create_dataset(key, data=value)
                
        print(f"Đã lưu {h5_path}")
        
    except Exception as e:
        print(f"Lỗi chuyển {mat_file}: {e}")
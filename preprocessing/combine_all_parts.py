import h5py
import numpy as np
import os

# Gộp toàn bộ các file .h5 (các part) thành một file lớn duy nhất (N, 3)
mat_dir = r"D:\Python\data-ppg-ecg\cuff+less+blood+pressure+estimation"
h5_files = [os.path.join(mat_dir, f) for f in os.listdir(mat_dir) if f.endswith('.h5')]
all_data = []
for h5f in h5_files:
    with h5py.File(h5f, 'r') as f:
        # Tìm dataset đầu tiên có shape (N, 3)
        data = None
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                arr = obj[:]
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    data = arr[:, :3]
                    break
            elif isinstance(obj, h5py.Group):
                for subkey in obj.keys():
                    subobj = obj[subkey]
                    if isinstance(subobj, h5py.Dataset):
                        arr = subobj[:]
                        if arr.ndim == 2 and arr.shape[1] >= 3:
                            data = arr[:, :3]
                            break
        if data is not None:
            all_data.append(data)
            print(f"{os.path.basename(h5f)}: {data.shape}")
        else:
            print(f"{os.path.basename(h5f)}: Không tìm thấy dataset phù hợp!")
if not all_data:
    raise RuntimeError("Không có file nào hợp lệ để gộp!")
all_data = np.vstack(all_data)
print(f"Tổng shape sau gộp: {all_data.shape}")

# Lưu file gộp
out_path = r"D:\Python\data-ppg-ecg\cuff+less+blood+pressure+estimation/data_Test/ALL_PARTS_COMBINED.h5"
with h5py.File(out_path, 'w') as f:
    f.create_dataset('data', data=all_data)
print(f"Đã lưu file gộp: {out_path}")

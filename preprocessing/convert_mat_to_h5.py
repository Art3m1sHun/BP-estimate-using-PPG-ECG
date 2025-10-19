

import h5py
import os

def copy_h5_group(src, dst):
    for key in src:
        item = src[key]
        if isinstance(item, h5py.Dataset):
            dst.create_dataset(key, data=item[()])
        elif isinstance(item, h5py.Group):
            grp = dst.create_group(key)
            copy_h5_group(item, grp)

mat_dir = r"D:\Python\data-ppg-ecg\cuff+less+blood+pressure+estimation"
mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]

for mat_file in mat_files:
    mat_path = os.path.join(mat_dir, mat_file)
    h5_path = mat_path.replace('.mat', '.h5')
    print(f"Chuyển {mat_file} -> {os.path.basename(h5_path)} ...")
    try:
        with h5py.File(mat_path, 'r') as src:
            with h5py.File(h5_path, 'w') as dst:
                copy_h5_group(src, dst)
        print(f"Đã lưu {h5_path}")
    except Exception as e:
        print(f"Lỗi chuyển {mat_file}: {e}")

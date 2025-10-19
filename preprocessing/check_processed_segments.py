import h5py
import numpy as np
import matplotlib.pyplot as plt

# Đường dẫn file đã xử lý
file_path = r"D:\Python\data-ppg-ecg\cuff+less+blood+pressure+estimation\filtered_segments.h5"

with h5py.File(file_path, 'r') as f:
    ppg_segments = f['ppg_segments'][:]
    ecg_segments = f['ecg_segments'][:]
    labels = f['labels'][:]

print(f"Số segment: {ppg_segments.shape[0]}")
print(f"Shape PPG: {ppg_segments.shape}, ECG: {ecg_segments.shape}, Labels: {labels.shape}")
print("5 nhãn đầu tiên:")
print(labels[:5])

# Vẽ thử 2 segment đầu tiên
fs = 125
for i in range(5):
    plt.figure(figsize=(12,4))
    plt.subplot(2,1,1)
    plt.plot(np.arange(ppg_segments.shape[1])/fs, ppg_segments[i])
    plt.title(f"PPG Segment {i+1} - SBP: {labels[i,0]:.1f}, DBP: {labels[i,1]:.1f}, HR: {labels[i,2]:.1f}")
    plt.xlabel("Time (s)")
    plt.ylabel("PPG")
    plt.subplot(2,1,2)
    plt.plot(np.arange(ecg_segments.shape[1])/fs, ecg_segments[i])
    plt.title(f"ECG Segment {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG")
    plt.tight_layout()
    plt.show()
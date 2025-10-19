import h5py
import numpy as np
import scipy.signal as signal
import pywt
import os

# === Đường dẫn file gộp và file đầu ra ===
input_path = r"D:\Python\data-ppg-ecg\cuff+less+blood+pressure+estimation\ALL_PARTS_COMBINED.h5"
output_path = r"D:\Python\data-ppg-ecg\cuff+less+blood+pressure+estimation\filtered_segments.h5"

# Đọc dữ liệu
with h5py.File(input_path, 'r') as f:
    print("📂 Dataset trong file:")
    f.visit(print)
    data = f['data'][:]  # Đảm bảo trong file có dataset 'data'
    ppg_raw = data[:, 0]
    abp_raw = data[:, 1]
    ecg_raw = data[:, 2]

fs = 125

# --- Tiền xử lý PPG ---
sos_ppg = signal.cheby2(4, 20, [0.5, 10], btype='bandpass', fs=fs, output='sos')
ppg_f = signal.sosfiltfilt(sos_ppg, ppg_raw)
x = np.arange(len(ppg_f))
trend = np.polyval(np.polyfit(x, ppg_f, 3), x)
ppg_f = ppg_f - trend

# --- Tiền xử lý ECG ---
sos_ecg = signal.butter(8, 0.1, btype='highpass', fs=fs, output='sos')
ecg_hp = signal.sosfiltfilt(sos_ecg, ecg_raw)
coeffs = pywt.wavedec(ecg_hp, 'db6', level=3, mode='symmetric')
sigma = np.median(np.abs(coeffs[-1]))/0.6745
thr = sigma*np.sqrt(2*np.log(len(ecg_hp)))
coeffs[1:] = [pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]]
ecg_f = pywt.waverec(coeffs, 'db6', mode='symmetric')[:len(ecg_hp)]

# --- Phân đoạn cửa sổ trượt ---
win = 8*fs
step = 2*fs
idx = [(s, s+win) for s in range(0, len(ecg_f)-win+1, step)]


# --- Tạo các đoạn tín hiệu cho PPG, ECG, ABP ---
ppg_segments = [ppg_f[s:e] for s, e in idx]
ecg_segments = [ecg_f[s:e] for s, e in idx]
abp_segments = [abp_raw[s:e] for s, e in idx]

# --- Bước 5: Tính nhãn từ ABP (SBP, DBP, HR) ---
labels = []
for abp_seg in abp_segments:
    # Tìm đỉnh tâm thu (SBP)
    peaks, _ = signal.find_peaks(abp_seg, distance=int(0.27*fs), prominence=np.std(abp_seg)*0.6)
    if len(peaks) < 2:
        labels.append([np.nan, np.nan, np.nan])
        continue
    sbps = abp_seg[peaks]
    dbps = []
    for i in range(len(peaks)-1):
        a, b = peaks[i], peaks[i+1]
        dbps.append(np.min(abp_seg[a:b]))
    SBP = float(np.mean(sbps))
    DBP = float(np.mean(dbps))
    HR = 60.0/np.mean(np.diff(peaks)/fs)
    labels.append([SBP, DBP, HR])
labels = np.array(labels)

# --- Bước 6: Lọc chất lượng đoạn (ngưỡng sinh lý) ---
mask = (~np.isnan(labels).any(axis=1)) & \
       (labels[:,0] > 80) & (labels[:,0] < 180) & \
       (labels[:,1] > 60) & (labels[:,1] < 130) & \
       (labels[:,2] > 40) & (labels[:,2] < 220)
ppg_segments = np.array(ppg_segments)[mask]
ecg_segments = np.array(ecg_segments)[mask]
labels = labels[mask]
print(f"Số segment hợp lệ sau lọc sinh lý: {ppg_segments.shape[0]}")

# --- Bước 7: Downsample số lượng đoạn ---
#ppg_segments = ppg_segments[::4]
#ecg_segments = ecg_segments[::4]
#labels = labels[::4]
print(f"Số segment sau downsample: {ppg_segments.shape[0]}")

# --- Lưu kết quả ---
with h5py.File(output_path, 'w') as f:
    f.create_dataset('ppg_segments', data=ppg_segments)
    f.create_dataset('ecg_segments', data=ecg_segments)
    f.create_dataset('labels', data=labels)
print(f"Đã lưu segment và nhãn sau tiền xử lý vào {output_path}")

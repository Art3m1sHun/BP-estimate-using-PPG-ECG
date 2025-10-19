# Cuff-less Blood Pressure Estimation — Tiền xử lý & Chuẩn bị dữ liệu

Pipeline tiền xử lý tín hiệu ECG/PPG và tính nhãn ABP (SBP/DBP/HR) để huấn luyện mô hình ước lượng huyết áp **không dùng cuff**. Repo này tập trung vào các bước làm sạch, phân đoạn, gán nhãn và kiểm tra chất lượng.

> **Dataset:** [UCI — Cuff-less Blood Pressure Estimation](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation)  
> **Tần số lấy mẫu mặc định:** `fs = 125 Hz`  
> **Kênh tín hiệu:** ECG, PPG, ABP

---

## Mục lục

- [Chuẩn bị môi trường](#chuẩn-bị-môi-trường)  
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)  
- [Chạy nhanh (Quickstart)](#chạy-nhanh-quickstart)  
- [Các script chính](#các-script-chính)  
- [Quy trình tiền xử lý chi tiết](#quy-trình-tiền-xử-lý-chi-tiết)  
  - [Bước 1 — Sàng lọc cấp bản ghi](#bước-1--sàng-lọc-cấp-bản-ghi)  
  - [Bước 2 — Xử lý PPG](#bước-2--xử-lý-ppg)  
  - [Bước 3 — Xử lý ECG](#bước-3--xử-lý-ecg)  
  - [Bước 4 — Phân đoạn cửa sổ trượt](#bước-4--phân-đoạn-cửa-sổ-trượt)  
  - [Bước 5 — Tính nhãn từ ABP](#bước-5--tính-nhãn-từ-abp)  
  - [Bước 6 — Lọc chất lượng cấp đoạn](#bước-6--lọc-chất-lượng-cấp-đoạn)  
  - [Bước 7 — Giảm trùng lặp số lượng đoạn](#bước-7--giảm-trùng-lặp-số-lượng-đoạn)  
  - [Bước 8 — Chuẩn hóa (khuyến nghị)](#bước-8--chuẩn-hóa-khuyến-nghị)  
  - [Bước 9 — Đóng gói đầu vào mô hình](#bước-9--đóng-gói-đầu-vào-mô-hình)

---

## Chuẩn bị môi trường

Yêu cầu tối thiểu:

- Python ≥ 3.8  
- Thư viện: `numpy`, `scipy`, `pywt`, `h5py`, `tqdm`, `matplotlib` (khuyến nghị cho kiểm tra trực quan)

Cài đặt nhanh:

```bash
pip install numpy scipy pywt h5py tqdm matplotlib
```

---

## Cấu trúc thư mục

```text
.
├── convert_mat_to_h5.py
├── combine_all_parts.py
├── preprocess_all_combined.py
├── check_processed_segments.py
├── signal_preprocessing.py
├── README.md
└── data/
    └── Part_1.mat   # dữ liệu gốc sau khi giải nén
    └── Part_2.mat
    └── Part_3.mat
    └── Part_4.mat
```

> Thư mục đích có thể là `/data` ở root project (như hướng dẫn bên dưới). Điều chỉnh path trong các script nếu bạn dùng layout khác.

---

## Chạy nhanh (Quickstart)

```bash
# Step 1 — Giải nén dataset về /data
# (tải từ: https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation)
unzip cuff+less+blood+pressure+estimation.zip -d /data

# Step 2 — Chuyển .mat → .h5
python convert_mat_to_h5.py

# Step 3 — Gộp các phần thành 1 file lớn
python combine_all_parts.py

# Step 4 — Tiền xử lý (đang thực hiện đến Bước 7)
python preprocess_all_combined.py

# Step 5 — Kiểm tra kết quả sau tiền xử lý
python check_processed_segments.py
```

---

## Các script chính

- **`convert_mat_to_h5.py`** — Đọc các file `.mat` và chuyển sang định dạng `HDF5 (.h5)` tiện cho I/O tuần tự.  
- **`combine_all_parts.py`** — Hợp nhất nhiều file/subject thành một kho dữ liệu chung (hoặc theo batch).  
- **`preprocess_all_combined.py`** — Chạy pipeline tiền xử lý từ Bước 1 → 7 (lọc bản ghi, lọc tín hiệu, phân đoạn, gán nhãn, lọc chất lượng, downsample).  
- **`check_processed_segments.py`** — Sanity check: trực quan hóa/lấy thống kê cơ bản sau tiền xử lý.  
- **`signal_preprocessing.py`** — Chứa các hàm xử lý lõi (filtering, detrend, peak detection, tính SBP/DBP/HR, v.v.).

---

## Quy trình tiền xử lý chi tiết

### Bước 1 — Sàng lọc cấp bản ghi

**Mục tiêu:** loại bỏ record “xấu” trước khi cắt đoạn.

- **Độ dài:** loại record có thời lượng `< 8 phút`  
  → cần `>= 8 × 60 × 125` mẫu.  
- **Thiếu dữ liệu:** loại record có **NaN** ở bất kỳ kênh nào.  
- **Kiểm tra sinh lý sơ bộ (toàn record hoặc một cửa sổ ≥ 8 phút liên tục):**
  - Từ **ABP**, tìm đỉnh tâm thu (SBP) và đáy tâm trương (DBP) theo nhịp.
  - Tính **HR** từ khoảng cách giữa các đỉnh tâm thu (xem Bước 5 về tìm đỉnh).
  - Loại record nếu **trung bình** các chỉ số nằm ngoài:
    - `SBP ≤ 80` hoặc `≥ 180` mmHg  
    - `DBP ≤ 60` hoặc `≥ 130` mmHg  
    - `HR < 40` hoặc `> 220` bpm
  - Nếu không đủ `≥ 2` nhịp hợp lệ, loại.

> **Gợi ý cài đặt nhanh (Python):** kiểm tra `len(sig)`, `np.isnan`, sau đó ước lượng sơ bộ SBP/DBP/HR toàn record bằng thủ tục Bước 5.

---

### Bước 2 — Xử lý PPG (lọc + hiệu chỉnh nền)

**Mục tiêu:** loại nhiễu thấp/cao tần và trôi đường nền.

- **Chebyshev II band-pass** bậc 4, dải `0.5–10 Hz` (zero-phase):

  ```python
  from scipy import signal
  sos = signal.cheby2(4, 20, [0.5, 10], btype='bandpass', fs=125, output='sos')
  ppg_f = signal.sosfiltfilt(sos, ppg_raw)
  ```

- **Detrend đa thức** bậc 3 để bỏ baseline drift:

  ```python
  import numpy as np
  x = np.arange(len(ppg_f))
  trend = np.polyval(np.polyfit(x, ppg_f, 3), x)
  ppg_f = ppg_f - trend
  ```

---

### Bước 3 — Xử lý ECG (khử trôi + khử nhiễu cao tần)

**Mục tiêu:** ổn định đường nền, giảm nhiễu.

- **Khử trôi nền** (high-pass ~ `0.1 Hz`, bậc 8, zero-phase):

  ```python
  from scipy import signal
  sos = signal.butter(8, 0.1, btype='highpass', fs=125, output='sos')
  ecg_hp = signal.sosfiltfilt(sos, ecg_raw)
  ```

  > *Ghi chú:* mô tả “0.1 Hz low-pass” đôi khi được dùng để chỉ bỏ trôi; trong triển khai thực tế, **high-pass 0.1 Hz** là nhất quán để loại DC/trôi nền.

- **Wavelet denoise (DWT)** với `db6`, level 3, soft-threshold kiểu VisuShrink:

  ```python
  import pywt, numpy as np
  coeffs = pywt.wavedec(ecg_hp, 'db6', level=3, mode='symmetric')
  sigma = np.median(np.abs(coeffs[-1])) / 0.6745
  thr = sigma * np.sqrt(2 * np.log(len(ecg_hp)))
  coeffs[1:] = [pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]]
  ecg_f = pywt.waverec(coeffs, 'db6', mode='symmetric')[:len(ecg_hp)]
  ```

---

### Bước 4 — Phân đoạn cửa sổ trượt

**Mục tiêu:** tạo mẫu huấn luyện có độ dài cố định.

- **Chiều dài đoạn:** `8 giây` → `1000` mẫu.  
- **Chồng lấp:** `75%` → bước trượt `2 giây` → `250` mẫu.  
- **Chỉ số (start, end):**

  ```python
  fs = 125
  win = 8 * fs        # 1000
  step = 2 * fs       # 250
  idx = [(s, s + win) for s in range(0, len(ecg_f) - win + 1, step)]
  ```

---

### Bước 5 — Tính nhãn trong từng đoạn từ ABP

**Mục tiêu:** gán `SBP/DBP/HR` cho mỗi đoạn 8 s.

- **Tìm đỉnh tâm thu trên ABP** bằng `find_peaks` với khoảng cách tối thiểu theo **HR tối đa**:
  - `HR_max = 220 bpm` → `distance_min ≈ fs * (60 / 220) ≈ 0.27 * fs` → ~`34` mẫu ở `fs=125`.
  - Dùng **prominence** tỷ lệ theo độ lệch chuẩn tín hiệu.

  ```python
  from scipy import signal
  import numpy as np

  distance = int(0.27 * 125)
  prom = np.std(abp_seg) * 0.6
  peaks, _ = signal.find_peaks(abp_seg, distance=distance, prominence=prom)
  if len(peaks) < 2:
      # loại đoạn
      ...
  ```

- **SBP/DBP theo nhịp** và **trung bình đoạn**:

  ```python
  sbps = abp_seg[peaks]
  dbps = []
  for i in range(len(peaks) - 1):
      a, b = peaks[i], peaks[i + 1]
      dbps.append(np.min(abp_seg[a:b]))
  SBP = float(np.mean(sbps))
  DBP = float(np.mean(dbps))
  ```

- **Tính HR (bpm)**:

  ```python
  RR_sec = np.diff(peaks) / 125.0
  HR = 60.0 / np.mean(RR_sec)
  ```

---

### Bước 6 — Lọc chất lượng cấp đoạn

**Mục tiêu:** loại các đoạn nhiễu hoặc không hợp lệ.

- **Ngưỡng sinh lý trong đoạn** (từ SBP/DBP/HR vừa tính):
  - Loại nếu bất kỳ điều kiện sau đúng:
    - `SBP ≤ 80` hoặc `≥ 180` mmHg  
    - `DBP ≤ 60` hoặc `≥ 130` mmHg  
    - `HR < 40` hoặc `> 220` bpm

- **Đồng bộ nhịp ECG–PPG**: số lượng đỉnh trong cùng đoạn **phải bằng nhau**.
  - **ECG (R-peaks):**
    ```python
    r_peaks, _ = signal.find_peaks(ecg_seg,
                                   distance=int(0.27*fs),
                                   prominence=np.std(ecg_seg)*0.6)
    ```
  - **PPG (đỉnh xung):**
    ```python
    ppg_peaks, _ = signal.find_peaks(ppg_seg,
                                     distance=int(0.27*fs),
                                     prominence=np.std(ppg_seg)*0.4)
    ```
  - Nếu `len(r_peaks) != len(ppg_peaks)` → **loại đoạn**.

> **Mẹo:** Chuẩn hóa biên độ cục bộ trước khi tìm đỉnh: `x = (x - mean) / std`. Đặt `prominence = k * std` (ECG `k≈0.6`; PPG `k≈0.4`) rồi tinh chỉnh theo chất lượng.

---

### Bước 7 — Giảm trùng lặp số lượng đoạn (downsample segments)

**Mục tiêu:** giảm tải tính toán do overlap dày.

- **Giữ mỗi đoạn thứ 4** theo thứ tự thời gian để còn ~`25%` số đoạn ban đầu:

  ```python
  X = X_all_segments[::4]         # (N', 1000, 2)
  y = y_all_segments[::4]         # (N', 3)  -> [SBP, DBP, HR]
  ```

> *Trạng thái hiện tại:* pipeline đã triển khai đến **Bước 7**.

---

### Bước 8 — Chuẩn hóa (khuyến nghị)

Không bắt buộc, nhưng giúp huấn luyện ổn định:

- **Theo đoạn** (nhanh, ít rò rỉ thống kê):
  - Với mỗi đoạn, cho từng kênh: `z = (x - mean) / (std + ε)`.
- **Theo bản ghi** (giữ quan hệ tương đối giữa các đoạn cùng record):
  - Tính `mean/std` trên **toàn record** cho mỗi kênh, áp dụng cho **tất cả** đoạn của record đó.

> Tránh chuẩn hóa dựa trên **toàn bộ dữ liệu gộp** để hạn chế **data leakage**.

---

### Bước 9 — Đóng gói đầu vào mô hình

- **Đầu vào mỗi đoạn:** tensor `(T, C) = (1000, 2)` với thứ tự kênh **`[ECG, PPG]`**.  
- **Nhãn:** vector **`[SBP, DBP, HR]`** theo **trung bình nhịp** trong đoạn.  
- **Hợp nhất** tất cả đoạn **hợp lệ** từ mọi record:
  - `X ∈ (N, 1000, 2)`, `y ∈ (N, 3)`
- **Shuffle** trước khi chia **fold/huấn luyện**.

---



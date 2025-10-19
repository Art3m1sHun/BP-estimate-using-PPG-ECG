import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from CNN_two_scale_layer import model_CNN
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
# Thêm vào đầu file
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# === PHẦN 1: TẢI DỮ LIỆU TỪ FILE H5 ===

file_path = r'D:\Python\data-ppg-ecg\cuff+less+blood+pressure+estimation\filtered_segments.h5' # <-- THAY ĐỔI ĐƯỜNG DẪN NÀY


ECG_DATASET_NAME = 'ecg_segments'
PPG_DATASET_NAME = 'ppg_segments'
LABEL_DATASET_NAME = 'labels'


# Mở file và tải toàn bộ dữ liệu vào mảng NumPy
with h5py.File(file_path, 'r') as hf:
    # Ký hiệu [:] đảm bảo toàn bộ dữ liệu được đọc vào bộ nhớ
    ecg_data = hf[ECG_DATASET_NAME][:]
    ppg_data = hf[PPG_DATASET_NAME][:]
    labels_data = hf[LABEL_DATASET_NAME][:]


# === THÊM 2 DÒNG NÀY ĐỂ SỬA LỖI CHIỀU DỮ LIỆU ===
ecg_data = np.expand_dims(ecg_data, axis=-1)
ppg_data = np.expand_dims(ppg_data, axis=-1)
# ===============================================

print("Tải dữ liệu thành công!")
print(f"Số lượng mẫu ECG: {ecg_data.shape}")
print(f"Số lượng mẫu PPG: {ppg_data.shape}")
print(f"Số lượng nhãn: {labels_data.shape}")


# === PHẦN 2: CHUẨN BỊ DỮ LIỆU VÀ MÔ HÌNH ===

# Chia dữ liệu thành 80% cho training và 20% cho validation
# random_state để đảm bảo kết quả chia là như nhau mỗi lần chạy
X_ecg_train, X_ecg_val, X_ppg_train, X_ppg_val, y_train, y_val = train_test_split(
    ecg_data, ppg_data, labels_data, test_size=0.2, random_state=42
)

# Khởi tạo mô hình
model = model_CNN()

# Biên dịch mô hình
model.compile(optimizer='adam', loss = 'mse', metrics=['mae']) # Thêm MAE để dễ theo dõi

model.summary()

# === Callback ===

class R2ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data  # (X_val, y_val)

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0)
        r2 = r2_score(y_val, y_pred)
        logs = logs or {}
        logs['val_r2'] = r2
        print(f" — val_r2: {r2:.4f}")
        


learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss', # THAY ĐỔI Ở ĐÂY
    patience=4,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)
checkpoint = ModelCheckpoint(
    r"D:\Python\data-ppg-ecg\cuff+less+blood+pressure+estimation\BP_estimation.h5", # Thêm đuôi .h5 để lưu file đầy đủ
    monitor='val_loss', # THAY ĐỔI Ở ĐÂY
    verbose=1,
    save_best_only=True,
    mode='min' # THAY ĐỔI Ở ĐÂY
)
r2_callback = R2ScoreCallback(validation_data=([X_ecg_val, X_ppg_val], y_val))
callbacks = [learning_rate_reduction, checkpoint, r2_callback]

# === PHẦN 3: HUẤN LUYỆN MÔ HÌNH ===

plot_model(model, to_file="my_model.png", show_shapes=True)

print("\nBắt đầu quá trình huấn luyện...")

# **Điểm quan trọng:**
# - Đầu vào X phải là một LIST hoặc TUPLE chứa các mảng đầu vào: [X_ecg_train, X_ppg_train]
# - Dữ liệu validation cũng có cấu trúc tương tự: ([X_ecg_val, X_ppg_val], y_val)
history = model.fit(
    [X_ecg_train, X_ppg_train], # Dữ liệu đầu vào cho training
    y_train,                     # Nhãn cho training
    batch_size=16,               # Số mẫu dữ liệu cho mỗi lần cập nhật trọng số
    epochs=2000,                   # Số lần lặp lại toàn bộ tập dữ liệu training
    validation_data=([X_ecg_val, X_ppg_val], y_val),
    callbacks=callbacks
)

print("\nHuấn luyện hoàn tất! ✅")


# Lấy các giá trị loss và mae từ đối tượng history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_mae = history.history['mae']
val_mae = history.history['val_mae']

# Lấy số lượng epoch đã chạy
epochs_range = range(len(train_loss))

# Tạo một figure chứa 2 biểu đồ con
plt.figure(figsize=(14, 6))

# Biểu đồ 1: Training & Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Biểu đồ Loss (MSE) qua các Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Biểu đồ 2: Training & Validation MAE
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_mae, label='Training MAE')
plt.plot(epochs_range, val_mae, label='Validation MAE')
plt.legend(loc='upper right')
plt.title('Biểu đồ MAE qua các Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.grid(True)

# Hiển thị biểu đồ
plt.suptitle('Kết quả quá trình huấn luyện', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Điều chỉnh để tiêu đề chính không bị che
plt.show()

# === PHẦN 4 (ĐÃ SỬA): ĐÁNH GIÁ MÔ HÌNH VÀ TRỰC QUAN HÓA KẾT QUẢ ===

# 1. Dự đoán trên tập validation (CHỈ GỌI 1 LẦN DUY NHẤT)
print("\nBắt đầu dự đoán trên tập validation...")
predictions = model.predict([X_ecg_val, X_ppg_val])
print("Dự đoán hoàn tất!")

# === SỬA LẠI ĐOẠN CODE TÍNH TOÁN R² ===

print("\nĐánh giá hiệu suất mô hình trên toàn bộ tập validation:")

# Tính R² riêng lẻ cho từng cột (chiều)
# y_val[:, 0] là cột SBP, predictions[:, 0] là dự đoán SBP
r2_sbp = r2_score(y_val[:, 0], predictions[:, 0])
r2_dbp = r2_score(y_val[:, 1], predictions[:, 1])
r2_hr  = r2_score(y_val[:, 2], predictions[:, 2])

# Bây giờ các biến này là những con số duy nhất và có thể in ra
print(f"  - Chỉ số R-squared (R²) cho SBP: {r2_sbp:.4f}")
print(f"  - Chỉ số R-squared (R²) cho DBP: {r2_dbp:.4f}")
print(f"  - Chỉ số R-squared (R²) cho HR:  {r2_hr:.4f}")
# Tính R² cho 10 mẫu đầu tiên
r2_first_10 = r2_score(y_val[:10], predictions[:10])
print(f"  - Chỉ số R-squared (R²) tổng hợp cho 10 mẫu đầu tiên: {r2_first_10:.4f}")


# 3. In bảng so sánh cho 10 mẫu đầu tiên
print("\nSo sánh kết quả dự đoán với nhãn thật (cho 10 mẫu đầu tiên):")
print("-" * 65)
print(f"{'Mẫu':<5} | {'Dự đoán (SBP, DBP, HR)':<30} | {'Thực tế (SBP, DBP, HR)':<30}")
print("-" * 65)
for i in range(10):
    pred_str = f"{predictions[i, 0]:>6.2f}, {predictions[i, 1]:>6.2f}, {predictions[i, 2]:>6.2f}"
    true_str = f"{y_val[i, 0]:>6.2f}, {y_val[i, 1]:>6.2f}, {y_val[i, 2]:>6.2f}"
    print(f"{i:<5} | {pred_str:<30} | {true_str:<30}")
print("-" * 65)


# 4. Vẽ biểu đồ Scatter Plot (Dự đoán vs. Thực tế)
output_labels = ['Huyết áp tâm thu (SBP)', 'Huyết áp tâm trương (DBP)', 'Nhịp tim (HR)']
plt.figure(figsize=(18, 5)) # Tăng chiều rộng để vừa 3 biểu đồ

for i in range(3):
    plt.subplot(1, 3, i+1) # SỬA LỖI Ở ĐÂY
    plt.scatter(y_val[:, i], predictions[:, i], alpha=0.5)
    
    min_val = min(np.min(y_val[:, i]), np.min(predictions[:, i]))
    max_val = max(np.max(y_val[:, i]), np.max(predictions[:, i]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Dự đoán hoàn hảo')
    
    r2_current = r2_score(y_val[:, i], predictions[:, i])
    plt.title(f'Dự đoán vs. Thực tế cho {output_labels[i]}\n(R² = {r2_current:.4f})')
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.grid(True)
    plt.legend()

plt.suptitle('So sánh giá trị dự đoán và giá trị thực tế', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()








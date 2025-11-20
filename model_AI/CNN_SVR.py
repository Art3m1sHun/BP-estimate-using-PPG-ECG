import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model

# --- 1. Định nghĩa Tham số ---
# Giả sử đầu vào là ảnh/tín hiệu 2D (ví dụ 128x128) với 1 kênh màu (grayscale)
input_shape = (128, 128, 1)
output_units = 1  # Dự đoán 1 giá trị số liên tục

# --- 2. Xây dựng Mô hình ---
model = Sequential()

# Tầng đầu vào (có thể bỏ qua nếu dùng Conv2D với input_shape)
# model.add(Input(shape=input_shape))

# C1: 8 filters, 3x3 kernel, input channel 1
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

# C2: 16 filters, 3x3 kernel
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

# C3: 32 filters, 3x3 kernel
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

# C4: 32 filters, 3x3 kernel
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

# C5: 64 filters, 3x3 kernel
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(pool_size=(2, 2), strides=2))


# Tầng Fully Connected (FC)
# Phẳng hóa đầu ra 3D thành 1D
model.add(Flatten())

# Thêm một Tầng Dense ẩn (ví dụ: 128 neurons), tùy chọn
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

# Regression Layer (Tầng Hồi quy)
# Số neurons bằng số lượng giá trị cần dự đoán (output_units)
# Activation='linear' (hoặc không khai báo) cho bài toán hồi quy
model.add(Dense(output_units, activation='linear'))

# --- 3. Biên dịch Mô hình ---
# Sử dụng 'mse' (Mean Squared Error) là hàm mất mát phổ biến cho hồi quy
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae']) # mae: Mean Absolute Error

# Hiển thị tóm tắt mô hình
model.summary()
plot_model(model, to_file='model_architecture.png', show_shapes=True)
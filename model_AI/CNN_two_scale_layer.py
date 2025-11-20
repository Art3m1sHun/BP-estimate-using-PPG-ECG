import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, concatenate,
    Dense, GlobalAveragePooling1D, BatchNormalization, Dropout
)
from tensorflow.keras.models import Model

def two_scale_conv_block(input_tensor, num_filters=32):
    conv_large = Conv1D(num_filters, 3, activation='relu', padding='same')(input_tensor)
    pool_large = MaxPooling1D(pool_size=2, strides=2)(conv_large)

    conv_small = Conv1D(num_filters, 1, activation='relu', padding='same')(input_tensor)
    pool_small = MaxPooling1D(pool_size=2, strides=2)(conv_small)

    return concatenate([pool_large, pool_small], axis=-1)

def model_CNN(input_shape=(1000, 1)):
    ecg_input = Input(shape=input_shape, name='ecg_input')
    ppg_input = Input(shape=input_shape, name='ppg_input')

    # Two-scale blocks cho ECG & PPG
    ecg_block = two_scale_conv_block(two_scale_conv_block(ecg_input))
    ppg_block = two_scale_conv_block(two_scale_conv_block(ppg_input))

    # Ghép đặc trưng hai nhánh
    combined = concatenate([ecg_block, ppg_block], axis=-1)

    # Dense tầng giữa
    output1 = Dense(128, activation='relu', name='feature_dense')(combined)

    # Global pooling
    x = GlobalAveragePooling1D()(output1)

    # Thêm BatchNorm và Dropout ngay trước lớp đầu ra
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # bạn có thể chỉnh 0.3 → 0.5 nếu bị overfitting

    # Lớp đầu ra
    out = Dense(3, activation='linear', name='out')(x)

    # Tạo mô hình hoàn chỉnh
    model = Model(inputs=[ecg_input, ppg_input], outputs=out, name='Two_Scale_LRCN_Model')
    return model

model = model_CNN()
model.summary()
plot_model(model, to_file='two_scale_layer_CNN.png', show_shapes=True)
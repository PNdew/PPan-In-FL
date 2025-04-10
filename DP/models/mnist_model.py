import tensorflow as tf
from tensorflow.keras import layers, models

def get_model():
    model = models.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 lớp cho các chữ số 0-9
    ])
    return model
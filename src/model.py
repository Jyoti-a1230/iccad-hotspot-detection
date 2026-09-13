"""Baseline CNN architecture for hotspot classification."""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout

def build_model(input_shape=(224, 224, 1)):
    """Returns a freshly-initialized baseline CNN. Call this once per benchmark
    to avoid accidentally continuing training from a previous benchmark's weights."""
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='valid', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

    model.add(Conv2D(64, kernel_size=(3,3), padding='valid'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

    model.add(Conv2D(128, kernel_size=(3,3), padding='valid'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

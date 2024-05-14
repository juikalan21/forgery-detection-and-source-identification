import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

X = np.random.rand(100, 128, 128, 3)  
y_forgery = np.random.randint(0, 2, 100)  
y_source = np.random.randint(0, 5, 100)  

X_train, X_test, y_forgery_train, y_forgery_test, y_source_train, y_source_test = train_test_split(
X, y_forgery, y_source, test_size=0.2, random_state=42)

forgery_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

forgery_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
forgery_model.fit(X_train, y_forgery_train, epochs=5, validation_data=(X_test, y_forgery_test))

source_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  
])

source_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
source_model.fit(X_train, y_source_train, epochs=5, validation_data=(X_test, y_source_test))



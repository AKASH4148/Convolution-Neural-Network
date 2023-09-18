"""
Dropout is a regularization technique used to prevent overfitting.
 During training, dropout randomly deactivates a portion of neurons, forcing the network to learn more robust and general features.
"""



import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#creating a model with dropout
model=tf.keras.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D((2,2)),
    Dropout(0.23),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

#compile the model
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
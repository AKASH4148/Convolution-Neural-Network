"""
Batch Normalization:
Batch normalization is a technique that normalizes the inputs of a layer 
during training, aiming to stabilize and accelerate the training process. 
It helps to mitigate issues like vanishing/exploding gradients and can lead to faster convergence.
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D,Flatten,Dense

#Creating a model with BatchNormalization
model=tf.keras.Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(64,64,3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


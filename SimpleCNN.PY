"""
Convolution Neural Network
A convolution neural network is a class of deep neural network primarily designed to process and anlyze grid-like data,
such as images and videos. CNNs are well suited for task invoving pattern recognition and featurn extraction from visual data.
"""

#importing required library
import tensorflow as tf
from tensorflow.keras import layers, models

#creating a simple cnn model
model=models.Sequential([
    #Convolution layer with 32 filters, with each of size 3x3, and relu activation function
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    #Max pooling layer with pool size 2x2
    layers.MaxPooling2D((2,2)),
    #Flatting the 2D data to 1D
    layers.Flatten(),
    
    #Fully connected layer with 64 nurons and relu activation functions
    layers.Dense(64, activation="relu"),
    
    #Output layer with 10 neurons for classification
    layers.Dense(10, activation="softmax")
])
model.summary()

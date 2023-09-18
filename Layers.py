"""
Layers :
A CNN consists multiple layers that performs diffrent tasks. The key layer is :

Convolution Layer :
This is where the filter size detect the pattern

Pooling Layer :
This layer reduces the dimensions of the images, making network faster and more efficient. it helps retain important features.

Fully Connected Layer:
This layer connects the features learned by the previous layers to make final descision about what's an image
"""
#required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

#Load and preprocess the data
(x_train, y_train), (x_test, y_test)=mnist.load_data()
x_train=np.expand_dims(x_train, axis=-1).astype('float32')/255.0
x_test= np.expand_dims(x_test, axis=-1).astype('float32')/255.0
y_train=to_categorical(y_train, num_classes=10)
y_test=to_categorical(y_test, num_classes=10)
print(x_train.shape)
print(x_test.shape)

#create a cnn model
model=models.Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3), activation="relu"),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

#Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

#evaluate the model
test_loss, test_accuracy=model.evaluate(x_test,y_test)
print(f"Test loss : {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
"""
Data augementation is a technique used to artificially increase the diversity and size of a 
dataset by applying various transformations to the original data. This can help improve the 
generalization of machine learning models and prevent overfitting.
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

#Example image for data augmentation
img_path='kas.jpg' #path
img=plt.imread(img_path)

#Create an ImageDataGenerator  with augmentation options
datagen= ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#reshape the image for generator
img=np.expand_dims(img, axis=0)

#Generate augmented images
augmented_img=datagen.flow(img, batch_size=1)

#Plot the original and augumented images
fig, axes=plt.subplots(1,5, figsize=(15,5))
for i in range(5):
    augmented_images=next(augmented_img)[0].astype(np.uint8)
    axes[i].imshow(augmented_images)
    axes[i].axis('off')
plt.show()

"""
One of the main benefit of data augmentation is addressing overfitting. 
By introducing variability in the training data, the model becomes more robust 
and less likely to memorize the training samples
"""

import tensorflow as tf
from tensorflow.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Create a simple cnn model
model=Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu")
    MaxPooling2D((2,2))
    Conv2D(128, (3,3), activation="relu")
    MaxPooling2D((2,2))
    Flatten()
    Dense(512, activation="relu")
    Dense(1, activation="sigmoid")
])

#Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

#Create an ImageDataGenerator with data augmentation
train_datagen=ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#Load training data using the generator
train_generator=train_datagen.flow_from_directory(
    'train_data_dir', #path of our training data
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

#fit the model with augemented data
model.fit(train_generator, steps_per_epoch=100, epochs=10)

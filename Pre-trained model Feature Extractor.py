"""
Pretrained model as a feature extractor
Pre-trained CNNs are models that has been trained on huge datasets, 
often for image classification tasks. They have learned to extract valuable feature from images.

We can use use this pre-trained models as feature extractor by removing the top classfication 
layers and using the remaining layers to extracts features from your own images.
"""

#required libraries
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

#Load a pretrained resenet50 model without top classfication layer
base_model= ResNet50(weights='imagenet', include_top=False)

#Load and preprocess the image
img_path='images.jfif' #path
img=image.load_img(img_path, target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
x=preprocess_input(x)
print(x.shape)

#get the feature extracted by the pre-trained model
features=base_model.predict(x)

#print the shape of the extracted features
print("Shape of extracted features", features.shape)
"""
Transfer Learning :
Transfer learning is a technique in machine learning and deep learning where a model that has been trained on 
one task is reused as the starting point for a model on a second task

In the context of CNNs, transfer learning involves using a pre-trained neural network model as a foundation for
 solving a related but diffrent problems.
"""
#Transfer Learning
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

#Load a pretrained model vgg16 model without the top classification layer
base_model=VGG16(weights="imagenet", include_top=False)

#Load and preprocess an example image
img_path='images.jfif' #path
img=image.load_img(img_path, target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
x=preprocess_input(x)
print(x.shape)

#get the feature extracted by the pretrained model
feature=base_model.predict(x)

#print the shape of extrcated model
print("Shape of extracted feature :",feature.shape)


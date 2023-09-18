"""
Object Detection with Region Proposal Networks (RPNs) and Anchor Boxes:
Object detection involves identifying objects in an image and drawing 
bounding boxes around them. Region Proposal Networks (RPNs) are a part of modern object
detection methods like Faster R-CNN.

Anchor boxes are predefined boxes of different sizes and aspect ratios that the 
network uses to suggest potential object locations.
"""
import tensorflow as tf 
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

#Load the pre-trained ResNet50 model without the top classfication layers
base_model=ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))

#Create a RPN and classifier on top
x=base_model.output
x=Conv2D(256, (3,3), activation='relu', padding='same')(x) #RPN
region_proposals= Conv2D(9*4, (1,1), activation='linear')(x) #9 anchor boxes * 4Coordinates
classfication=Conv2D(9*num_classes, (1,1), activation='sigmoid')(x) #9 anchor boxes * num_classes

model=Model(input=base_model.input, output=[region_propposals, classfication])

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')
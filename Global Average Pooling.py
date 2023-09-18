"""
Global Average Pooling (GAP) is a technique used to replace 
fully connected (FC) layers in the final stages of a CNN. 
It helps reduce overfitting and improve localization.
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Activation, Input

input_layer=Input(shape=(64,64,3))
x=Conv2D(32, (3,3), activation='relu')(input_layer)
x=Conv2D(64, (3,3), activation='relu')(x)
x=Conv2D(128, (3,3), activation='relu')(x)
output_layer=GlobalAveragePooling2D()(x) #Replace full connected layer with GAP
output_layer=Activation('softmax')(output_layer)

model=tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

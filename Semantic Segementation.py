"""
Semantic Segementation with U-Net
It involves classifying each pixel in an image into a specific class label, 
U-Net is used for semantic segememtation that consists of an encoder and decoder, 
allowing the network to capture both high level and fine grained feature.
"""
import tensorflow as tf
from tensoflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, concatenate

def build_unet(input_shape, num_classes):
    inputs=Input(input_shape)
    conv1=Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    conv1=Conv2D(64, (3,3), activation='relu', padding='same')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    
    #....Continuing with encoding ond decoding
    up9=UpSampling2D(siz=(2,2))(conv8)
    conv9=Conv2D(64, (2,2), activation='relu', padding='same')(up9)
    conv9=Conv2D(64, (2,2), activation='relu', padding='same')(conv9)
    
    oututs=Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
    model=tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

#Build U-Net model
inpit_shape=(256,256, 3)
num_classes=21 #number of segmentation classes including background
model=build_unet(input_shape, num_classes)

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')
        
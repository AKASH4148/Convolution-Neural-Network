"""
 Residual connections, are used to mitigate vanishing gradient 
 problem and make it easier for deep networks to train. 
 They enable gradients to flow directly through network,
helping in the training of very deep architectures.
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Input

def residual_block(x, filters):
    #A single residual block with skip connections
    shortcut=x
    x=Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x=Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x=Add()([shortcut, x])
    return x

input_layer=Input(shape=(64,64, 3))
x=residual_block(input_layer, 32)
x=residual_block(x,64)
x=residual_block(x, 128)
output_layer=Conv2D(10, (3,3), activation='softmax')(x)

model=tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
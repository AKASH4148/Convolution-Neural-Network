"""Filter in convolution layers learn to recognize various feature like edges,
textures, and more complex patterns, During training, CNNs adjust the filter weights so 
that they become sensitive to specific patterns in the input images.
"""
#inutiative understanding of filters
import matplotlib.pyplot as plt
import numpy as np

#creating a single image with a texture-like pattern
image=np.array([
    [1, 2, 1, 2, 1],
    [2, 1, 2, 1, 2],
    [1, 2, 1, 2, 1],
    [2, 1, 2, 1, 2],
    [1, 2, 1, 2, 1]
])

#Defining a filter that might learn to recognize texture-like pattern
texture_filter=np.array([
    [1, 0, -1],
    [0, 2, 0],
    [-1, 0, 1]
])

#applying the convolution using the tecture filter
feature_map=convolve2d(image, texture_filter, mode="valid")

#visualizing the original image and the feature map
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(feature_map, cmap='gray')
plt.title("Feature Map(Texture)")
plt.show()

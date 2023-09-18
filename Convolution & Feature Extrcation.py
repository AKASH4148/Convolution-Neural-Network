"""
Convolution & Feature extraction
convolution layers in CNNs are responsible for detecting local patterns and features in images. 
They use filters to slide over the input image and perform a mathmatical operation called convolution.
"""
#Convolution and feature extraction
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
#creating a simple grascale image
image=np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

#Defining an edge detection filter
edge_filter=np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

#Applying convolution using the edge detection filter
feature_map= convolve2d(image, edge_filter, mode="valid")

#visualizing the original image and the feature map
plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(feature_map, cmap='gray')
plt.title("Feature Map(Edges)")
plt.show()
import warnings
import numpy as np
import rasterio as rio
import glob, os
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
from skimage.transform import pyramid_expand, pyramid_reduce, resize

def s2toRGB(x):
    """
    sentinel 2 image with 10 channels to RGB
    """
    B,G,R = x[:3]
    X = np.stack([R,G,B],axis=-1)
    X = np.clip(X*4,0,1)
    return X

def preprocess_s1(s1):
    
    # first restrict the values from -35 to 0
    # then add 35 so we are in the [0,35] window
    # with 0 beeing the lowest refection
    # then divide by 35 to get it into [0,1]

    return (np.clip(s1,-35,0)+35)/35

def preprocess_s2(s2):
    
    # devide by 10k to get the 0-100% reflection
    # window in the... then clip to [0,1]

    return np.clip(s2/10000,0,1)


def downsampleKeepPixelSpacing(img, factor):
    """
    Downsamples an image while preserving the pixel spacing.

    Args:
        img (ndarray): The input image to be downsampled.
        factor (float): The downsampling factor.

    Returns:
        ndarray: The downsampled image with preserved pixel spacing.
    """

    # Get the input shape of the image
    in_shape = img.shape

    # Downsample the image using pyramid_reduce function
    ds_version = pyramid_reduce(img, factor, channel_axis=0, order=3)

    # Upsample the downsampled image using pyramid_expand function
    us_version = pyramid_expand(ds_version, factor, channel_axis=0, order=3)

    # Check if the shape of the upsampled image is different from the input shape
    if not us_version.shape == in_shape:

        # Calculate the differences in shape between the input and upsampled image
        diffs = [np.abs(a - b) for a, b in zip(in_shape, us_version.shape)]

        # Assert that the maximum difference is less than or equal to 4
        assert max(diffs) <= 4

        # Resize the upsampled image to match the input shape
        us_version = resize(us_version, in_shape)

    # Return the upsampled image
    return us_version
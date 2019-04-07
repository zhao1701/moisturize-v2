#!/usr/bin/env python

"""
This module contains utilities for predicting with and inspecting
autoencoders models.
"""


from pathlib import Path

import numpy as np
from keras.models import Model
from keras.models import load_model
from .utils import unpack_tensors


def tile_multi_image_traversal(latent_traversals, num_rows):
    traversal_resolution, num_samples, img_height, img_width, num_channels = \
        latent_traversals.shape
    assert (num_samples % num_rows == 0), (
        'The number of rows of the stitched image must be an integer divisor '
        'of the number of samples in the batch.')
    num_cols = num_samples // num_rows
    latent_traversals = latent_traversals.reshape(
        traversal_resolution, num_rows, num_cols, img_height, img_width,
        num_channels)
    latent_traversals = latent_traversals.transpose(0, 1, 3, 2, 4, 5)
    latent_traversals = latent_traversals.reshape(
        traversal_resolution, num_rows * img_height, num_cols * img_width,
        num_channels)
    return latent_traversals

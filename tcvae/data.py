#!/usr/bin/env python

"""
This module contains utilities for reading and handling image data.
"""


from pathlib import Path

import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array


class ImageDataGenerator(Sequence):
    """
    A data generator for reading and processing batches of data from disk.
    This can be used to provide data to a Keras model without needing to
    load an entire dataset into memory.

    Parameters
    ----------
    data_dir : pathlib.Path or str
        The path to the directory containing image data.
    batch_size : int
        The number of images in each batch of data.
    shuffle : bool
        If true, shuffles the images following each training epoch.
    filetype : str
        The extension of the image data. Ex: `jpg` or `png`
    square_crop_length : int
        The width and height of the cropped image.
    """

    def __init__(
            self, data_dir, batch_size=32, shuffle=True,
            filetype='jpg', square_crop_length=128):

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filenames = list(data_dir.glob('*.{}'.format(filetype)))
        self.num_samples = len(self.filenames)
        self.square_crop_length = square_crop_length
        if self.shuffle:
            self.filenames = np.random.permutation(self.filenames).tolist()

    def __getitem__(self, index):
        """
        Retrieves a batch of images.

        Parameters
        ----------
        index : int
            The index of the batch to be returned.

        Returns
        -------
        imgs : tuple of length 2
            A tuple whose first element is a batch of image data (a Numpy
            array of shape (batch_size, square_crop_length,
            square_crop_length, num_channels). The second element is None and is
            necessary for proper integration with a Keras-based autoencoder.
        """
        # Decide which batch of images to load
        idx_start = index * self.batch_size
        idx_end = (index + 1) * self.batch_size
        batch_files = self.filenames[idx_start:idx_end]

        # Load and process image batch
        imgs = [read_img(file) for file in batch_files]
        if self.square_crop_length:
            imgs = [
                crop_square(img, side_length=self.square_crop_length)
                for img in imgs]
        imgs = np.array(imgs)

        # Return in form (x, y)
        imgs = (imgs, None)
        return imgs

    def __len__(self):
        return self.num_samples // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.filenames = np.random.permutation(self.filenames).tolist()


def read_img(file):
    """
    Load an image from disk.

    Parameters
    ----------
    file : str
        Path to image file.

    Returns
    -------
    img : np.ndarray
        An array of shape (height, width, num_channels).
    """
    img = load_img(file)
    img = img_to_array(img)
    img /= 255.  # Restrict pixels to between 0 and 1
    return img


def crop_square(img, side_length=128):
    """
    Create a centered square cropping of an image.

    Parameters
    ----------
    img : np.ndarray
        A numpy array of shape (height, width, num_channels).
    side_length : int
        The height and width of the cropped image.

    Returns
    -------
    img : np.ndarray
        A numpy array of shape (side_length, side_length, num_channels).
    """
    height, width, num_channels = img.shape

    # Crop image to square
    extra_padding = (max(height, width) - min(height, width)) // 2
    if height > width:
        img = img[extra_padding:-extra_padding]
    elif height < width:
        img = img[:, extra_padding:-extra_padding]

    # Zoom
    extra_padding = (min(height, width) - side_length) // 2
    assert (extra_padding >= 0)
    img = img[extra_padding:-extra_padding, extra_padding:-extra_padding]
    return img

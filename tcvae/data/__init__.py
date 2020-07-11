#!/usr/bin/env python

"""
This module contains utilities for reading and handling image data.
"""

import functools as ft
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from ..utils import check_path
from .augmentation import *


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
    file_type : str
        The extension of the image data. Ex: `jpg` or `png`
    """

    def __init__(
            self, data_dir, batch_size=32, shuffle=True,
            transformers=(), file_type='jpg', n_processes=-1):

        self.transformers = transformers
        self.transformation_fn = TransformComposer(*transformers)
        self.n_processes = get_parallelization(n_processes)
        self.data_dir = check_path(data_dir, Path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_names = list(self.data_dir.glob('*.{}'.format(file_type)))
        if self.shuffle:
            self.file_names = np.random.permutation(self.file_names).tolist()
        self.iteration_index = 0

    @property
    def n_samples(self):
        return len(self.file_names)

    def __repr__(self):
        lines = [
            'Image Data Generator',
            f'Data path: {self.data_dir}',
            f'Number of files: {self.n_samples}',
            f'Parallelization: {self.n_processes}']
        lines.insert(1, '-' * len(lines[0]))
        return '\n'.join(lines)

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
        
        # Batch size at last index not guaranteed to match original
        # batch size. Because this will cause problems with TCVAE-type
        # latent losses, we always exclude the final batch.
        if index == len(self)-1:
            index = 0
        
        # Decide which batch of images to load
        idx_start = index * self.batch_size
        idx_end = (index + 1) * self.batch_size
        batch_files = self.file_names[idx_start:idx_end]
        imgs = self.load_processed_images(batch_files)

        # Return in form (x, y)
        imgs = (imgs, imgs)
        return imgs

    def load_processed_images(self, files):
        transformation_fn = TransformComposer(*self.transformers)
        with Pool(processes=self.n_processes) as p:
            np.random.seed()  # Necessary else child processes always the same
            imgs = p.map(read_img, files)
            imgs = p.map(transformation_fn, imgs)
        imgs = np.array(imgs)
        return imgs

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration_index < len(self):
            batch, _ = self[self.iteration_index]
            self.iteration_index += 1
            return batch
        else:
            raise StopIteration

    def reset_iterator(self):
        self.iteration_index = 0

    def load_data(self):
        self.reset_iterator()
        full_dataset = list()
        for batch in self:
            full_dataset.append(batch)
        full_dataset = np.vstack(full_dataset)
        self.reset_iterator()
        return full_dataset

    def load_n_images(self, n=1, random=True):
        if random:
            files = np.random.choice(self.file_names, size=n)
        else:
            files = sorted(self.file_names)[:n]
        imgs = self.load_processed_images(files)
        return imgs

    def on_epoch_end(self):
        if self.shuffle:
            self.file_names = np.random.permutation(self.file_names).tolist()


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
    file = check_path(file, str)
    img = load_img(file)
    img = img_to_array(img)
    img /= 255.  # Restrict pixels to between 0 and 1
    
    assert(np.all(img <= 1.0) and np.all(img >= 0.0))
    return img


def get_parallelization(n_processes):
    """
    Calculates the number of jobs that can be run simultaneously.

    Parameters
    ----------
    n_processes : int
        The desired number of processes to be executed simultaneously.
        If set to a number less than 1 or exceeding the max hardware
        limit, function returns the number of CPUs on the machine.

    Returns
    -------
    int
        The number of processes to be executed in parallel.
    """
    assert(isinstance(n_processes, int))
    n_processes = min(n_processes, cpu_count())  # Clip at hardware limit
    n_processes = cpu_count() if n_processes < 1 else n_processes
    return n_processes


def crop_center_square(img, side_length=128):
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

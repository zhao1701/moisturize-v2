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


class Predictor:

    def __init__(self, model):

        if isinstance(model, str):
            model = load_model(model)
        elif isinstance(model, Path):
            model = load_model(model.as_posix())
        elif isinstance(model, Model):
            pass
        else:
            raise ValueError(
                'Argument for parameter `model` must be a Keras model or a '
                'path to one.')

        self.encoder = model.get_layer('encoder')
        self.decoder = model.get_layer('decoder')
        self.tensors = unpack_tensors(
            self.encoder, self.decoder)

        self.model = Model(
            inputs=self.tensors['x'], outputs=self.tensors['y_pred'],
            name='vae')

        self.num_latents = int(self.tensors['z'].shape[-1])

    def encode(self, x, batch_size=32):
        _, z_mu, z_log_sigma = self.encoder.predict(x, batch_size=batch_size)
        z_sigma = np.exp(z_log_sigma)
        return z_mu, z_sigma

    def reconstruct(self, x, batch_size=32):
        y = self.model.predict(x, batch_size=batch_size)
        return y

    def decode(self, z, batch_size=32):
        y = self.decoder.predict(z, batch_size=batch_size)
        return y

    def make_traversal(
            self, x, latent_index, traversal_start=-4, traversal_end=4,
            traversal_resolution=25, batch_size=32, output_format='stitched',
            num_rows=None, num_cols=None):

        z_mu, _ = self.encode(x, batch_size=batch_size)
        traversal_sequence = np.linspace(
            traversal_start, traversal_end, traversal_resolution)

        latent_traversals = np.empty(shape=(traversal_resolution,) + x.shape)
        for traversal_index, traversal_point in enumerate(traversal_sequence):
            z_mu_traversal = z_mu.copy()
            z_mu_traversal[:, latent_index] = traversal_point
            y_traversal = self.decode(z_mu_traversal)
            latent_traversals[traversal_index] = y_traversal

        if output_format == 'images_first':
            latent_traversals = latent_traversals.transpose(1, 0, 2, 3, 4)
        elif output_format == 'stitched':
            latent_traversals = stitch_multi_image_traversal(
                latent_traversals, num_rows, num_cols)
        elif output_format == 'traversal_first':
            pass
        else:
            raise ValueError(
                'Argument for `output_format` must be one of the following '
                'strings: `images_first`, `traversal_first`, or `stitched`.')

        return latent_traversals

    def make_all_traversals(
            self, x, traversal_start=-4, traversal_end=4,
            traversal_resolution=25, batch_size=32):
        # TODO: variance thresholding, return output as dictionary with
        # latent index as key.
        pass



def stitch_multi_image_traversal(latent_traversals, num_rows, num_cols):
    traversal_resolution, num_samples, img_height, img_width, num_channels = \
        latent_traversals.shape
    assert (num_rows * num_cols == num_samples), (
        'The number of rows and columns of the stitched image is '
        'incompatible with the number of samples in the batch.')
    latent_traversals = latent_traversals.reshape(
        traversal_resolution, num_rows, num_cols, img_height, img_width,
        num_channels)
    latent_traversals = latent_traversals.transpose(0, 1, 3, 2, 4, 5)
    latent_traversals = latent_traversals.reshape(
        traversal_resolution, num_rows * img_height, num_cols * img_width,
        num_channels)
    return latent_traversals

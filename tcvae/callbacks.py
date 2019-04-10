#!/usr/bin/env python

"""
This module contains custom callback functions to be used for model training.
"""

import csv
from pathlib import Path

import imageio
import numpy as np
from keras.callbacks import Callback
from tcvae.utils import check_path, make_directory, deprocess_img
from tcvae.models import _make_autoencoder_models, TCVAE
from tcvae.data import ImageDataGenerator


class ReconstructionCheck(Callback):
    """
    Parameters
    ----------
    images : np.ndarray
        A batch of images to be reconstructed with shape (num_images,
        img_height, img_width, num_channels)
    output_dir : str or pathlib.Path
        The directory in which reconstructed images are stored.
    stem : str
        The name of the saved image files (without file extension).
    fmt : str
        One of {`jpg`, `png`}. Determines the extension of the saved image
        files.
    """

    def __init__(
            self, images, output_dir, stem='reconstruction', fmt='jpg'):
        super().__init__()
        self.images = images
        self.output_dir = check_path(output_dir, Path)
        self.stem = stem
        if fmt[0] != '.':
            fmt = '.' + fmt
        self.format = fmt
        self.model_predict = None

    def on_train_begin(self, logs=None):
        make_directory(self.output_dir, overwrite=True)
        encoder = self.model.get_layer('encoder')
        decoder = self.model.get_layer('decoder')
        _, self.model_predict, _ = _make_autoencoder_models(
            encoder, decoder)

    def on_epoch_end(self, epoch, logs=None):
        # Make reconstructions and stitch with original image
        reconstructions = self.model_predict.predict(self.images)
        enum_zip = enumerate(zip(self.images, reconstructions))
        for index, (original, reconstruction) in enum_zip:
            original = deprocess_img(original)
            reconstruction = deprocess_img(reconstruction)
            img_pair = np.hstack([original, reconstruction])
            base = '{}-{:0>2}{}'.format(self.stem, index, self.format)
            save_file = self.output_dir / base
            imageio.imwrite(save_file.as_posix(), img_pair)


class LatentTraversalCheck(Callback):
    """
    Parameters
    ----------
    image : np.ndarray
        An image on which to perform traversals along all latent dimensions.
    output_dir : str or pathlib.Path
        The directory in which traversal animations are stored.
    stem : str
        The name of the saved image files (without file extension).
    """
    def __init__(
            self, image, output_dir, stem='traversal', traversal_range=(-4, 4),
            traversal_resolution=10):
        super().__init__()
        if image.ndim == 4:
            assert(image.shape[0] == 1), (
                'Traversal check can only be performed on a single image.')
        else:
            image = np.expand_dims(image, axis=0)
        self.image = image
        self.output_dir = check_path(output_dir, Path)
        self.stem = stem
        self.tcvae = None
        self.traversal_range = traversal_range
        self.traversal_resolution = traversal_resolution

    def on_train_begin(self, logs=None):
        if not self.output_dir.is_dir():
            make_directory(self.output_dir)
        encoder = self.model.get_layer('encoder')
        decoder = self.model.get_layer('decoder')
        self.tcvae = TCVAE(encoder, decoder)

    def on_epoch_end(self, epoch, logs=None):
        traversals = self.tcvae.make_all_traversals(
            self.image, self.traversal_range, self.traversal_resolution,
            std_threshold=None, output_format='traversals_first')
        for index, traversal in traversals.items():
            traversal = traversal.squeeze()  # Remove image sample dimension
            traversal = deprocess_img(traversal)
            base = '{}-{:0>2}.gif'.format(self.stem, index)
            save_file = self.output_dir / base
            imageio.mimwrite(save_file.as_posix(), traversal)


class LatentDistributionLogging(Callback):
    """
    Parameters
    ----------
    csv_file : str or pathlib.Path
        The path of the CSV file where the average latent means and standard
        distributions are logged following each training epoch.
    data : np.ndarray or tcvae.data.ImageDataGenerator
        A batch of images whose latent distributions are averaged following
        each training epoch.
    """

    def __init__(self, csv_file, data, batch_size=512):
        super().__init__()
        self.csv_file = check_path(csv_file, path_type=Path)
        self.data = data
        self.batch_size = batch_size

        self.encoder = None
        self.num_latents = None
        self.csv_header = None
        self.batch_means = None
        self.batch_stds = None

    def on_train_begin(self, logs=None):
        self.encoder = self.model.get_layer('encoder')
        z, z_mu, z_log_sigma = self.encoder.outputs
        self.num_latents = int(z.shape[-1])
        if not self.csv_file.is_file():
            mean_header = [
                'z_mu_{:0>2}'.format(i) for i in range(self.num_latents)]
            std_header = [
                'z_sigma_{:0>2}'.format(i) for i in range(self.num_latents)]
            csv_header = mean_header + std_header
            with open(self.csv_file.as_posix(), 'at') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)

    def on_epoch_end(self, epoch, logs=None):
        if isinstance(self.data, np.ndarray):
            _, z_mu, z_log_sigma = self.encoder.predict(
                self.data, batch_size=self.batch_size)
        elif isinstance(self.data, ImageDataGenerator): 
            _, z_mu, z_log_sigma = self.encoder.predict_generator(
                self.data)
        else:
            raise ValueError(
                'Argument for parameter `data` must be a Numpy array or '
                'ImageDataGenerator.')
        z_sigma = np.exp(z_log_sigma)
        z_mu_epoch = z_mu.mean(axis=0).tolist()
        z_sigma_epoch = z_sigma.mean(axis=0).tolist()
        row = z_mu_epoch + z_sigma_epoch
        with open(self.csv_file.as_posix(), 'at') as f:
            writer = csv.writer(f)
            writer.writerow(row)


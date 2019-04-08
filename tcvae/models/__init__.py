#!/usr/bin/env python

"""
This module contains utilities for constructing, saving, and loading models.
"""


import pickle as pkl
from pathlib import Path

import numpy as np
from keras.models import Model, load_model

from tcvae.utils import (
    unpack_tensors, check_compatibility, check_path, make_directory)
from tcvae.inference import tile_multi_image_traversal


class TCVAE:
    """
    The TCVAE class acts as a wrapper around encoder and decoder models
    defined with Keras. It packages customized loss functions with the
    underlying models and provides convenience functions for model training,
    inference, and latent traversal generation.

    Parameters
    ----------
    encoder : keras.model.Model
        A Keras model that transforms batches of images into their sampled
        latent encodings, latent means, and latent variances (in the form of
        the log of the standard deviation).
    decoder : keras.model.Model
        A Keras model that transforms latent encodings into reconstructions
        of the encoder's original inputs.
    loss_dict : dict
        A dictionary where the keys consist of loss functions and
        corresponding values are floats indicating how the loss functions
        should be weighted.
    """

    # TODO: Add attributes to class documentation

    def __init__(self, encoder, decoder, loss_dict):
        self.encoder = encoder
        self.decoder = decoder
        self.loss_dict = loss_dict  # For saving

        models = _make_autoencoder_models(self.encoder, self.decoder)
        self.model_train, self.model_predict, self.tensors = models
        self.loss, self.metrics = _make_loss_and_metrics(
            loss_dict, self.tensors)
        self.num_latents = int(self.tensors['z'].shape[-1])

    def compile(self, optimizer, **kwargs):
        """
        Configures the model for training.

        Parameters
        ----------
        optimizer : keras.optimizers.Optimizer or str
            See Keras documentation for more details
        """
        self.model_train.compile(
            optimizer=optimizer, loss=self.loss, metrics=self.metrics, **kwargs)

    def save(
            self, save_dir, encoder_stem='encoder', decoder_stem='decoder',
            losses_stem='losses', overwrite=False):
        """
        Saves model resources into a user-specified directory.

        Parameters
        ----------
        save_dir : str or pathlib.Path
            The directory where model resources will be saved.
        encoder_stem : str
            The name of the saved encoder file without file extension.
        decoder_stem : str
            The name of the saved decoder file without file extension.
        losses_stem : str
            The name of the saved loss dictionary without file extension.
        overwrite : bool
            Whether to overwrite an already existing directory.
        """
        make_directory(save_dir, overwrite=overwrite)
        encoder_file, decoder_file, loss_file = _process_stems(
            save_dir, encoder_stem, decoder_stem, losses_stem)

        self.encoder.save(encoder_file)
        self.decoder.save(decoder_file)
        with open(loss_file, 'wb') as f:
            pkl.dump(self.loss_dict, f)


    @classmethod
    def load(
            cls, save_dir, encoder_stem='encoder', decoder_stem='decoder',
            losses_stem='losses'):
        """
        Loads model resources from a user-specified directory.

        Parameters
        ----------
        save_dir : str or pathlib.Path
            The directory where model resources will be saved.
        encoder_stem : str
            The name of the saved encoder file without file extension.
        decoder_stem : str
            The name of the saved decoder file without file extension.
        losses_stem : str
            The name of the saved loss dictionary without file extension.

        Returns
        -------
        model : tcvae.models.TCVAE
            An uncompiled TCVAE model.
        """

        encoder_file, decoder_file, loss_file = _process_stems(
            save_dir, encoder_stem, decoder_stem, losses_stem)
        encoder = load_model(encoder_file)
        decoder = load_model(decoder_file)
        with open(loss_file, 'rb') as f:
            loss_dict = pkl.load(f)
        model = cls(encoder, decoder, loss_dict)
        return model


    def encode(self, x, batch_size=32):
        """
        Transform images into their latent distributions.

        Parameters
        ----------
        x : np.ndarray, shape (num_samples, img_height, img_width, num_channels)
            An array of images.
        batch_size : int
            The number of images in each prediction batch.

        Returns
        -------
        z_mu : np.ndarray, shape (num_samples, num_latents)
            The latent means of the input images.
        z_sigma : np.ndarray, shape (num_samples, num_latents)
            The latent standard deviation of the input images.
        """
        _, z_mu, z_log_sigma = self.encoder.predict(x, batch_size=batch_size)
        z_sigma = np.exp(z_log_sigma)
        return z_mu, z_sigma

    def reconstruct(self, x, batch_size=32):
        """
        Transform images into their reconstructions.

        Parameters
        ----------
        x : np.ndarray, shape (num_samples, img_height, img_width, num_channels)
            An array of images.
        batch_size : int
            The number of images in each prediction batch.

        Returns
        -------
        y : np.ndarray, shape (num_samples, img_height, img_width, num_channels)
            An array or reconstructed images.
        """
        y = self.model_predict.predict(x, batch_size=batch_size)
        return y

    def decode(self, z, batch_size=32):
        """
        Transform latent encodings into image reconstructions.

        Parameters
        ----------
        z : np.ndarray, shape (num_samples, num_latents)
            The latent encodings of images.
        batch_size : int
            The number of images in each prediction batch.

        Returns
        -------
        y : np.ndarray, shape (num_samples, img_height, img_width, num_channels)
            An array or reconstructed images.
        """
        y = self.decoder.predict(z, batch_size=batch_size)
        return y

    def make_traversal(
            self, x, latent_index, traversal_range=(-4, 4),
            traversal_resolution=25, batch_size=32, output_format='tiled',
            num_rows=None):
        """
        A traversal of a specific component of a latent encoding involves
        interpolating that component across a range of values and generating
        reconstructions from each interpolation point.

        Parameters
        ----------
        x : np.ndarray, shape (num_samples, img_height, img_width, num_channels)
            An array of images.
        latent_index : int
            The index of the latent encoding on which to perform the
            traversal. Must be within the range [0, num_latents).
        traversal_range : tuple of floats, length 2
            The lower and upper bounds of the latent component interpolation.
        traversal_resolution : int
            The number of points in the latent component interpolation.
        batch_size : int
            The number of images in each prediction batch.
        output_format : str, one of {tiled, images_first, traversal_first}
            Specifies the format in which the traversals are returned.
            * tiled : For each traversal, multiple images are tiled
                together. The output shape is (traversal_resolution,
                num_rows * img_height, num_cols * img_width, num_channels).
            * images_first : The output shape is (num_samples,
                traversal_resolution, img_height, img_width, num_channels).
            * traversal_first :  The output shape is (traversal_resolution,
                num_samples, img_height, img_width, num_channels).
        num_rows : int or None
            The number of rows of images when multiple input images are tiled
            together. Only needs to be specified when
            `output_format='tiled'`.

        Returns
        -------
        latent_traversals : np.ndarray
            An array of latent traversals for each input image with format
            determined by the `output_format` parameter.
        """
        traversal_start, traversal_end = traversal_range
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
        elif output_format == 'tiled':
            latent_traversals = tile_multi_image_traversal(
                latent_traversals, num_rows)
        elif output_format == 'traversals_first':
            pass
        else:
            raise ValueError(
                'Argument for `output_format` must be one of the following '
                'strings: `images_first`, `traversal_first`, or `tiled`.')

        return latent_traversals

    def make_all_traversals(
            self, x, traversal_range=(-4, 4), traversal_resolution=25,
            batch_size=32, std_threshold=0.8, output_format='tiled',
            num_rows=None):
        """
        Performs traversals over all latent dimensions with high information
        capacity.

        Parameters
        ----------
        x : np.ndarray, shape (num_samples, img_height, img_width, num_channels)
            An array of images.
        traversal_range : tuple of floats, length 2
            The lower and upper bounds of the latent component interpolation.
        traversal_resolution : int
            The number of points in the latent component interpolation.
        batch_size : int
            The number of images in each prediction batch.
        std_threshold : float
            A number within the range [0, 1.0]. Latent dimensions with
            distributions whose standard deviation is above this threshold
            are not included in traversal generation as high standard
            deviations indicate the dimension does not encode sufficient
            information useful for producing reconstructions.
        output_format : str, one of {tiled, images_first, traversal_first}
            Specifies the format in which the traversals are returned.
            * tiled : For each traversal, multiple images are tiled
                together. The output shape is (traversal_resolution,
                num_rows * img_height, num_cols * img_width, num_channels).
            * images_first : The output shape is (num_samples,
                traversal_resolution, img_height, img_width, num_channels).
            * traversal_first :  The output shape is (traversal_resolution,
                num_samples, img_height, img_width, num_channels).
        num_rows : int or None
            The number of rows of images when multiple input images are tiled
            together. Only needs to be specified when
            `output_format='tiled'`.

        Returns
        -------
        traversal_dict : dict
            A dictionary of traversals for each latent component, with keys
            denoting the index of the latent component being traversed and
            values containing the corresponding traversal along that component.
        """

        # Perform thresholding so traversals are only performed on latent
        # components whose latent distribution has standard deviation less than
        # `std_threshold`
        z_mu, z_sigma = self.encoder(x, batch_size=batch_size)
        z_sigma = z_sigma.mean(axis=0)
        latent_indices = np.argwhere(z_sigma <= std_threshold).squeeze()

        # Traversals
        traversal_dict = dict.fromkeys(latent_indices)
        for latent_index in latent_indices:
            traversal_dict[latent_index] = self.make_traversal(
                latent_index=latent_index, traversal_range=traversal_range,
                traversal_resolution=traversal_resolution,
                batch_size=batch_size, output_format=output_format,
                num_rows=num_rows)
        return traversal_dict

    # TODO: compile, train, save


def _process_stems(
        save_dir, encoder_stem, decoder_stem, losses_stem):

    paths = [save_dir, encoder_stem, decoder_stem, losses_stem]
    save_dir, encoder_stem, decoder_stem, losses_stem = check_path(
        paths, path_type=Path)

    encoder_base = encoder_stem.with_suffix('.h5')
    decoder_base = decoder_stem.with_suffix('.h5')
    losses_base = losses_stem.with_suffix('.dict')
    bases = encoder_base, decoder_base, losses_base
    files = [save_dir / base for base in bases]
    encoder_file, decoder_file, loss_file = check_path(files, path_type=str)
    return encoder_file, decoder_file, loss_file


def _make_autoencoder_models(encoder, decoder):
    check_compatibility(encoder, decoder)
    tensor_dict = unpack_tensors(encoder, decoder)

    # Create VAE model for training
    model_train = Model(
        inputs=tensor_dict['x'], outputs=tensor_dict['y'],
        name='tcvae-train')

    # Create VAE model for inference
    model_predict = Model(
        inputs=tensor_dict['x'], outputs=tensor_dict['y_pred'],
        name='tcvae-predict')
    return model_train, model_predict, tensor_dict


def _make_loss_and_metrics(loss_dict, tensor_dict):
    # Convert loss functions to loss tensors
    loss_tensor_dict = {
        loss_fn(**tensor_dict): coefficient
        for loss_fn, coefficient
        in loss_dict.items()}

    # Convert loss tensors to Keras-compatible loss functions
    loss_names = [loss_fn.__name__ for loss_fn in loss_dict.keys()]
    loss_closure_dict = {
        _convert_to_closure(loss_tensor, loss_name): coefficient
        for loss_name, (loss_tensor, coefficient)
        in zip(loss_names, loss_tensor_dict.items())}

    # Total loss
    total_loss_fn = _make_total_loss_fn(loss_closure_dict)
    metrics = list(loss_closure_dict.keys())
    return total_loss_fn, metrics


def _convert_to_closure(loss_tensor, loss_name):
    def keras_loss_fn(x, y):
        return loss_tensor
    keras_loss_fn.__name__ = loss_name
    return keras_loss_fn


def _make_total_loss_fn(loss_dict):
    def total_loss_fn(x, y):
        loss = 0
        for loss_fn, coefficient in loss_dict.items():
            loss += coefficient * loss_fn(x, y)
        return loss
    return total_loss_fn

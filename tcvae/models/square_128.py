#!/usr/bin/env python

"""
This module defines common VAE encoder and decoder architectures for 128x128
images.
"""


import numpy as np
from keras.models import Model
from keras.layers import (
    Conv2D, Conv2DTranspose, BatchNormalization, Reshape, Input)

from tcvae.layers import Variational


def _check_iterable_arg(arg, expected_length):
    """
    Checks if a numeric argument is an iterable of a predefined length.
    """
    numeric_types = int, float
    iterable_types = np.ndarray, list, tuple
    if isinstance(arg, numeric_types):
        arg = [arg] * expected_length
    elif isinstance(arg, iterable_types):
        assert(len(arg) == expected_length), (
            'Expected an iterable of length {} but got {} instead.'.format(
                expected_length, len(arg)))
    else:
        raise ValueError(
            'Argument must be of type int, float, np.ndarray, list, or tuple.')
    return arg


def make_encoder_7_convs(
        filters=(32, 64, 128, 128, 256, 512), num_latents=32, num_channels=3,
        activation='relu', batch_normalization=True, bn_momentum=0.1,
        bn_epsilon=1e-5):
    """
    Creates an all-convolutional encoder model that converts input tensors
    into latent tensors. The transformation from input tensor to either a
    mean tensor or variance tensor involves 7 convolutions each, with the
    first 6 shared between the mean and variance tensor.

    Parameters
    ----------
    filters : list or tuple of ints, length 6
        The number of filters each shared convolutional layer uses.
    num_latents : int
        The number of filters in the final convolutional layers whose outputs
        are the latent mean and variance tensors.
    num_channels : int
        The number of color channels in the input image.
    activation : str
        The activation function applied by the shared convolutional layers.
        Must be a Keras-recognizable string, such as `relu`, `elu`, `tanh`,
        or `linear`.
    batch_normalization : bool
        Whether to apply batch normalization following each shared
        convolutional layer.
    bn_momentum : float or list of floats with length 6
        The batch normalization momentum term. See Keras' BatchNormalization
        layer documentation for more details.
    bn_epsilon : float or list of floats with length 6
        The batch normalization momentum term. See Keras' BatchNormalization
        layer documentation for more details.

    Returns
    -------
    encoder : keras.model.Model
        An encoder model that converts images into latent
        distributions/samples.
    """

    # Check arguments
    num_shared_convs = 6
    filters = _check_iterable_arg(filters, num_shared_convs)
    if batch_normalization is True:
        bn_momentums = _check_iterable_arg(bn_momentum, num_shared_convs)
        bn_epsilons = _check_iterable_arg(bn_epsilon, num_shared_convs)
    activations = [activation] * (num_shared_convs-1) + ['linear']
    paddings = ['same'] * (num_shared_convs-1) + ['valid']
    strides = [2] * (num_shared_convs-1) + [1]

    input_ = Input(shape=(128, 128, num_channels), name='x')
    x = input_
    for idx in range(num_shared_convs):
        conv_layer = Conv2D(
            filters[idx], 4, strides=strides[idx], padding=paddings[idx],
            activation=activations[idx])
        x = conv_layer(x)
        if batch_normalization is True:
            bn_layer = BatchNormalization(
                axis=-1, momentum=bn_momentums[idx],
                epsilon=bn_epsilons[idx])
            x = bn_layer(x)
    z_mu = Conv2D(num_latents, 1)(x)
    z_mu = Reshape((num_latents,))(z_mu)
    z_log_sigma = Conv2D(num_latents, 1)(x)
    z_log_sigma = Reshape((num_latents,))(z_log_sigma)
    z = Variational()([z_mu, z_log_sigma])
    encoder = Model(inputs=input_, outputs=[z, z_mu, z_log_sigma], name='encoder')
    return encoder


def make_decoder_7_deconvs(
        filters=(512, 256, 128, 128, 64, 32), num_latents=32, num_channels=3,
        activation='relu', batch_normalization=True, bn_momentum=0.1,
        bn_epsilon=1e-5):
    """
    Creates an all-convolutional decoder model that converts the latent
    tensor into reconstruction tensors. The transformation from latent tensor to
    reconstruction involves 7 convolutions.

    Parameters
    ----------
    filters : list or tuple of ints, length 6
        The number of filters each shared convolutional layer uses.
    num_latents : int
        The number of filters in the final convolutional layers whose outputs
        are the latent mean and variance tensors.
    num_channels : int
        The number of filters in the final convolutional layer whose output
        is the reconstruction. Typically 3 for color images.
    activation : str
        The activation function applied by the internal convolutional layers.
        Must be a Keras-recognizable string, such as `relu`, `elu`, `tanh`,
        or `linear`.
    batch_normalization : bool
        Whether to apply batch normalization following each internal
        convolutional layer.
    bn_momentum : float or list of floats with length 6
        The batch normalization momentum term. See Keras' BatchNormalization
        layer documentation for more details.
    bn_epsilon : float or list of floats with length 6
        The batch normalization momentum term. See Keras' BatchNormalization
        layer documentation for more details.

    Returns
    -------
    decoder : keras.model.Model
        A decoder model that converts sampled latent vectors into image
        reconstructions.
    """
    # Check arguments
    num_internal_deconvs = 6
    filters = _check_iterable_arg(filters, num_internal_deconvs)
    if batch_normalization is True:
        bn_momentums = _check_iterable_arg(bn_momentum, num_internal_deconvs)
        bn_epsilons = _check_iterable_arg(bn_epsilon, num_internal_deconvs)
    activations = [activation] * num_internal_deconvs
    paddings = ['valid'] * 2 + ['same'] * 4
    strides = [1] * 2 + [2] * 4
    filter_sizes = [1] + [4] * 5

    input_ = Input(shape=(num_latents,))
    z = Reshape((1, 1, -1))(input_)
    for idx in range(num_internal_deconvs):
        conv_layer = Conv2DTranspose(
            filters[idx], kernel_size=filter_sizes[idx],
            strides=strides[idx], padding=paddings[idx],
            activation=activations[idx])
        z = conv_layer(z)
        if batch_normalization is True:
            bn_layer = BatchNormalization(
                axis=-1, momentum=bn_momentums[idx],
                epsilon=bn_epsilons[idx])
            z = bn_layer(z)
    y = Conv2DTranspose(
        num_channels, 4, strides=2, padding='same', activation='sigmoid')(z)
    decoder = Model(inputs=input_, outputs=y, name='decoder')
    return decoder

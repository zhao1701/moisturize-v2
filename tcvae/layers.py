#!/usr/bin/env python

"""
This module contains custom Keras-style layers for building variational
autoencoders.
"""


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Add, Multiply


class Variational(Layer):
    """
    This Keras-style layer performs the reparameterization trick to sample
    a latent vector (z) from a multi-dimensional mean (mu) and a
    multi-dimensional standard deviation (sigma) with no covariance:

        z = mu + epsilon * sigma

    where epsilon is sampled from a standard multivariate Gaussian.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, *args, **kwargs):
        """
        Samples a latent vector from a multi-dimensional Gaussian distribution.

        Parameters
        ----------
        x : list of tensors
            A list consisting of 2 tensors representing the latent
            distributions output within an encoder network: z_mu and
            z_log_sigma. Both tensors must be of rank 2 where the first axis
            indexes the data sample and the second indexes the latent dimension.

        Returns
        -------
        z : tensor
            A rank-2 tensor representing a random sample from a
            multivariate Gaussian distribution parametrized by the
            input tensors <x>. The first axis indexes the data sample and
            the second indexes the latent dimension.
        """
        assert(isinstance(x, list))
        z_mu, z_log_sigma = x
        eps = K.random_normal(K.shape(z_log_sigma))
        z = Add()([z_mu, Multiply()([K.exp(z_log_sigma), eps])])
        return z

    def compute_output_shape(self, input_shape):
        assert(isinstance(input_shape, list))
        z_mu_shape, z_log_sigma_shape = input_shape
        assert (z_mu_shape == z_log_sigma_shape)
        return z_mu_shape

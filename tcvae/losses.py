#!/usr/bin/env python

"""
This module contains various loss functions that can be added to a VAE.
"""


from keras import backend as K


def kl_divergence(z_mu, z_log_sigma):
    """
    Computes the Kullback-Leibler divergence between the latent Gaussian
    distributions and a standard multivariate Gaussian.

    Parameters
    ----------
    z_mu : tensor, shape (None, num_latents)
        A rank-2 tensor representing a batch of latent mean vectors.
    z_log_sigma : tensor, shape (None, num_latents)
        A rank-2 tensor representing a batch of latent variance vectors,
        though the log of the standard deviation is used instead for
        numerical stability.

    Returns
    -------
    kl_divergence_ : tensor, rank 0
        A scalar tensor representing the mean KL-divergence between the
        latent distributions and a standard multivariate Gaussian.
    """
    temp = 1 + 2*z_log_sigma - K.square(z_mu) - K.exp(2*z_log_sigma)
    temp = -0.5 * K.sum(temp, axis=-1)
    kl_divergence_ = K.mean(temp)
    return kl_divergence_


def sum_squared_error(y_true, y_pred):
    """
    Computes the average SSE between input image and reconstruction over
    batches of image data. In other words, for each image and its
    corresponding reconstruction, an SSE is calculated, and the SSEs of all
    image-reconstruction pairs within a batch are averaged.

    Parameters
    ----------
    y_true : tensor, shape (num_samples, height, width, num_channels)
        A batch of input images.
    y_pred : tensor, shape (num_samples, height, width, num_channels)
        A batch of corresponding image reconstructions.

    Returns
    -------
    mean_sse : tensor, rank 0
        The average sum of squared errors for images and reconstructions
        within a batch.
    """
    temp = K.square(y_true - y_pred)
    # Sum squared errors of channels, width, and height
    for i in range(3):
        temp = K.sum(temp, axis=-1)
    mean_sse = K.mean(temp)
    return mean_sse

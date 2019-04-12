#!/usr/bin/env python

"""
This module contains various loss functions that can be added to a VAE.
"""


from keras import backend as K


def kl_divergence(z_mu, z_log_sigma, **_):
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


def sum_squared_error(x, y, **_):
    """
    Computes the average SSE between input image and reconstruction over
    batches of image data. In other words, for each image and its
    corresponding reconstruction, an SSE is calculated, and the SSEs of all
    image-reconstruction pairs within a batch are averaged.

    Parameters
    ----------
    x : tensor, shape (num_samples, height, width, num_channels)
        A batch of input images.
    y : tensor, shape (num_samples, height, width, num_channels)
        A batch of corresponding image reconstructions.

    Returns
    -------
    mean_sse : tensor, rank 0
        The average sum of squared errors for images and reconstructions
        within a batch.
    """
    temp = K.square(x - y)
    # Sum squared errors of channels, width, and height
    for i in range(3):
        temp = K.sum(temp, axis=-1)
    mean_sse = K.mean(temp)
    return mean_sse


STR_TO_LOSS_FUNC = dict(
    sum_squared_error=sum_squared_error,
    kl_divergence=kl_divergence,
)


def convert_loss_dict_keys(loss_dict):
    loss_dict = {
        STR_TO_LOSS_FUNC[loss_str]: coef
        for (loss_str, coef) in loss_dict.items()}
    return loss_dict

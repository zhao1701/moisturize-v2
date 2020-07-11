#!/usr/bin/env python

"""
This module contains various loss functions that can be added to a VAE.
"""
from inspect import signature

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


### RECONSTRUCTION LOSS FUNCTIONS ###


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


def sum_absolute_error(x, y, **_):
    """
    Computes the average SAE between input image and reconstruction over
    batches of image data. In other words, for each image and its
    corresponding reconstruction, an SAE is calculated, and the SAEs of all
    image-reconstruction pairs within a batch are averaged.

    Parameters
    ----------
    x : tensor, shape (num_samples, height, width, num_channels)
        A batch of input images.
    y : tensor, shape (num_samples, height, width, num_channels)
        A batch of corresponding image reconstructions.

    Returns
    -------
    mean_sae : tensor, rank 0
        The average sum of absolute errors for images and reconstructions
        within a batch.
    """
    temp = K.abs(x - y)
    # Sum absolute errors of channels, width, and height
    for i in range(3):
        temp = K.sum(temp, axis=-1)
    mean_sae = K.mean(temp)
    return mean_sae


def sigmoid_cross_entropy(x, y, **_):
    """
    Computes the sigmoid cross entropy for each pixel and its
    reconstruction, sums these values for all pixels within an image, 
    then averages these sums.
    
    Parameters
    ----------
    x : tensor, shape (num_samples, height, width, num_channels)
        A batch of input images.
    y : tensor, shape (num_samples, height, width, num_channels)
        A batch of corresponding image reconstructions.

    Returns
    -------
    mean_sce : tensor, rank 0
        The average sum sigmoid cross entropies for images and
        reconstructions within a batch.
    """
    temp = K.binary_crossentropy(x, y)
    # Sum binary crossentropy of channels, width, and height
    for i in range(3):
        temp = K.sum(temp, axis=-1)
    mean_sce = K.mean (temp)
    return mean_sce


def dssim(x, y, **_):
    ms_ssim = tf.image.ssim_multiscale(
        x, y, max_val=1, power_factors=(0.0448, 0.2856, 0.3001, 0.2363))
    ms_ssim = K.mean(ms_ssim)
    dssim = 1 - ms_ssim
    
    # Make DSSIM comparable to other reconstruction metrics
    num_pixels = int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3])
    dssim = dssim * num_pixels
    return dssim


### VAE LATENT LOSS FUNCTIONS ###


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


### TCVAE LATENT LOSS FUNCTIONS ###

    
def mutual_information_index(z, z_mu, z_log_sigma, logiw_matrix, **_):

    logqz_condx = _calc_logqz_condx(z, z_mu, z_log_sigma)
    logqz = _calc_logqz(z, z_mu, z_log_sigma, logiw_matrix)

    # This tensor corresponds to but is not equivalent to the mi_index term
    # of the decomposed divergence penalty. However, minimizing this
    # serves to minimize the true penalty term.
    mi_index = K.mean(logqz_condx - logqz)
    return mi_index

    
def total_correlation(z, z_mu, z_log_sigma, logiw_matrix, **_):

    logqz = _calc_logqz(z, z_mu, z_log_sigma, logiw_matrix)
    logqz_prod_marginals = _calc_logqz_prod_marginals(
        z, z_mu, z_log_sigma, logiw_matrix)

    # This tensor corresponds to but is not equivalent to the total
    # correlation term of the decomposed divergence penalty. However,
    # minimizing this serves to minimize the true penalty term.
    tc = K.mean(logqz - logqz_prod_marginals)
    return tc

    
def dimensional_kl(z, z_mu, z_log_sigma, logiw_matrix, **_):

    logpz = _calc_logpz(z, z_mu, z_log_sigma)
    logqz_prod_marginals = _calc_logqz_prod_marginals(
        z, z_mu, z_log_sigma, logiw_matrix)

    # This tensor corresponds to but is not equivalent to the dimensional
    # KL term of the decomposed divergence penalty. However,
    # minimizing this serves to minimize the true penalty term. 

    dim_kl = K.mean(logqz_prod_marginals - logpz)
    return dim_kl


### HELPER FUNCTIONS FOR TCVAE LOSSES ###


def _calc_logqz_condx(z, z_mu, z_log_sigma, **_):
    # Calculate log densities for sampled latent vectors given the
    # distributions. Generated by the encoder network from the input
    # images (aka q(z|x)).
    logqz_condx = _calc_gaussian_log_density(
        z, z_mu, z_log_sigma)
    logqz_condx = K.sum(logqz_condx, axis=1)
    return logqz_condx


def _calc_logqz(z, z_mu, z_log_sigma, logiw_matrix, **_):
    # Calculate the log densities from the aggregate latent posterior
    # distribution q(z).
    # log q(z) ~= log 1 /(NM) sum_m=1^M q(z|x_m)
    # = - log(MN) + logsumexp_m(q(z|x_m))
    _logqz = _calc_gaussian_log_density(
        K.reshape(z, shape=(-1, 1, int(z.shape[1]))),
        K.reshape(z_mu, shape=(1, -1, int(z_mu.shape[1]))),
        K.reshape(z_log_sigma, shape=(1, -1, int(z_log_sigma.shape[1]))))

    # Estimate log[q(z)]
    logqz = _logsumexp(
        logiw_matrix + K.sum(_logqz, axis=2), axis=1, keepdims=False)
    return logqz


def _calc_logqz_prod_marginals(z, z_mu, z_log_sigma, logiw_matrix, **_):
    # Calculate the log densities from the aggregate latent posterior
    # distribution q(z).
    # log q(z) ~= log 1 /(NM) sum_m=1^M q(z|x_m)
    # = - log(MN) + logsumexp_m(q(z|x_m))
    _logqz = _calc_gaussian_log_density(
        K.reshape(z, shape=(-1, 1, int(z.shape[1]))),
        K.reshape(z_mu, shape=(1, -1, int(z_mu.shape[1]))),
        K.reshape(z_log_sigma, shape=(1, -1, int(z_log_sigma.shape[1]))))
    
    # Estimate log[prod_j[q(z_j)]]
    logqz_prod_marginals = _logsumexp(
        K.expand_dims(logiw_matrix, axis=-1) + _logqz,
        axis=1, keepdims=False)
    logqz_prod_marginals = K.sum(logqz_prod_marginals, axis=1)
    return logqz_prod_marginals


def _calc_logpz(z, z_mu, z_log_sigma, **_):
    # Calculate log densities for sampled latent vectors in a standard
    # Gaussian distribution (zero mean, unit variance)
    logpz = _calc_gaussian_log_density(
      z, K.zeros_like(z_mu), K.zeros_like(z_log_sigma))
    logpz = K.sum(logpz, axis=1)
    return logpz


def _calc_gaussian_log_density(data, mu, log_sigma):
    """
    Given data samples and their distributions, calculate each samples' log
    density. Gaussian probability density function:

      p(x) = (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * (x - mu)/sigma)^2)

    Then:

      log[p(x)] = -log[sigma] - 0.5 * log[2 * pi] - 0.5 * [(x - mu)/sigma]^2
      log[p(x)] = -0.5 * [(x - mu / sigma)^2 + 2 * log[sigma] + log[2 * pi]]

    Let:

      c = log[2 * pi]
      inv_sigma = 1 / sigma
      tmp = (x - mu) / sigma = (x - mu) * inv_sigma

    Then:

      log[p(x)] = -0.5 * [tmp * tmp + 2 * log[sigma] + c]
    """
    
    c = np.log(2 * np.pi)
    inv_sigma = K.exp(-log_sigma)
    tmp = (data - mu) * inv_sigma
    log_density = -0.5 * (tmp*tmp + 2*log_sigma + c)
    return log_density


def _logsumexp(value, axis, keepdims=False):
    """
    A numerically stable computation for chaining the operations: log, sum, and
    exp.
    
        log[sum_i(exp(x_i))]
        = m - m + log[sum_i(exp(x_i))]
        = m + log[1/exp(m)] + log[sum_i(exp(x_i))]
        = m + log[(1/exp(m))*sum_i(exp(x_i))]
        = m + log[sum_i(exp(x_i)/exp(m))]
        = m + log[sum_i(exp(x_i-m))]
    """
    
    m = K.max(value, axis=axis, keepdims=True)
    _value = value - m
    if not keepdims:
        m = K.squeeze(m, axis=axis)
    result = m + K.log(K.sum(K.exp(_value), axis=axis, keepdims=keepdims))
    return result


def _log_importance_weight_matrix(batch_size, dataset_size, as_tensor=False):
    """
    Returns a weight matrix that is used to estimate the unconditional latent
    distribution q(z).
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = np.empty([batch_size, batch_size])
    W.fill(1 / M)
    W[:, 0] = 1 / N
    W[:, 1] = strat_weight
    W[M-1, 0] = strat_weight
    W = np.log(W)
    if as_tensor:
        W = K.constant(W, dtype='float32')
    return W


### UTILS ###


LOSSES = [
    kl_divergence, sum_squared_error, sum_absolute_error,
    sigmoid_cross_entropy, mutual_information_index, total_correlation,
    dimensional_kl, dssim]


def _check_loss_fn(loss_fn, batch_size, dataset_size):
    """
    Converts TCVAE latent loss functions into a format similar to that
    of reconstruction and VAE latent loss functions.
    """
    fn_signature = tuple(signature(loss_fn).parameters.keys())
    if 'batch_size' in fn_signature or 'dataset_size' in fn_signature:
        if batch_size is None or dataset_size is None:
            raise ValueError(
                '`batch_size` and `dataset_size` must be specified when '
                'using a TCVAE-style latent loss function.')
        return loss_fn(batch_size, dataset_size)
    else:
        return loss_fn


def convert_loss_dict_keys(loss_dict):
    """
    Creates a dictionary of loss functions that specifies their weighting.
    
    Parameters
    ----------
    loss_dict : dict
        A dictionary whose keys consist of strings of loss function names
        and whose values consist of floats specifying the weight of the
        corresponding loss function.
    
    Returns
    -------
    loss_dict: dict
        A dictionary whose keys consist of loss functions and whose values
        consist of floats specifying the weight of the corresponding loss
        function.
    """
    str_to_loss_fn = {loss_fn.__name__: loss_fn for loss_fn in LOSSES}
    loss_dict = {
        str_to_loss_fn[loss_str]: coef
        for (loss_str, coef) in loss_dict.items()}
    return loss_dict

#!/usr/bin/env python

"""
This module contains utilities for constructing, saving, and loading models.
"""


import os
import sys

from keras.models import Model

sys.path.append(os.path.dirname(__file__))
from tcvae.utils import unpack_tensors, check_compatibility
from tcvae.losses import kl_divergence, sum_squared_error


class TCVAE:

    def __init__(self, encoder, decoder, loss_dict, optimizer):

        self.encoder = encoder
        self.decoder = decoder

        models = _make_autoencoder_models(self.encoder, self.decoder)
        self.model_train, self.model_predict, self.tensor_dict = models
        self.loss, self.metrics = _make_loss_and_metrics(
            loss_dict, self.tensor_dict)


def _make_autoencoder_models(encoder, decoder):
    check_compatibility(encoder, decoder)
    tensor_dict = unpack_tensors(encoder, decoder)

    # Create VAE model for training
    model_train = Model(
        inputs=tensor_dict['x'], outputs=tensor_dict['y'],
        name='vae-train')

    # Create VAE model for inference
    model_predict = Model(
        inputs=tensor_dict['x'], outputs=tensor_dict['y_pred'],
        name='vae-predict')
    return model_train, model_predict, tensor_dict


def _make_loss_and_metrics(loss_dict, tensor_dict):
    # Convert loss functions to loss tensors
    loss_tensor_dict = {
        loss_fn(**tensor_dict):coefficient
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

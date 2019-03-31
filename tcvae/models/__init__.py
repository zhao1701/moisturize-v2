#!/usr/bin/env python

"""
This module contains utilities for constructing, saving, and loading models.
"""


import os
import sys

from keras.models import Model

from tcvae.losses import kl_divergence, sum_squared_error


def make_autoencoder_model(encoder, decoder):

    # Check encoder and decoder are compatible
    assert(encoder.input_shape == decoder.output_shape), (
        'Encoder input shapes and decoder output shapes must be the same.')
    assert(encoder.output_shape[0][-1] == decoder.input_shape[-1]), (
        'The number of latent dimensions the encoder outputs is different from '
        'what the decoder expects.')

    x = encoder.inputs[0]
    y = decoder(encoder(x)[0])
    vae = Model(inputs=x, outputs=y, name='vae')
    return vae

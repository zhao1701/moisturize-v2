#!/usr/bin/env python


import pytest
from keras.preprocessing.image import ImageDataGenerator

from tcvae.models import TCVAE
from tcvae.losses import kl_divergence, sum_squared_error
from tcvae.models.square_128 import (
    make_encoder_7_convs, make_decoder_7_deconvs)


DATA_DIR = 'data'


@pytest.fixture()
def datagen():
    img_datagen = ImageDataGenerator()
    datagen = img_datagen.flow_from_directory(DATA_DIR)
    return datagen


@pytest.fixture()
def model():
    encoder = make_encoder_7_convs()
    decoder = make_decoder_7_deconvs()
    loss_dict = {
        sum_squared_error: 1.0,
        kl_divergence: 1.0}
    model = TCVAE(encoder, decoder, loss_dict)
    return model


def test_constructor(model):
    model
    assert(model.num_latents == 32)

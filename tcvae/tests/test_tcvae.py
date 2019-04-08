#!/usr/bin/env python

import warnings

import pytest
from keras.preprocessing.image import ImageDataGenerator

from tcvae.models import TCVAE
from tcvae.losses import kl_divergence, sum_squared_error
from tcvae.models.square_128 import (
    make_encoder_7_convs, make_decoder_7_deconvs)


warnings.filterwarnings('ignore')
DATA_DIR = 'data'
SAVE_DIR = 'test-model'


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


def test_compile(model):
    model.compile('adam')


def test_save(model):
    model.save(SAVE_DIR, overwrite=True) 


def test_load():
    model = TCVAE.load(SAVE_DIR)

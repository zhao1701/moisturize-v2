#!/usr/bin/env python

import os
import warnings
from glob import glob

import pytest

from tcvae.models import TCVAE
from tcvae.data import ImageDataGenerator
from tcvae.losses import kl_divergence, sum_squared_error
from tcvae.models.square_128 import (
    make_encoder_7_convs, make_decoder_7_deconvs)


warnings.filterwarnings('ignore')
DATA_DIR = 'data'
SAVE_DIR = 'test-model'
NUM_IMAGES = len(glob(os.path.join(DATA_DIR, '*')))
NUM_LATENTS = 32
IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS = 128, 128, 3


@pytest.fixture()
def datagen():
    datagen = ImageDataGenerator(DATA_DIR)
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


def test_fit_generator(model, datagen):
    model.compile('adam')
    model.fit_generator(datagen, epochs=2)


def test_encode_generator(model, datagen):
    z_mu, z_sigma = model.encode_generator(datagen)
    assert(z_mu.shape == z_sigma.shape)
    assert(z_mu.shape == (NUM_IMAGES, NUM_LATENTS))


def test_reconstruct_generator(model, datagen):
    y = model.reconstruct_generator(datagen)
    assert(y.shape == (NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))


#!/usr/bin/env python

import os
import warnings
from glob import glob

import pytest

from tcvae.models import TCVAE
from tcvae.tests.fixtures import datagen, model
from tcvae.tests.constants import (
    SAVE_DIR, NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_LATENTS)


warnings.filterwarnings('ignore')


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


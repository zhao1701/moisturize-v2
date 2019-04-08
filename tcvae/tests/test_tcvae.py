#!/usr/bin/env python

import os
from glob import glob

import pytest

from tcvae.models import TCVAE
from tcvae.tests.fixtures import data, datagen, model
from tcvae.tests.constants import (
    SAVE_DIR, NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_LATENTS,
    BATCH_SIZE)


def test_constructor(model):
    model
    assert(model.num_latents == 32)


def test_compile(model):
    model.compile('adam')


def test_save(model):
    model.save(SAVE_DIR, overwrite=True) 


def test_load():
    model = TCVAE.load(SAVE_DIR)


def test_fit(model, data):
    model.compile('adam')
    model.fit(data, data, batch_size=BATCH_SIZE)


def test_fit_generator(model, datagen):
    model.compile('adam')
    model.fit_generator(datagen, epochs=2)


def test_encode(model, data):
    z_mu, z_sigma = model.encode(data)
    assert(z_mu.shape == z_sigma.shape)
    assert(z_mu.shape == (NUM_IMAGES, NUM_LATENTS))


def test_encode_generator(model, datagen):
    z_mu, z_sigma = model.encode_generator(datagen)
    assert(z_mu.shape == z_sigma.shape)
    assert(z_mu.shape == (NUM_IMAGES, NUM_LATENTS))


def test_reconstruct(model, data):
    y = model.reconstruct(data)
    assert(y.shape == (NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))


def test_reconstruct_generator(model, datagen):
    y = model.reconstruct_generator(datagen)
    assert(y.shape == (NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))


def test_decode(model, datagen):
    z_mu, z_sigma = model.encode_generator(datagen)
    y = model.decode(z_mu)
    assert(y.shape == (NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))


def test_make_traversal(model, data, datagen):
    latent_index = 0
    traversal_resolution = 3
    num_rows = 8
    num_cols = NUM_IMAGES // num_rows
    traversals = model.make_traversal(
        data, latent_index, traversal_resolution=traversal_resolution,
        output_format='traversals_first', num_rows=num_rows)
    assert(traversals.shape == (
        traversal_resolution, NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    traversals = model.make_traversal(
        data, latent_index, traversal_resolution=traversal_resolution,
        output_format='images_first', num_rows=num_rows)
    assert(traversals.shape == (
        NUM_IMAGES, traversal_resolution, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    traversals = model.make_traversal(
        data, latent_index, traversal_resolution=traversal_resolution,
        output_format='tiled', num_rows=num_rows)
    assert(traversals.shape == (
        traversal_resolution, num_rows * IMG_HEIGHT, num_cols * IMG_WIDTH, NUM_CHANNELS))
    traversals = model.make_traversal(
        datagen, latent_index, traversal_resolution=traversal_resolution,
        output_format='traversals_first', num_rows=num_rows)


def test_make_all_traversals(model, data):
    data = data[:2]
    traversal_resolution = 3
    num_rows = 2
    num_cols = 2 // num_rows
    traversals = model.make_all_traversals(
        data, traversal_resolution=traversal_resolution,
        output_format='traversals_first', num_rows=num_rows,
        std_threshold=float('inf'))
    

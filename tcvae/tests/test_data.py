import numpy as np

from tcvae.tests.constants import (
    DATA_DIR, NUM_IMAGES, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
from tcvae.tests.fixtures import datagen


def test_constructor(datagen):
    datagen


def test_len(datagen):
    num_batches = int(np.ceil(NUM_IMAGES / BATCH_SIZE))
    assert(len(datagen) == num_batches)
    datagen.batch_size = 64
    num_batches = int(np.ceil(NUM_IMAGES / 64))
    assert(len(datagen) == num_batches)


def test_getitem(datagen):
    batch = datagen[0]
    assert(np.all(batch[0] == batch[0]))
    

def test_next(datagen):
    batch_shape = BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS
    for index, batch in enumerate(datagen):
        if index < len(datagen) - 1:
            assert(batch.shape == batch_shape)


def test_load_data(datagen):
    data = datagen.load_data()
    assert(data.shape == NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

import pytest

from tcvae.models import TCVAE
from tcvae.data import ImageDataGenerator
from tcvae.tests.constants import DATA_DIR
from tcvae.losses import kl_divergence, sum_squared_error
from tcvae.models.square_128 import (
    make_encoder_7_convs, make_decoder_7_deconvs)


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

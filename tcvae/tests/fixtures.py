import pytest

from tcvae.models import TCVAE
from tcvae.data import ImageDataGenerator
from tcvae.tests.constants import (
    DATA_DIR, BATCH_SIZE, RECONSTRUCTION_CHECK_DIR, TRAVERSAL_CHECK_DIR,
    DISTRIBUTION_LOGGING_CSV_FILE)
from tcvae.losses import kl_divergence, sum_squared_error
from tcvae.models.square_128 import (
    make_encoder_7_convs, make_decoder_7_deconvs)
from tcvae.callbacks import (
    ReconstructionCheck, LatentTraversalCheck, LatentDistributionLogging)

@pytest.fixture()
def latent_distribution_logging():
    datagen = ImageDataGenerator(DATA_DIR, batch_size=BATCH_SIZE)
    logger = LatentDistributionLogging(
        DISTRIBUTION_LOGGING_CSV_FILE, datagen)
    return logger


@pytest.fixture()
def reconstruction_check():
    datagen = ImageDataGenerator(DATA_DIR, batch_size=BATCH_SIZE)
    test_imgs, _ = datagen[0]
    test_imgs = test_imgs[:8]
    reconstruction_check = ReconstructionCheck(
        test_imgs, RECONSTRUCTION_CHECK_DIR)
    return reconstruction_check


@pytest.fixture()
def traversal_check():
    datagen = ImageDataGenerator(DATA_DIR, batch_size=BATCH_SIZE)
    test_imgs, _ = datagen[0]
    test_img = test_imgs[0]
    traversal_check = LatentTraversalCheck(
        test_img, TRAVERSAL_CHECK_DIR)
    return traversal_check


@pytest.fixture()
def datagen():
    datagen = ImageDataGenerator(DATA_DIR, batch_size=BATCH_SIZE)
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


@pytest.fixture()
def data():
    datagen = ImageDataGenerator(DATA_DIR, batch_size=BATCH_SIZE)
    data = datagen.load_data()
    return data


#!/usr/bin/env python

"""
This module contains custom callback functions to be used for model training.
"""

from pathlib import Path

import imageio
import numpy as np
from keras.callbacks import Callback
from tcvae.utils import check_path, make_directory
from tcvae.models import _make_autoencoder_models


# TODO: reconstruction check callback
class ReconstructionCheck(Callback):
    """
    Parameters
    ----------
    images : np.ndarray
        A batch of images to be reconstructed with shape (num_images,
        img_height, img_width, num_channels)
    output_dir : str or pathlib.Path
    """

    def __init__(
            self, images, output_dir, stem='reconstruction', format='jpg'):
        super().__init__()
        self.images = images
        self.output_dir = check_path(output_dir, Path)
        self.stem = stem
        if format[0] != '.':
            format = '.' + format
        self.format = format
        self.model_predict = None

    def on_train_begin(self, logs=None):
        make_directory(self.output_dir, overwrite=True)

    def on_epoch_end(self, epoch, logs=None):
        # Construct prediction model
        if self.model_predict is None:
            encoder = self.model.get_layer('encoder')
            decoder = self.model.get_layer('decoder')
            _, self.model_predict, _ = _make_autoencoder_models(
                encoder, decoder)
        # Make reconstructions and stitch with original image
        reconstructions = self.model_predict.predict(self.images)
        enum_zip = enumerate(zip(self.images, reconstructions))
        for index, (original, reconstruction) in enum_zip:
            img_pair = np.hstack([original, reconstruction])
            base = '{}-{:0>2}{}'.format(self.stem, index, self.format)
            save_file = self.output_dir / base
            imageio.imwrite(save_file.as_posix(), img_pair)


# TODO: latent traversal check callback

# TODO: latent distribution monitoring callback
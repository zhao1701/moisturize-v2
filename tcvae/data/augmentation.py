import functools as ft

import numpy as np
from skimage.transform import rotate, resize


class RandomRotator:

    def __init__(self, rotation_range=(-45, 45), modes=('edge', 'constant')):
        self.rotation_range = rotation_range
        self.modes = modes

    def __call__(self, img):
        angle = np.random.uniform(*self.rotation_range)
        mode = np.random.choice(self.modes)
        return rotate(img, angle=angle, mode=mode)


class RandomSquareCropper:

    def __init__(self, target_edge_length=128, zoom_range=(0.5, 1.0)):
        self.target_edge_length = target_edge_length
        self.zoom_range = zoom_range

    def get_random_crop_edge_length(self, img):
        """
        Randomly sample the length of the edge of the cropping square.
        """
        height, width, _ = img.shape
        crop_edge_max = min(height, width)
        zoom_factor = np.random.uniform(*self.zoom_range)
        crop_edge_length = int(zoom_factor * crop_edge_max)
        return crop_edge_length

    @staticmethod
    def get_random_crop_corners(img, crop_edge_length):
        height, width, _ = img.shape
        x_start_max = height - crop_edge_length
        y_start_max = width - crop_edge_length
        x_start = np.random.randint(0, x_start_max)
        y_start = np.random.randint(0, y_start_max)
        x_end = x_start + crop_edge_length
        y_end = y_start + crop_edge_length
        return x_start, x_end, y_start, y_end

    @staticmethod
    def apply_crop(img, x_start, x_end, y_start, y_end):
        return img[x_start:x_end, y_start:y_end, :]

    def __call__(self, img):
        crop_edge_length = self.get_random_crop_edge_length(img)
        corners = self.get_random_crop_corners(img, crop_edge_length)
        img_cropped = self.apply_crop(img, *corners)
        img_cropped = resize(
            img_cropped, (self.target_edge_length, self.target_edge_length))
        return img_cropped


class RandomFlipper:

    def __init__(self, horizontal=True, vertical=False):
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, img):
        flip_horizontal = np.random.choice([True, False])\
            if self.horizontal else False
        flip_vertical = np.random.choice(([True, False]))\
            if self.vertical else False
        if flip_horizontal:
            img = img[:, ::-1, :]
        if flip_vertical:
            img = img[::-1, :, :]
        return img


class RandomIntensifier:

    def __init__(self, shift_range=(-0.25, 0.25), scale_range=(0.5, 1.5)):
        self.shift_range = shift_range
        self.scale_range = scale_range

    @staticmethod
    def normalize_img(img):
        img = img - img.min()
        img = img / img.max()
        return img

    def sample_params(self):
        shift = np.random.uniform(*self.shift_range)
        scale = np.random.uniform(*self.scale_range)
        return shift, scale

    def __call__(self, img):
        img = self.normalize_img(img)
        shift, scale = self.sample_params()
        img = (img - shift) * scale
        img = img.clip(0, 1)
        return img


class RandomColorMixer:

    def __init__(
            self, shift_range=(-0.1, 0.1),
            scale_range=(0.9, 1.1)):
        self.shift_range = shift_range
        self.scale_range = scale_range

    def sample_params(self, n_channels):
        shifts = np.random.uniform(*self.shift_range, size=n_channels)
        scales = np.random.uniform(*self.scale_range, size=n_channels)
        return shifts, scales

    def __call__(self, img):
        _, _, n_channels = img.shape
        shifts, scales = self.sample_params(n_channels)
        img = (img - shifts) * scales
        img = img.clip(0, 1)
        return img


class TransformComposer:

    def __init__(self, *functions):
        self.functions = functions

    def __call__(self, img):
        for fn in self.functions:
            img = fn(img)
        return img

#!/usr/bin/env python

"""
This module contains helper functions used across multiple modules.
"""


import os
import sys
import shutil
from pathlib import Path

import git
import json
import yaml
import numpy as np


def unpack_tensors(encoder, decoder):
    x = encoder.inputs[0]
    tensor_dict = dict(
        x=x, z=encoder(x)[0], z_mu=encoder(x)[1],
        z_log_sigma=encoder(x)[2])
    tensor_dict['y'] = decoder(tensor_dict['z'])
    tensor_dict['y_pred'] = decoder(tensor_dict['z_mu'])
    return tensor_dict


def check_compatibility(encoder, decoder):
    # Check encoder and decoder are compatible
    assert(encoder.input_shape == decoder.output_shape), (
        'Encoder input shapes and decoder output shapes must be the same.')
    assert(encoder.output_shape[0][-1] == decoder.input_shape[-1]), (
        'The number of latent dimensions the encoder outputs is different from '
        'what the decoder expects.')


def check_path(path, path_type=str):
    if isinstance(path, path_type):
        return path
    elif isinstance(path, (list, tuple)):
        paths = path.copy()
        paths = [check_path(path, path_type=path_type) for path in paths]
        return paths
    elif path_type == str:
        path = path.as_posix()
        return path
    elif path_type == Path:
        path = Path(path)
        return path
    else:
        raise ValueError(
            'Path checking only supports pathlib.Path or str types.')


def make_directory(path, overwrite=False):
    path = check_path(path, path_type=str)
    if os.path.isdir(path):
        if overwrite is False:
            raise FileExistsError
        else:
            shutil.rmtree(path)
    os.makedirs(path)


def deprocess_img(img):
    img = img * 255
    img = img.astype(np.uint8)
    return img


def import_project_root():
    repo = git.Repo('.', search_parent_directories=True)
    project_root = os.path.dirname(repo.git_dir)
    sys.path.append(project_root)


def read_yaml(file):
    with open(file, 'rt') as f:
        yaml_dict = yaml.load(f)
    return yaml_dict


def read_json(file):
    with open(file, 'rt') as f:
        json_dict = json.load(f)
    return json_dict


def write_json(file, dictionary):
    with open(file, 'wt') as f:
        json.dump(dictionary, f)

#!/usr/bin/env python

"""
This module contains helper functions used across multiple modules.
"""


def unpack_tensors(encoder, decoder, inference=False):
    tensor_dict = dict(
        x=encoder.inputs[0], z=encoder(x)[0], z_mu=encoder(x)[1],
        z_log_sigma=encoder(x)[2])
    if inference is False:
        tensor_dict['y'] = decoder(tensor_dict['z'])
    else:
        tensor_dict['y'] = decoder(tensor_dict['z_mu'])
    return tensor_dict

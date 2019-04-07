#!/usr/bin/env python

"""
This module contains helper functions used across multiple modules.
"""


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
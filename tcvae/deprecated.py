def make_autoencoder_model(
        encoder, decoder, loss_dict, optimizer):

    # Check encoder and decoder are compatible
    assert(encoder.input_shape == decoder.output_shape), (
        'Encoder input shapes and decoder output shapes must be the same.')
    assert(encoder.output_shape[0][-1] == decoder.input_shape[-1]), (
        'The number of latent dimensions the encoder outputs is different from '
        'what the decoder expects.')

    # Unpack model tensors
    tensor_dict = unpack_tensors(encoder, decoder, inference=False)

    # Create VAE model
    vae = Model(
        inputs=tensor_dict['x'], outputs=tensor_dict['y'], name='vae')

    # Convert loss functions to loss tensors
    loss_tensor_dict = {
        loss_fn(**tensor_dict):coefficient
        for loss_fn, coefficient
        in loss_dict.items()}

    # Convert loss tensors to Keras-compatible loss functions
    loss_names = [loss_fn.__name__ for loss_fn in loss_dict.keys()]
    loss_closure_dict = {
        convert_to_closure(loss_tensor, loss_name): coefficient
        for loss_name, (loss_tensor, coefficient)
        in zip(loss_names, loss_tensor_dict.items())}

    # Total loss
    total_loss_fn = make_total_loss_fn(loss_closure_dict)
    metrics = list(loss_closure_dict.keys())

    vae.compile(loss=total_loss_fn, optimizer=optimizer, metrics=metrics)
    return vae
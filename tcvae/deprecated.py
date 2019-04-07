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


class Predictor:

    def __init__(self, model):

        if isinstance(model, str):
            model = load_model(model)
        elif isinstance(model, Path):
            model = load_model(model.as_posix())
        elif isinstance(model, Model):
            pass
        else:
            raise ValueError(
                'Argument for parameter `model` must be a Keras model or a '
                'path to one.')

        self.encoder = model.get_layer('encoder')
        self.decoder = model.get_layer('decoder')
        self.tensors = unpack_tensors(
            self.encoder, self.decoder)

        self.model = Model(
            inputs=self.tensors['x'], outputs=self.tensors['y_pred'],
            name='vae')

        self.num_latents = int(self.tensors['z'].shape[-1])

    def encode(self, x, batch_size=32):
        _, z_mu, z_log_sigma = self.encoder.predict(x, batch_size=batch_size)
        z_sigma = np.exp(z_log_sigma)
        return z_mu, z_sigma

    def reconstruct(self, x, batch_size=32):
        y = self.model.predict(x, batch_size=batch_size)
        return y

    def decode(self, z, batch_size=32):
        y = self.decoder.predict(z, batch_size=batch_size)
        return y

    def make_traversal(
            self, x, latent_index, traversal_start=-4, traversal_end=4,
            traversal_resolution=25, batch_size=32, output_format='stitched',
            num_rows=None, num_cols=None):

        z_mu, _ = self.encode(x, batch_size=batch_size)
        traversal_sequence = np.linspace(
            traversal_start, traversal_end, traversal_resolution)

        latent_traversals = np.empty(shape=(traversal_resolution,) + x.shape)
        for traversal_index, traversal_point in enumerate(traversal_sequence):
            z_mu_traversal = z_mu.copy()
            z_mu_traversal[:, latent_index] = traversal_point
            y_traversal = self.decode(z_mu_traversal)
            latent_traversals[traversal_index] = y_traversal

        if output_format == 'images_first':
            latent_traversals = latent_traversals.transpose(1, 0, 2, 3, 4)
        elif output_format == 'stitched':
            latent_traversals = stitch_multi_image_traversal(
                latent_traversals, num_rows, num_cols)
        elif output_format == 'traversal_first':
            pass
        else:
            raise ValueError(
                'Argument for `output_format` must be one of the following '
                'strings: `images_first`, `traversal_first`, or `stitched`.')

        return latent_traversals

    def make_all_traversals(
            self, x, traversal_start=-4, traversal_end=4,
            traversal_resolution=25, batch_size=32):
        # TODO: variance thresholding, return output as dictionary with
        # latent index as key.
        pass
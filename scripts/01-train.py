import argparse as ap

from keras.optimizers import Adam
from keras.callbacks import (
    TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, CSVLogger)

from tcvae.models import TCVAE
from tcvae.models.square_128 import (
    make_encoder_7_convs, make_decoder_7_deconvs)
from tcvae.losses import convert_loss_dict_keys
from tcvae.data import ImageDataGenerator
from tcvae.utils import import_project_root, read_yaml, write_json
from tcvae.callbacks import (
    ReconstructionCheck, LatentTraversalCheck, LatentDistributionLogging)
from tcvae.visualization import plot_loss_history, plot_dist_history

import_project_root()
from config import EXPERIMENTS_DIR, YAML_DIR, CELEB_A_DIR


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument(
        'experiment_name', required=True, type=str,
        help=(
            'The base name of the directory where all experiment resources '
            'will be saved.'))
    parser.add_argument(
        'yaml_config_base', required=True, type=str,
        help=(
            'The base name of the yaml config file containing hyperparameters '
            'for the training run.'))
    parser.add_argument(
        '-e', '--num_epochs', type=int, default=200,
        help='The maximum number of epochs allowed for the model training run.')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=256,
        help='The number of images in each batch of training data.')
    parser.add_argument(
        '-D', '--num_dist_log_images', type=int, default=1024,
        help='Number of images used to monitor latent distribution history.')
    parser.add_argument(
        '-R', '--num_recon_check_images', type=int, default=8,
        help='Number of images to reconstruct at end of each training epoch.')
    args = parser.parse_args()
    return vars(args)


def set_constants():
    constants = dict(
        experiments_dir=EXPERIMENTS_DIR,
        yaml_dir=YAML_DIR,
        json_log_base='hyperparameter-log.json',
        data_dir=CELEB_A_DIR,
        num_dist_log_imgs=1024,
        num_recon_check_imgs=8,
        model_stem='model',
        training_stem='training',
        recon_check_stem='reconstruction-check',
        traversal_check_stem='traversal-check',
        latent_dist_history_log_base='latent-distribution-history.csv',
        loss_history_log_base='loss-history.csv',
        latent_dist_history_viz_base='latent-distribution-history.html',
        loss_history_viz_base='loss-history.html'
    )
    return constants


def main():

    # Build argument dictionary. YAML overrides constants. User args
    # override YAML.
    args = parse_args()
    constants = set_constants()
    config_file = constants['yaml_dir'] / args['yaml_config_base']
    yaml_dict = read_yaml(config_file)
    args = {**constants, **yaml_dict, **args}

    # Build paths
    experiment_dir = args['experiments_dir'] / args['experiment_name']

    log_file = experiment_dir / args['json_log_base']
    model_dir = experiment_dir / args['model_stem']
    training_dir = experiment_dir / args['training_stem']

    recon_check_dir = training_dir / args['recon_check_stem']
    traversal_check_dir = training_dir / args['traversal_check_stem']
    latent_dist_history_log_file = \
        training_dir / args['latent_dist_history_log_base']
    loss_history_log_file = training_dir / args['loss_history_log_base']
    latent_dist_history_viz_file = \
        training_dir / args['latent_dist_history_viz_base']
    loss_history_viz_file = training_dir / args['loss_history_viz_base']

    # Log hyperparameters
    write_json(log_file, args)

    # Load data
    datagen = ImageDataGenerator(args['data_dir'], args['batch_size'])
    dist_log_imgs = datagen.load_n_images(args['num_dist_log_imgs'])
    recon_check_imgs = dist_log_imgs[:args['num_recon_check_imgs']]
    traversal_check_img = dist_log_imgs[0]

    # Load model if exists and compile
    if model_dir.is_dir():
        model = TCVAE.load(model_dir)
    else:
        encoder = make_encoder_7_convs(
            **args.get('encoder_kwargs', {}))
        decoder = make_decoder_7_deconvs(
            **args.get('decoder_kwargs', {}))
        loss_dict = convert_loss_dict_keys(args['loss_dict'])
        model = TCVAE(encoder, decoder, loss_dict)
    optimizer = Adam(lr=args['learning_rate'])
    model.compile(optimizer)

    # Make callbacks
    reconstruction_checker = ReconstructionCheck(
        recon_check_imgs, recon_check_dir,
        **args.get('recon_check_kwargs', {}))
    traversal_checker = LatentTraversalCheck(
        traversal_check_img, traversal_check_dir,
        **args.get('traversal_check_kwargs', {}))
    dist_logger = LatentDistributionLogging(
        latent_dist_history_log_file, dist_log_imgs,
        **args.get('latent_dist_logging_kwargs', {}))
    nan_terminator = TerminateOnNaN()
    csv_logger = CSVLogger(loss_history_log_file, append=True)
    lr_reducer = ReduceLROnPlateau(
        verbose=1, **args.get('reduce_lr_kwargs', {}))
    early_stopper = EarlyStopping(
        verbose=1, restore_best_weights=True,
        **args.get('early_stopping_kwargs', {}))
    callbacks = [
        reconstruction_checker, traversal_checker, dist_logger,
        nan_terminator, csv_logger, lr_reducer, early_stopper]

    try:
        model.fit_generator(
            datagen, callbacks=callbacks, epochs=args['num_epochs'])
    finally:
        model.save(model_dir)
        plot_loss_history(
            loss_history_log_file, loss_history_viz_file)
        plot_dist_history(
            latent_dist_history_log_file, latent_dist_history_viz_file)


if __name__ == '__main__':
    main()

import os
import argparse as ap

import imageio

from tcvae.models import TCVAE
from tcvae.data import ImageDataGenerator
from tcvae.utils import import_project_root, make_directory
from tcvae.visualization import process_for_animation, animate_traversals

import_project_root()
from config import EXPERIMENTS_DIR, CELEB_A_DIR


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument(
        'experiment_name', type=str, help=(
            'The base name of the directory containing resources for a specific '
            'experiment.'))
    parser.add_argument(
        '-i', '--num_imgs', type=int, default=25,
        help='The number of images on which to perform traversals.')
    parser.add_argument(
        '-r', '--num_rows', type=int, default=5,
        help='The number of rows to use when tiling images together.')
    parser.add_argument(
        '-t', '--std_threshold', type=float, default=0.8,
        help=(
            'Latent dimensions with average standard deviations above this '
            'threshold are not included in traversals.'))
    parser.add_argument(
        '-o', '--overwrite', action='store_true',
        help='If true, overwrites directory containing existing traversals.')
    parser.add_argument(
        '-g', '--gpu', type=int, default=15,
        help='The ID of the GPU to use for generating traversals.')
    args = parser.parse_args()
    return args
    

def main():
    
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    experiment_dir = EXPERIMENTS_DIR / args.experiment_name
    model_dir = experiment_dir / 'model'
    traversal_dir = experiment_dir / 'traversals'
    make_directory(traversal_dir, overwrite=args.overwrite)
    
    model = TCVAE.load(model_dir)
    datagen = ImageDataGenerator(CELEB_A_DIR, args.num_imgs)
    imgs = datagen.load_n_images(args.num_imgs, random=False)
    
    traversals = model.make_all_traversals(
        imgs, num_rows=args.num_rows, std_threshold=args.std_threshold,
        verbose=True)
    for key, value in traversals.items():
        traversals[key] = process_for_animation(value)
        
    animate_traversals(traversals, traversal_dir)


if __name__ == '__main__':
    main()

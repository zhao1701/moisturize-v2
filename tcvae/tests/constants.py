import os
from glob import glob

DATA_DIR = 'data'
SAVE_DIR = 'test-model'
RECONSTRUCTION_CHECK_DIR = 'reconstrucion-check'
TRAVERSAL_CHECK_DIR = 'traversal-check'
DISTRIBUTION_LOGGING_CSV_FILE = 'latent-distribution-log.csv'

NUM_IMAGES = len(glob(os.path.join(DATA_DIR, '*')))
NUM_LATENTS = 32
IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS = 128, 128, 3
BATCH_SIZE = 32

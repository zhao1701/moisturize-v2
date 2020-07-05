#!/usr/bin/env python


from pathlib import Path


PROJECT_DIR = Path(__file__).parent.resolve()


APP_DIR = PROJECT_DIR / 'app'
DATA_DIR = PROJECT_DIR / 'data'
EXPERIMENTS_DIR = PROJECT_DIR / 'experiments'
LIBRARY_DIR = PROJECT_DIR / 'tcvae'
NOTEBOOKS_DIR = PROJECT_DIR / 'notebooks'


TRAIN_DATA_DIR = DATA_DIR / 'train'
MODELS_DIR = LIBRARY_DIR / 'models'
YAML_DIR = EXPERIMENTS_DIR / 'yaml'

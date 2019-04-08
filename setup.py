#!/usr/bin/env python

from setuptools import setup

setup(
    name='tcvae',
    version='0.0.0',
    description=(
        'Keras implementation of Total Correlation Variational '
        'Autoencoder'),
    author='Derek Zhao',
    author_email='chong.zhao@columbia.edu',
    license='MIT',
    packages=setuptools.find_packages(),
    zip_safe=False)


#!/usr/bin/env python

"""
This module contains utilities for predicting with and inspecting
autoencoders models.
"""


import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

from tcvae.utils import check_path


def tile_multi_image_traversal(latent_traversals, num_rows):
    traversal_resolution, num_samples, img_height, img_width, num_channels = \
        latent_traversals.shape
    assert (num_samples % num_rows == 0), (
        'The number of rows of the stitched image must be an integer divisor '
        'of the number of samples in the batch.')
    num_cols = num_samples // num_rows
    latent_traversals = latent_traversals.reshape(
        traversal_resolution, num_rows, num_cols, img_height, img_width,
        num_channels)
    latent_traversals = latent_traversals.transpose(0, 1, 3, 2, 4, 5)
    latent_traversals = latent_traversals.reshape(
        traversal_resolution, num_rows * img_height, num_cols * img_width,
        num_channels)
    return latent_traversals


def plot_loss_history(csv_file, html_file=None):
    csv_file = check_path(csv_file, path_type=str)
    df = pd.read_csv(csv_file)
    df = df.drop('epoch', axis='columns')
    metrics = df.columns
    num_metrics = len(metrics)
    colors = [
        'blue', 'green', 'red', 'orange', 'purple', 'yellow']

    # Individual area plots for each metric
    area_plots = list()
    kdim = hv.Dimension('epoch', label='Epoch', range=(None, None))
    for index, metric in enumerate(metrics):
        label = metric.capitalize().replace('_', ' ')
        vdim = hv.Dimension(metric, label=label, range=(None, None))
        ylim = (df[metric].min(), df[metric].max())
        xlabel = 'Epoch' if index == (num_metrics - 1) else ''
        area_plot = hv.Area(
            (df.index, df[metric]), vdims=vdim, kdims=kdim)
        area_plot = area_plot.opts(
            ylim=ylim, color=colors[index], xlabel=xlabel)
        area_plots.append(area_plot)

    # Composition of multiple line plots for each metric
    line_plots = list()
    vdim = hv.Dimension('value', label='Value', range=(None, None))
    for index, metric in enumerate(metrics):
        label = metric.capitalize().replace('_', ' ')
        line_plot = hv.Curve(
            (df.index, df[metric]), vdims=vdim, kdims=kdim, label=label)
        line_plot = line_plot.opts(color=colors[index])
        line_plots.append(line_plot)
    overlay = hv.Overlay(line_plots).opts(xlabel='')

    # Create final layout
    all_plots = [overlay] + area_plots
    layout = hv.Layout(all_plots).cols(1).opts(
        opts.Area(width=800, height=200, alpha=0.2),
        opts.Curve(width=800, height=200)).opts(title='Training history')

    # Save HTML file
    if html_file is not None:
        html_file = check_path(html_file, path_type=str)
        hv.save(layout, html_file)
    return layout


def plot_dist_history(csv_file, html_file=None):
    csv_file = check_path(csv_file, path_type=str)
    df = pd.read_csv(csv_file)
    width = 1000

    sigma_cols = df.columns[df.columns.str.contains('sigma')]
    mu_cols = df.columns[df.columns.str.contains('mu')]

    # Create bar plot with most recent latent standard deviations
    kdim = hv.Dimension('latents', label='Latent dimension')
    vdim = hv.Dimension('sigma', label='Current standard deviation')
    sigmas_latest = df[sigma_cols].iloc[-1]
    sigmas_latest = [
        (index.split('_')[-1], value) for index, value
        in sigmas_latest.iteritems()]
    sigma_bar_plot = hv.Bars(sigmas_latest, kdims=kdim, vdims=vdim)
    sigma_bar_plot = sigma_bar_plot.opts(width=width)

    # Create line plots of latent standard deviation history
    kdim = hv.Dimension('epoch', label='Epoch')
    sigma_line_plots = [
        hv.Curve(
            (df.index, df[col]),
            vdims=hv.Dimension(col, label='Standard deviation'),
            kdims=kdim, label='Latent {}'.format(index)).opts(alpha=0.5)
        for index, col in enumerate(sigma_cols)]
    sigma_line_overlay = hv.Overlay(sigma_line_plots).opts(
        opts.Curve(width=width, height=400, show_grid=True))

    # Create line plots of latent mean history
    mu_line_plots = [
        hv.Curve(
            (df.index, df[col]),
            vdims=hv.Dimension(col, label='Mean'),
            kdims=kdim, label='Latent {}'.format(index)).opts(alpha=0.5)
        for index, col in enumerate(mu_cols)]
    mu_line_overlay = hv.Overlay(mu_line_plots).opts(
        opts.Curve(width=width, height=400, show_grid=True))

    # Create composite layout
    layout = sigma_bar_plot + sigma_line_overlay + mu_line_overlay
    layout = layout.cols(1).opts(
        title='Latent Distribution History')

    # Save HTML file
    if html_file is not None:
        html_file = check_path(html_file, path_type=str)
        hv.save(layout, html_file)
    return layout

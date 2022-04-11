import matplotlib.pyplot as plt
from typing import List, Tuple, Union
import numpy as np
import os

from Utils import get_der_sum


def update_range(values, g_range):
    left_obj = np.min(values) if g_range[0] is None else min(np.min(values), g_range[0])
    right_obj = np.max(values) if g_range[1] is None else max(np.max(values), g_range[1])
    return left_obj, right_obj


def get_measure_func(name: str):
    if name == 'var':
        return np.var
    elif name == 'mean':
        return np.mean
    else:
        raise Exception('No such function')


def add_info(patch: np.ndarray, points=None, mean: bool = True, std: bool = True,
             der_sum: bool = True):
    info = ''
    if mean:
        info += f'Mean: {float("{:.3f}".format(np.mean(patch)))}\n'
    if std:
        info += f'STD: {float("{:.3f}".format(np.std(patch)))}\n'
    if der_sum:
        info += f'Der Sum: {float("{:.3f}".format(get_der_sum(patch)))}\n'
    if points is not None:
        if len(points[0]) > 1:
            particle_center_var = np.sum((np.squeeze(points).T - np.squeeze(points).mean(axis=1)) ** 2) / len(points)
        elif len(points[0]) == 1:
            particle_center_var = 0.0
        info += f'Center Var: {float("{:.3f}".format(particle_center_var))}'
    return info


def save_patches_with_info(patches : List[np.ndarray], labels: List[str], path, snr, info=True, plot=False):
    fig = plt.figure()
    fig.suptitle(f'filter size - {0}, SNR - {float("{:.3f}".format(snr))}')
    for i, patch in enumerate(patches):
        ax = fig.add_subplot(1, len(patches), i+1)
        ax.imshow(patch)
        ax.axis('off')
        if info:
            ax.title.set_text(f'{labels[i]}\n' + add_info(patch))
    plt.savefig(path + f'original_SNR_{float("{:.3f}".format(snr))}.png', edgecolor='none')
    if plot:
        plt.show()


def save_filtered_patches(patches: List[np.ndarray], labels: List[str], filter_size: int,
                          true_centers: List[Tuple[List[int], List[int]]], points: List[Tuple[List[int], List[int]]],
                          path: str, mark_points: bool = True, info=True, plot=False):
    os.makedirs(path, exist_ok=True)
    fig = plt.figure()
    fig.suptitle(f'filter size - {filter_size} with top {len(points[0][0])} points')

    for i, patch in enumerate(patches):
        ax = fig.add_subplot(1, len(patches), i + 1)
        ax.imshow(patch)
        if mark_points:
            for j in range(len(points[i][0])):
                ax.scatter(points[i][1][j], points[i][0][j], color='r')
            ax.scatter(true_centers[i][1], true_centers[i][0], color='purple')
        ax.axis('off')
        if info:
            ax.title.set_text(f'{labels[i]}\n' + add_info(patch, points[i]))

    plt.savefig(path + f'filter_size_{filter_size}.png', edgecolor='none')
    if plot:
        plt.show()


def save_radial_mean(patches: List[np.ndarray], labels: List[str], filter_size: int,
                          points: List[Tuple[List[int], List[int]]], path: str):

    radial_info = {}
    shifted_patches = {}

    mean_range = (None, None)
    var_range = (None, None)

    for i, patch in enumerate(patches):
        rd_mean, shifted_by_mean = radial_mean_with_center_top_n(image=patch, points=points[i], func_type='mean')
        rd_var, shifted_by_var = radial_mean_with_center_top_n(image=patch, points=points[i], func_type='var')

        mean_range = update_range(rd_mean, mean_range)
        var_range = update_range(rd_var, var_range)

        radial_info[labels[i]] = {'radial_mean': rd_mean, 'radial_var': rd_var}
        shifted_patches[labels[i]] = {'radial_mean': shifted_by_mean, 'radial_var': shifted_by_var}

    graph_shape = (2, len(radial_info))

    os.makedirs(path, exist_ok=True)
    fig = plt.figure(figsize=(8, 8), dpi=80)
    fig.suptitle(f'filter size - {filter_size}, with mean of top {len(points[0][0])} points')

    # for i in range(graph_shape[1]):
    #     ax1 = fig.add_subplot(graph_shape[0], graph_shape[1], i + 1)
    #     ax1.plot(radial_info[labels[i]]['radial_mean'])
    #     ax1.set_ylim(mean_range[0], mean_range[1])
    #     ax1.set(xticklabels=[])
    #     ax1.tick_params(bottom=False)
    #     ax1.title.set_text(f'Mean ' + labels[i])
    #
    #     ax2 = fig.add_subplot(graph_shape[0], graph_shape[1], i + 1 + graph_shape[1])
    #     ax2.plot(radial_info[labels[i]]['radial_var'])
    #     ax2.set_ylim(var_range[0], var_range[1])
    #     # ax2.set(xticklabels=[])
    #     # ax2.tick_params(bottom=False)
    #     ax2.title.set_text(f'Var ' + labels[i])

    ax1 = fig.add_subplot(graph_shape[0], 1, 1)
    for i in range(2):
        ax1.plot(radial_info[labels[i]]['radial_mean'], label=labels[i])
    ax1.legend()
    ax1.set_ylim(mean_range[0], mean_range[1])
    ax1.set(xticklabels=[])
    ax1.tick_params(bottom=False)
    ax1.title.set_text(f'Mean')

    ax2 = fig.add_subplot(graph_shape[0], 1, 2)
    for i in range(2):
        ax2.plot(radial_info[labels[i]]['radial_var'], label=labels[i])
    ax2.legend()
    ax2.set_ylim(var_range[0], var_range[1])
    # ax2.set(xticklabels=[])
    # ax2.tick_params(bottom=False)
    ax2.title.set_text(f'Var')

    plt.savefig(path + f'radial_mean_{filter_size}.png', edgecolor='none')
    plt.close()


def radial_mean_with_center_top_n(image: np.ndarray, points: Tuple[List[int], List[int]], func_type: str,
                                  allow_edges: bool = True):
    x = points[0]
    y = points[1]
    shape = image.shape
    quad_size = int(shape[0] / 4)
    func = get_measure_func(name=func_type)
    mean_radial_mean = []
    true_shifted = None

    for i in range(len(x)):
        if allow_edges:
            x_i = x[i]
            y_i = y[i]
        else:
            x_i = np.clip(x[i], quad_size, shape[0] - quad_size)
            y_i = np.clip(y[i], quad_size, shape[1] - quad_size)
        num_of_radiuses = int(min(shape) / 2)
        shift_x = np.ceil(shape[0] / 2).astype(np.int) - x_i
        shift_y = np.ceil(shape[1] / 2).astype(np.int) - y_i
        shifted = np.roll(image, shift=(shift_x, shift_y), axis=(0, 1))
        xs, ys = np.ogrid[0:shape[0], 0:shape[1]]

        shifted = shifted - shifted.mean()
        img_nrm = image - image.mean()
        radiuses = np.hypot(xs - x_i, ys - y_i)
        rbin = (num_of_radiuses * radiuses / radiuses.max()).astype(np.int)
        radial_mean = np.asarray([func(img_nrm[rbin <= i]) for i in np.arange(1, rbin.max() + 1)])
        mean_radial_mean.append(radial_mean[:2 * quad_size - 1])

        # Save the shift by the first point
        if i == 0:
            true_shifted = shifted
    return np.mean(mean_radial_mean, axis=0), true_shifted

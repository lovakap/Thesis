import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, SpectralBiclustering
from Implementations.centering_conv_ver_1 import apply_filter, fft_convolve_2d
from patch_simulator import PatchSimulator
from plot_utils import save_patches_with_info, save_filtered_patches, save_radial_mean, plot_distance_from_mean_noise, \
    plot_radial_info_with_ranges, save_graphs
from general_utils import get_filter_sizes, get_n_points, create_result_dir
import os
import matplotlib.pyplot as plt


def possible_in(x, y, im_size, edge):
    x_in = (x > edge) and x < (im_size - edge)
    y_in = (y > edge) and y < (im_size - edge)
    return x_in and y_in


noise_mean = .0
noise_std = [.5, 1, 2, 4, 8, 16, 32, 48, 64, 96]
# noise_std = [48, 64, 96]
# noise_std = [96]
noise_std = [ns * 1e-4 for ns in noise_std]
blob_mean = .5
blob_std = .5
"""Sizing"""
# blob_size = 65
blob_size = 100
particle_ratio = 0.7
image_size = int(blob_size / particle_ratio)
particle_vals_scale = 1# 1e5
patch_types = ['blob', 'projection', 'empty']

circle_cut = True
add_noise = True
max_val = False
mark_center = False
filter_loc = True

top_n = 10
train_number = 200
test_number = 50
f_size = 101
start = 10
end = 60
edge = 30
moving_avg = 3
filter_sizes_ratio = [0.5, 0.7, 0.9]
filter_sizes = get_filter_sizes(filter_sizes_ratio, image_size)
# filter_sizes = get_filter_sizes(filter_sizes_ratio, PATCH_SIZE)

projection_path = 'Data/projections/projection.npy'
res_dir = 'Results/simulations/'
# exp_num = f'Image_Size_{image_size}_Ratio_{particle_ratio}_Mean_{blob_mean}_std_{blob_std}_{top_n}/'
exp_num = f'distance_from_mean_noise/Clustering/Train Patches_{train_number} Top_{top_n}'
# exp_num = f'Image_Size_{image_size}_Ratio_{float("{:.3f}".format(blob_size / image_size))}_Mean_{blob_mean}_std_{blob_std}_{top_n}_scaled/'
# os.makedirs('results/' + exp_num, exist_ok=True)
os.makedirs(res_dir + exp_num, exist_ok=True)
res_table = []
for std in noise_std:
    patch_sampler = PatchSimulator(particle_size=blob_size, particle_ratio=particle_ratio,
                                   projection_path=projection_path, blob_mean=blob_mean, blob_std=blob_std,
                                   noise_mean=noise_mean, noise_std=std, particle_vals_scale=particle_vals_scale)
    positive_examples = []
    negative_examples = []

    for sample in range(train_number):
        single_object, single_object_center, snr = patch_sampler.get_patch(patch_type=patch_types[1])
        empty_image, empty_image_center, _ = patch_sampler.get_patch(patch_type=patch_types[2])

        filtered_particle = apply_filter(image=single_object, filter_size=f_size, circle_cut=circle_cut)
        filtered_empty = apply_filter(image=empty_image, filter_size=f_size, circle_cut=circle_cut)

        particle_points = get_n_points(filtered_particle, n=top_n, max_values=max_val)
        empty_points = get_n_points(filtered_empty, n=top_n, max_values=max_val)

        radial_info = save_radial_mean(patches=[single_object, empty_image], labels=['Object', 'Empty'],
                                       filter_size=f_size,
                                       points=[single_object_center, empty_points], path='', plot=False, return_vals=True)
                                       # points=[particle_points, empty_points], path='', plot=False, return_vals=True)
        # positive_examples.append(radial_info['Object']['radial_mean'][start:end])# + radial_info['Object']['radial_var'])
        # negative_examples.append(radial_info['Empty']['radial_mean'][start:end])# + radial_info['Empty']['radial_var'])
        positive_examples.append(
            np.convolve(radial_info['Object']['radial_mean'][start:end], np.ones(moving_avg) / moving_avg, 'same'))
        negative_examples.append(
            np.convolve(radial_info['Empty']['radial_mean'][start:end], np.ones(moving_avg) / moving_avg, 'same'))

    #save random sample
    temp_path = res_dir + exp_num + f'/SNR_{float("{:.6f}".format(snr))}_sample/'
    os.makedirs(temp_path, exist_ok=True)
    save_patches_with_info([single_object, empty_image], ['With Object', 'Empty'], path=temp_path, snr=snr)
    save_filtered_patches(patches=[filtered_particle, filtered_empty], labels=['With Object', 'Empty'],
                          true_centers=[single_object_center, empty_image_center], filter_size=f_size,
                          points=[particle_points, empty_points], path=temp_path)

    save_radial_mean(patches=[single_object, empty_image], labels=['With Object', 'Empty'],
                     filter_size=f_size, points=[particle_points, empty_points], path=temp_path)

    # save_graphs([positive_examples[-1], negative_examples[-1]], labels=['With Object', 'Empty'],
    #             x_range=np.arange(start, end), label=f'Moving_avg_{moving_avg}', path=temp_path)

    positive_examples = positive_examples[:50]
    train_examples = positive_examples + negative_examples

    test_examples = positive_examples + negative_examples
    test_labels = np.array([1] * len(positive_examples) + [0] * len(negative_examples))
    label_weights = np.array([.1] * len(positive_examples) + [.9] * len(negative_examples))

    model = KMeans(n_clusters=2, random_state=0).fit(train_examples)
    # model = SpectralClustering(n_clusters=2, random_state=0).fit(train_examples)
    # model = SpectralBiclustering(n_clusters=2, random_state=0).fit(train_examples)
    # test_pred_labels = model.predict(test_examples)
    test_pred_labels = model.labels_

    temp = test_pred_labels == test_labels
    acc = np.sum(temp) / len(test_labels)
    trp = temp[test_labels == 1].sum() / np.sum(test_labels == 1)
    trn = temp[test_labels == 0].sum() / np.sum(test_labels == 0)
    if acc < 0.4:
        acc = 1 - acc
        trn = 1 - trn
        trp = 1 - trp
        save_graphs([model.cluster_centers_[1], model.cluster_centers_[0]], labels=['With Object', 'Empty'],
                    x_range=np.arange(start, end), label=f'Centers_of_moving_{moving_avg}', path=temp_path)

    else:
        save_graphs([model.cluster_centers_[0], model.cluster_centers_[1]], labels=['With Object', 'Empty'],
                    x_range=np.arange(start, end), label=f'Centers_of_moving_{moving_avg}', path=temp_path)

    print(f'\nWith {train_number} train examples and SNR ~ {float("{:.5f}".format(snr))}')
    print(f'\nAcc : {float("{:.3f}".format(acc))},  True Positive : {trp},  True Negative: {trn}\n')
    res_table.append([np.round(snr, 6), acc, trp, trn])

import pandas as pd
tbl = pd.DataFrame(res_table, columns=['SNR', 'Accuracy', 'True Positive', 'True Negative'])
tbl.to_csv(res_dir + exp_num + '/table_res.csv', index=False)
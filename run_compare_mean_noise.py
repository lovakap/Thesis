import numpy as np

from Implementations.centering_conv_ver_1 import apply_filter, fft_convolve_2d
from patch_simulator import PatchSimulator
from plot_utils import save_patches_with_info, save_filtered_patches, save_radial_mean, plot_distance_from_mean_noise, \
    plot_radial_info_with_ranges
from general_utils import get_filter_sizes, get_n_points, create_result_dir
import os
import matplotlib.pyplot as plt

def possible_in(x,y, im_size, edge):
    x_in = (x > edge) and x < (im_size - edge)
    y_in = (y > edge) and y < (im_size - edge)
    return x_in and y_in

noise_mean = .0
noise_std = [.5, 1, 2, 4, 8, 16, 32]
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

top_n = 10
noise_patches = 50
test_number = 500
f_size = 101
start = 10
end = 50
edge = 30
filter_sizes_ratio = [0.5, 0.7, 0.9]
filter_sizes = get_filter_sizes(filter_sizes_ratio, image_size)
# filter_sizes = get_filter_sizes(filter_sizes_ratio, PATCH_SIZE)

projection_path = 'Data/projections/projection.npy'
res_dir = 'Results/simulations/'
# exp_num = f'Image_Size_{image_size}_Ratio_{particle_ratio}_Mean_{blob_mean}_std_{blob_std}_{top_n}/'
exp_num = f'distance_from_mean_noise/Classification/Noise Patches_{noise_patches} Top_{top_n}'
# exp_num = f'Image_Size_{image_size}_Ratio_{float("{:.3f}".format(blob_size / image_size))}_Mean_{blob_mean}_std_{blob_std}_{top_n}_scaled/'
# os.makedirs('results/' + exp_num, exist_ok=True)
os.makedirs(res_dir + exp_num, exist_ok=True)

for std in noise_std:
    patch_sampler = PatchSimulator(particle_size=blob_size, particle_ratio=particle_ratio,
                                   projection_path=projection_path, blob_mean=blob_mean, blob_std=blob_std,
                                   noise_mean=noise_mean, noise_std=std, particle_vals_scale=particle_vals_scale)

    noise_means = []
    noise_vars = []
    for noise_patch in range(noise_patches):
        temp_empty_image, temp_empty_image_center, _ = patch_sampler.get_patch(patch_type=patch_types[2])
        filtered_empty = apply_filter(image=temp_empty_image, filter_size=f_size, circle_cut=circle_cut)
        temp_empty_points = get_n_points(filtered_empty, n=top_n, max_values=max_val)
        if possible_in(temp_empty_points[0][0], temp_empty_points[1][0], image_size, edge):
            radial_info = save_radial_mean(patches=[temp_empty_image], labels=['Empty'], filter_size=f_size,
                                           points=[temp_empty_points], path='', plot=False, return_vals=True)
            noise_means.append(radial_info['Empty']['radial_mean'])
            noise_vars.append(radial_info['Empty']['radial_var'])

    noise_radial_mean = np.mean(noise_means, axis=0)
    noise_radial_var = np.mean(noise_vars, axis=0)

    # noise_distance_ = np.square(np.array(noise_means) - noise_radial_mean)[:, start:].sum(axis=-1)
    noise_distance_ = np.square(np.clip(np.array(noise_means), 0, None))[:, start:end].sum(axis=-1)
    threshold_dist = noise_distance_.std() #* start
    mean_distance = noise_distance_.mean()

    real_mean_dist = []
    real_var_dist = []
    noise_mean_dist = []
    noise_var_dist = []

    for test_n in range(test_number):
        single_object, single_object_center, snr = patch_sampler.get_patch(patch_type=patch_types[1])
        empty_image, empty_image_center, _ = patch_sampler.get_patch(patch_type=patch_types[2])

        filtered_particle = apply_filter(image=single_object, filter_size=f_size, circle_cut=circle_cut)
        filtered_empty = apply_filter(image=empty_image, filter_size=f_size, circle_cut=circle_cut)

        particle_points = get_n_points(filtered_particle, n=top_n, max_values=max_val)
        empty_points = get_n_points(filtered_empty, n=top_n, max_values=max_val)

        radial_info = save_radial_mean(patches=[single_object, empty_image], labels=['Object', 'Empty'], filter_size=f_size,
                                       points=[particle_points, empty_points], path='', plot=False, return_vals=True)
        # radial_info = save_radial_mean(patches=[single_object, empty_image], labels=['Object', 'Empty'], filter_size=f_size,
        #                                        points=[particle_points, empty_points], path='', plot=False, return_vals=True)


        if possible_in(particle_points[0][0], particle_points[1][0], image_size, edge):
            real_mean_dist.append(np.square(np.clip(radial_info['Object']['radial_mean'], 0, None))[start:end].sum())
        if possible_in(empty_points[0][0], empty_points[1][0], image_size, edge):
            noise_mean_dist.append(np.square(np.clip(radial_info['Empty']['radial_mean'], 0, None))[start:end].sum())
        # real_mean_dist.append(np.square(radial_info['Object']['radial_mean'] - noise_radial_mean)[start:].sum())
        # noise_mean_dist.append(np.square(radial_info['Empty']['radial_mean'] - noise_radial_mean)[start:].sum())

        # real_var_dist.append(np.square(radial_info['Object']['radial_var'] - noise_radial_var)[start:].sum())
        # noise_var_dist.append(np.square(radial_info['Empty']['radial_var'] - noise_radial_var)[start:].sum())

        # real_mean_dist.append((radial_info['Object']['radial_mean'] - noise_radial_mean)[start:].sum())
        # noise_mean_dist.append((radial_info['Empty']['radial_mean'] - noise_radial_mean)[start:].sum())
        #
        # real_var_dist.append((radial_info['Object']['radial_var'] - noise_radial_var)[start:].sum())
        # noise_var_dist.append((radial_info['Empty']['radial_var'] - noise_radial_var)[start:].sum())

    trp = (np.array(real_mean_dist) > mean_distance + threshold_dist).sum()
    flp = test_number - trp
    fln = (np.array(noise_mean_dist) > mean_distance + threshold_dist).sum()
    trn = test_number - fln

    print(f'\nSnr {float("{:.5f}".format(snr))}')
    print(f'True Positive   {float("{:.3f}".format(trp / test_number))}, False Positive  {float("{:.3f}".format(flp / test_number))}')
    print(f'True Negative  {float("{:.3f}".format(trn / test_number))}, False Negative  {float("{:.3f}".format(fln/test_number))}')
    plt.plot(real_mean_dist, label='Real')
    plt.plot(noise_mean_dist, label='Noise')
    plt.plot([mean_distance + threshold_dist] * len(real_mean_dist), label='Mean + Std')
    plt.legend()
    plt.title(
        f'Snr {float("{:.5f}".format(snr))}, TRP: {float("{:.3f}".format(trp / test_number))}, TRN: {float("{:.3f}".format(trn / test_number))}')
    plt.savefig(res_dir + exp_num + f'/SNR_{float("{:.5f}".format(snr))}.png')
    # plt.show()
    plt.close()
    # plot_radial_info_with_ranges(radial_info, noise_means, noise_vars, snr, noise_patches, top_n, res_dir + exp_num)
    # plot_distance_from_mean_noise(real_mean_dist, noise_mean_dist, real_var_dist, noise_var_dist, snr, noise_patches,
    #                               top_n, res_dir + exp_num)
    # print('end')
    # path = create_result_dir(f'{exp_num}SNR_{float("{:.3f}".format(snr))}', res_dir)
    # print(f'SNR : {float("{:.3f}".format(snr))}')
    #
    #
    #
    # # save_patches_with_info([single_object, empty_image], ['With Object', 'Empty'], path=path, snr=snr)
    #
    # # for f_size in filter_sizes:
    # for f_size in [101]:
    #     filtered_particle = apply_filter(image=single_object, filter_size=f_size, circle_cut=circle_cut)
    #     filtered_empty = apply_filter(image=empty_image, filter_size=f_size, circle_cut=circle_cut)
    #
    #     particle_points = get_n_points(filtered_particle, n=top_n, max_values=max_val)
    #     empty_points = get_n_points(filtered_empty, n=top_n, max_values=max_val)
    #     # particle_points = single_object_center
    #
    #     save_filtered_patches(patches=[filtered_particle, filtered_empty], labels=['With Object', 'Empty'],
    #                           true_centers=[single_object_center, empty_image_center], filter_size=f_size,
    #                           points=[particle_points, empty_points], path=path)
    #
    #     save_radial_mean(patches=[single_object, empty_image], labels=['With Object', 'Empty'],
    #                      filter_size=f_size, points=[particle_points, empty_points], path=path,
    #                      noise_mean=noise_radial_mean, noise_var=noise_radial_var)
    #     # save_radial_mean(patches=[filtered_particle, filtered_empty], labels=['With Object', 'Empty'],
    #     #                          filter_size=f_size, points=[particle_points, empty_points], path=path)

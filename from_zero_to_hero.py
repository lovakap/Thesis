import pandas as pd
from Utils import get_image_for_testing, create_result_dir, save_plot, get_der_sum, save_double_plot_top_n,\
    get_patched_var, save_double_plot_var, save_double_radial_mean_top_n, plot_image_with_marks, get_empty_image
from mrc_utils import read_mrc, get_coordinates, crop_random_particle, crop_random_patch, apply_ctf_on, add_patches, \
    PATCH_SIZE, HALF_PATH_SIZE, apply_patches
from Implementations.centering_conv_ver_1 import apply_filter, fft_convolve_2d
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import time

noise_mean = .0
noise_std = [.5, 1, 2, 4, 8, 16, 32]
# noise_std = [16]
blob_mean = .5
blob_std = .5
image_size = 100
blob_size = 50


circle_cut = True
add_noise = True
max_val = False
mark_center = False
top_n = 5

res_dir = 'Results/zero_to_hero/'
exp_num = f'Image_Size_{image_size}_Ratio_{float("{:.3f}".format(blob_size / image_size))}_Mean_{blob_mean}_std_{blob_std}/'
# os.makedirs('results/' + exp_num, exist_ok=True)
os.makedirs(res_dir + exp_num, exist_ok=True)
for std in noise_std:
    single_object, center, snr = get_image_for_testing(image_size=image_size, sub_image_size=blob_size,
                                                       add_noise=add_noise,
                                                       noise_mean=noise_mean, noise_std=std, blob_mean=blob_mean,
                                                       blob_std=blob_std, random_choise=False)

    empty_image = get_empty_image(image_size=image_size, noise_mean=noise_mean, noise_std=std, add_noise=add_noise)

    path = create_result_dir(f'{exp_num}SNR_{float("{:.3f}".format(snr))}', res_dir)
    print(f'SNR : {float("{:.3f}".format(snr))}')

    fig = plt.figure()
    fig.suptitle(f'filter size - {0}, SNR - {float("{:.3f}".format(snr))}')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(single_object)
    ax1.axis('off')
    ax1.title.set_text(f'With Object'
                       f'\nMean: {float("{:.3f}".format(np.mean(single_object)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(single_object)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(single_object)))}')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(empty_image)
    ax2.axis('off')
    ax2.title.set_text(f'Without Object'
                       f'\nMean: {float("{:.3f}".format(np.mean(empty_image)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(empty_image)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(empty_image)))}')
    plt.savefig(path + f'original_SNR_{float("{:.3f}".format(snr))}.png', edgecolor='none')
    plt.close()


    filter_size = [0.5, 0.7, 0.9]
    for f_size in [int(PATCH_SIZE * i) for i in filter_size]:
        if f_size % 2 == 0:
            f_size -= 1
        filtered_particle = apply_filter(image=single_object, filter_size=f_size, circle_cut=circle_cut)
        filtered_patch = apply_filter(image=empty_image, filter_size=f_size, circle_cut=circle_cut)
        # filtered_patch = apply_filter(image=var_patch, filter_size=int(f_size/2), circle_cut=circle_cut)
        # save_double_plot_var(filtered_particle, filtered_patch, filter_size=f_size, path=path, snr=0,
        #                      max_val=max_val, mark_center=True, x=0, y=0)

        if max_val:
            p1 = np.unravel_index(np.argsort(filtered_particle.flatten())[-top_n:], filtered_particle.shape)
            p2 = np.unravel_index(np.argsort(filtered_patch.flatten())[-top_n:], filtered_patch.shape)
        else:
            p1 = np.unravel_index(np.argsort(filtered_particle.flatten())[:top_n], filtered_particle.shape)
            p2 = np.unravel_index(np.argsort(filtered_patch.flatten())[:top_n], filtered_patch.shape)

        save_double_plot_top_n(particle=filtered_particle, random=filtered_patch, filter_size=f_size, path=path, p1=p1,
                               p2=p2, particle_count=0)
        # save_double_radial_mean_top_n(particle=single_object, with_var=empty_image, filter_size=f_size, path=path,
        #                               p1=p1, p2=p2)
        save_double_radial_mean_top_n(particle=single_object, with_var=empty_image, filtered_particle=filtered_particle,
                                      filtered_with_var=filtered_patch, filter_size=f_size, path=path, p1=p1, p2=p2)

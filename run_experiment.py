from Utils import get_image_for_testing, create_result_dir, save_plot
from Implementations.centering_conv_ver_1 import apply_filter
import matplotlib.pyplot as plt
import numpy as np
import os
# run 1
noise_mean = 0.0
noise_std = [0.1, 0.3, 0.5, 0.7, 0.9]
blob_mean = 0.5
blob_std = 0.3
image_size = 500
blob_size = 75
circle_cut = True
exp_num = f'Experiment_{1}/'
os.makedirs('results/' + exp_num, exist_ok=True)

for std in noise_std:
    snr = blob_std**2 / std**2
    # path = create_result_dir(f'blob_std_{std}_mean_{}_image_size_{image_size}')
    path = create_result_dir(f'{exp_num}SNR_{float("{:.3f}".format(snr))}')

    print(f"std : {std}")
    image, center = get_image_for_testing(image_size=image_size, sub_image_size=blob_size, add_noise=True,
                                          noise_mean=noise_mean, noise_std=std, blob_mean=blob_mean, blob_std=blob_std)
    print(f"center - {center}")
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'original_std_{float("{:.3f}".format(snr))}')
    plt.scatter(center[1], center[0], color='red')
    plt.colorbar()
    plt.savefig(path + f'original_SNR_{float("{:.3f}".format(snr))}.png', edgecolor='none')
    plt.close()

    # filter_size = [5, 11, 15, 25, 31, 41]
    # filter_size = [25, 31, 41, 51, 61]
    # filter_size = [41, 51, 61, 75, 91, 111, 151, 175]
    filter_size = [75, 91, 111]
    for f_size in filter_size:
        filtered_image = apply_filter(image=image, filter_size=f_size, circle_cut=circle_cut)
        print(f'filter size {f_size}, number of min points: {np.sum(filtered_image == filtered_image.min())}')
        save_plot(image=filtered_image, filter_size=f_size, path=path, snr=snr, max_val=False, mark_center=False)
        # save_dual_plot(image=filtered_image, filter_size=f_size, path=path, snr=snr, max_val=False, mark_center=False)
        print(f'found center {np.unravel_index(np.argmin(filtered_image,axis=None), filtered_image.shape)}')
        # plt.imshow(filtered_image)
        # plt.show()

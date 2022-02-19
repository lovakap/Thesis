from Utils import get_image_for_testing, create_result_dir, save_plot, get_empty_image, get_glued_image, \
    save_triple_plot, get_image_der2, get_der_sum
from Implementations.centering_conv_ver_1 import apply_filter
import matplotlib.pyplot as plt
import numpy as np
import os

# run 3 - comparison of landscapes with correct SNR calculation

noise_mean = .0
# noise_std = [0.1, 0.2, 0.3]
noise_std = [1., 2., 4., 6., 8., 16.]
# noise_std = [4.]
blob_mean = .0
blob_std = 1.
image_size = 100
blob_size = 70

circle_cut = True
add_noise = True
max_val = False
mark_center = True

exp_num = f'Image_Size_{image_size}_Ratio_{float("{:.3f}".format(blob_size / image_size))}_Mean_{blob_mean}_std_{blob_std}/'
os.makedirs('results/' + exp_num, exist_ok=True)

for std in noise_std:
    single_object, center, snr = get_image_for_testing(image_size=image_size, sub_image_size=blob_size,
                                                       add_noise=add_noise,
                                                       noise_mean=noise_mean, noise_std=std, blob_mean=blob_mean,
                                                       blob_std=blob_std)
    empty_image = get_empty_image(image_size=image_size, noise_mean=noise_mean, noise_std=std, add_noise=add_noise)

    glued_object = get_glued_image(image_size=image_size, sub_image_size=blob_size, add_noise=add_noise,
                                   noise_mean=noise_mean, noise_std=std, blob_mean=blob_mean, blob_std=blob_std)

    path = create_result_dir(f'{exp_num}SNR_{float("{:.3f}".format(snr))}')
    print(f'SNR : {float("{:.3f}".format(snr))}')

    fig = plt.figure()
    fig.suptitle(f'filter size - {0}, SNR - {float("{:.3f}".format(snr))}')
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(single_object)
    ax1.axis('off')
    ax1.title.set_text(f'With Object'
                       f'\nMean: {float("{:.3f}".format(np.mean(single_object)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(single_object)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(single_object)))}')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(empty_image)
    ax2.axis('off')
    ax2.title.set_text(f'Without Object'
                       f'\nMean: {float("{:.3f}".format(np.mean(empty_image)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(empty_image)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(empty_image)))}')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(glued_object)
    ax3.axis('off')
    ax3.title.set_text(f'Glued Object'
                       f'\nMean: {float("{:.3f}".format(np.mean(glued_object)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(glued_object)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(glued_object)))}')
    plt.savefig(path + f'original_SNR_{float("{:.3f}".format(snr))}.png', edgecolor='none')
    plt.close()

    filter_size = [0.1, 0.25, 0.5, 0.75, 1.]
    for f_size in [int(image_size * i) for i in filter_size]:
        if f_size % 2 == 0:
            f_size -= 1
        filtered_single_object = apply_filter(image=single_object, filter_size=f_size, circle_cut=circle_cut)
        filtered_empty = apply_filter(image=empty_image, filter_size=f_size, circle_cut=circle_cut)
        filtered_glued_object = apply_filter(image=glued_object, filter_size=f_size, circle_cut=circle_cut)

        save_triple_plot(single_object=filtered_single_object, empty=filtered_empty, glued=filtered_glued_object,
                         filter_size=f_size, path=path, snr=snr, max_val=max_val, mark_center=mark_center, x=center[1],
                         y=center[0])
        # plt.imshow(filtered_image)
        # plt.show()

import pandas as pd
from Utils import get_image_for_testing, create_result_dir, save_plot, get_der_sum, save_double_plot,\
    get_patched_var, save_double_plot_var, save_double_radial_mean, plot_image_with_marks
from mrc_utils import read_mrc, get_coordinates, crop_random_particle, crop_random_patch, apply_ctf_on, add_patches, \
    PATCH_SIZE, HALF_PATH_SIZE, apply_patches
from Implementations.centering_conv_ver_1 import apply_filter, fft_convolve_2d
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time


# file_path = 'Data/Micrographs/002'
# file_name = '002'
circle_cut = True
max_val = False
file_name = 'Falcon_2012_06_12-14_57_34_0'
# file_name = 'Falcon_2012_06_12-14_33_35_0'
# file_name = 'Falcon_2012_06_12-15_07_41_0'
file_path = 'Data/mrc_files/' + file_name
full_micrograph = read_mrc(file_path + '.mrc').T
full_micrograph = apply_ctf_on(full_micrograph, file_name)

kernel_size = PATCH_SIZE
g = gaussian(kernel_size, 1).reshape(kernel_size, 1)
g = np.dot(g, g.T)
full_micrograph = fft_convolve_2d(full_micrograph, g)

coordinates = get_coordinates(file_path + '.coord')
cropped_particle, cropped_particle_center, cropped_particle_intersection = crop_random_particle(
    micrograph=full_micrograph, coordinates=coordinates)
cropped_patch, cropped_patch_center, cropped_patch_intersection = crop_random_patch(micrograph=full_micrograph,
                                                                                    coordinates=coordinates)
list_patches = apply_patches(coordinates)
fig, ax = plt.subplots()
ax.imshow(full_micrograph, cmap='gray')
for p in list_patches:
    ax.add_patch(p)
plt.show()

sub_fold = 'Results/Real_Data/'
path = create_result_dir(f'{file_name}_{PATCH_SIZE}', sub_fold, create_sub_folder=True)

fig, ax = plt.subplots()
ax.imshow(full_micrograph[max(cropped_particle_center[0] - 4 * PATCH_SIZE, 0): min(cropped_particle_center[0] + 4 * PATCH_SIZE, full_micrograph.shape[0]),
           max(cropped_particle_center[1] - 4 * PATCH_SIZE, 0): min(cropped_particle_center[1] + 4 * PATCH_SIZE, full_micrograph.shape[1])], cmap='gray')
ax.title.set_text(
    f'Patch at [{max(cropped_particle_center[0] - 4 * PATCH_SIZE, 0)} : {min(cropped_particle_center[0] + 4 * PATCH_SIZE, full_micrograph.shape[0])}, {max(cropped_particle_center[1] - 4 * PATCH_SIZE, 0)}: {min(cropped_particle_center[1] + 4 * PATCH_SIZE, full_micrograph.shape[1])}]')
cord_x = 3 * PATCH_SIZE if cropped_particle_center[0] >= 4 * PATCH_SIZE else (cropped_particle_center[0] - PATCH_SIZE)
cord_y = 3 * PATCH_SIZE if cropped_particle_center[1] >= 4 * PATCH_SIZE else (cropped_particle_center[1] - PATCH_SIZE)

ptch = patches.Rectangle((cord_x, cord_y), PATCH_SIZE, PATCH_SIZE, linewidth='.5', edgecolor='r', facecolor='None')
ax.add_patch(ptch)
plt.savefig(path + 'patch.png')


fig = plt.figure()
fig.suptitle(f'patch size - {PATCH_SIZE}')
ax1 = fig.add_subplot(1, 2, 1)
# ax1.imshow(cropped_particle, cmap='gray')
ax1.imshow(cropped_particle, cmap='gray')
list_patches = add_patches(cropped_particle_center, [coordinates[i] for i in cropped_particle_intersection])
for p in list_patches:
    ax1.add_patch(p)
ax1.axis('off')
ax1.title.set_text(f'Particle with {len(cropped_particle_intersection)} intersection'
                   f'\nMean: {float("{:.3f}".format(np.mean(cropped_particle)))}'
                   f'\nSTD: {float("{:.3f}".format(np.std(cropped_particle)))}'
                   f'\nDer Sum: {float("{:.3f}".format(get_der_sum(cropped_particle)))}')
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(cropped_patch, cmap='gray')
list_patches = add_patches(cropped_patch_center, [coordinates[i] for i in cropped_patch_intersection])
for p in list_patches:
    ax2.add_patch(p)
ax2.axis('off')
ax2.title.set_text(f'Patch with {len(cropped_patch_intersection)} intersection'
                   f'\nMean: {float("{:.3f}".format(np.mean(cropped_patch)))}'
                   f'\nSTD: {float("{:.3f}".format(np.std(cropped_patch)))}'
                   f'\nDer Sum: {float("{:.3f}".format(get_der_sum(cropped_patch)))}')
plt.savefig(path + f'original_patches.png', edgecolor='none')
plt.close()

# filter_size = [0.1, 0.25, 0.5, 0.75, 1.]
filter_size = [0.5, 0.7, 0.9]
for f_size in [int(PATCH_SIZE * i) for i in filter_size]:
    if f_size % 2 == 0:
        f_size -= 1
    filtered_particle = apply_filter(image=cropped_particle, filter_size=f_size, circle_cut=circle_cut)
    filtered_patch = apply_filter(image=cropped_patch, filter_size=f_size, circle_cut=circle_cut)
    # filtered_patch = apply_filter(image=var_patch, filter_size=int(f_size/2), circle_cut=circle_cut)
    # save_double_plot_var(filtered_particle, filtered_patch, filter_size=f_size, path=path, snr=0,
    #                      max_val=max_val, mark_center=True, x=0, y=0)
    save_double_plot(particle=filtered_particle, random=filtered_patch, filter_size=f_size, path=path, snr=0.0,
                     max_val=max_val, particle_count=len(cropped_patch_intersection))
    if max_val:
        p1 = np.unravel_index(np.argmax(filtered_particle), filtered_particle.shape)
        p2 = np.unravel_index(np.argmax(filtered_patch), filtered_patch.shape)
    else:
        p1 = np.unravel_index(np.argmin(filtered_particle), filtered_particle.shape)
        p2 = np.unravel_index(np.argmin(filtered_patch), filtered_patch.shape)

    save_double_radial_mean(particle=cropped_particle, with_var=cropped_patch,
                            filter_size=f_size, path=path, p1=p1, p2=p2, alpha=3)


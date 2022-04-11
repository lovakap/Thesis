import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.stats import wasserstein_distance
from skimage.draw import circle, ellipse, disk
from matplotlib.patches import Rectangle
import random
from typing import Tuple
import os
MEAN = 0.0
STD = 0.5


def smooth(y, points):
    box = np.ones(points) / points
    smoothed = np.convolve(y, box, mode='same')
    return smoothed


def radial_mean_with_center(image: np.ndarray, x: int = None, y: int = None,  take_max=False, alpha: int =3) -> np.array:
    shape = image.shape
    if x is None or y is None:
        if take_max:
            x, y = np.unravel_index(np.argmax(image), image.shape)
        else:
            x, y = np.unravel_index(np.argmin(image), image.shape)
    quad = int(shape[0]/4)
    x = np.clip(x, quad, shape[0] - quad)
    y = np.clip(y, quad, shape[1] - quad)

    num_of_radiuses = int(min(shape) / 2)
    shift_x = np.ceil(shape[0] / 2).astype(np.int) - x
    shift_y = np.ceil(shape[1] / 2).astype(np.int) - y
    shifted = np.roll(image, shift=(shift_x, shift_y), axis=(0, 1))
    xs, ys = np.ogrid[0:shape[0], 0:shape[1]]

    shifted = shifted - shifted.mean()
    img_nrm = image - image.mean()
    # radiuses = np.hypot(xs - shape[0] / 2, ys - shape[1] / 2)
    radiuses = np.hypot(xs - x, ys - y)
    rbin = (num_of_radiuses * radiuses / radiuses.max()).astype(np.int)
    radial_mean = np.asarray([np.mean(img_nrm[rbin <= i]) for i in np.arange(1, rbin.max() + 1)])
    # radial_mean = ndimage.mean(shifted, labels=rbin, index=np.arange(1, rbin.max() + 1))

    # return np.cumsum(radial_mean)
    # return smooth(radial_mean, alpha)
    return radial_mean, shifted


def get_func(name):
    if name == 'var':
        return np.var
    elif name == 'mean':
        return np.mean
    else:
        raise Exception('No such function')


def radial_mean_with_center_top_n(image: np.ndarray, x, y, func) -> np.array:
    shape = image.shape
    quad = int(shape[0]/4)
    mean_radial_mean = []
    true_shifted = None
    for i in range(len(x)):
        # x_i = np.clip(x[i], quad, shape[0] - quad)
        # y_i = np.clip(y[i], quad, shape[1] - quad)
        x_i = x[i]
        y_i = y[i]
        num_of_radiuses = int(min(shape) / 2)
        shift_x = np.ceil(shape[0] / 2).astype(np.int) - x_i
        shift_y = np.ceil(shape[1] / 2).astype(np.int) - y_i
        shifted = np.roll(image, shift=(shift_x, shift_y), axis=(0, 1))
        xs, ys = np.ogrid[0:shape[0], 0:shape[1]]

        shifted = shifted - shifted.mean()
        img_nrm = image - image.mean()
        # radiuses = np.hypot(xs - shape[0] / 2, ys - shape[1] / 2)
        radiuses = np.hypot(xs - x_i, ys - y_i)
        rbin = (num_of_radiuses * radiuses / radiuses.max()).astype(np.int)
        radial_mean = np.asarray([func(img_nrm[rbin <= i]) for i in np.arange(1, rbin.max() + 1)])
        mean_radial_mean.append(radial_mean[:2*quad - 1])
        if i == 0:
            true_shifted = shifted
    return np.mean(mean_radial_mean, axis=0), true_shifted


def fft_convolve_2d(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    fft_image = np.fft.fft2(image)
    fft_filter = np.fft.fft2(filter, s=image.shape)
    fft_out = np.multiply(fft_image, fft_filter)
    out = np.fft.ifft2(fft_out)
    roll = int(filter.shape[0] / 2)
    return np.roll(out.real, -roll, axis=[0, 1])


def get_image_der(image: np.array):
    Dx = (np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))/8
    Dy = (np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))/8

    Ix = fft_convolve_2d(image, Dx)
    Iy = fft_convolve_2d(image, Dy)

    return Ix, Iy


def get_image_der2(image: np.array):
    Dx = (np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))/8
    Dy = (np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))/8

    Ix = fft_convolve_2d(image, Dx+Dy)

    return Ix


def get_der_sum(image: np.array):
    Ix, Iy = get_image_der(image)
    der_sum = np.sum(np.abs(Ix)) + np.sum(np.abs(Iy))
    return der_sum


def circle_filter(kernel_size: int) -> np.array:
    im = np.zeros((kernel_size, kernel_size))
    cc, rr = disk((int(kernel_size / 2), int(kernel_size / 2)), int(kernel_size / 2))
    im[cc, rr] = 1
    return im


def linear_kernel(kernel_size: int, circle_cut: bool = False) -> np.array:
    if kernel_size % 2 == 0:
        kernel_size -= 1
    l = list(range(int(kernel_size/2) + 1))
    l.reverse()
    l = np.array(list(range(int(kernel_size/2))) + l[0:]).reshape(kernel_size, 1) + 1
    l = l / l.sum()
    l = np.dot(l, l.T)
    if circle_cut:
        l *= circle_filter(kernel_size=kernel_size)
    return l


def gaussian_kernel(kernel_size: int, circle_cut: bool = False, steepness: float = .25) -> np.array:
    g = gaussian(kernel_size, kernel_size * steepness).reshape(kernel_size, 1)
    # g = gaussian(kernel_size, kernel_size).reshape(kernel_size, 1)
    g = np.dot(g, g.T)
    if circle_cut:
        g *= circle_filter(kernel_size=kernel_size)
    return g/g.sum()


def get_empty_image(image_size: int, add_noise: bool = False, noise_std: float = STD,
                    noise_mean: float = MEAN) -> np.array:
    image = np.zeros((image_size, image_size))
    if add_noise:
        image, _ = apply_noise(image, noise_std, noise_mean)
    return image


def get_glued_image(image_size: int, sub_image_size: int, add_noise: bool = False,
                          blob_std: float = STD, noise_std: float = STD, blob_mean: float = MEAN,
                          noise_mean: float = MEAN) -> np.array:
    image = np.zeros((image_size, image_size))
    # pick random place for a particle in the image
    rand_x = random.choice(range(image_size - sub_image_size))
    rand_x2 = random.choice(range(image_size - sub_image_size))
    rand_y = random.choice(range(image_size - sub_image_size))
    rand_y2 = random.choice(range(image_size - sub_image_size))

    rn_val = np.random.normal(blob_mean, blob_std, image.shape)
    cc, rr = disk((rand_x + int(sub_image_size / 2), rand_y + int(sub_image_size / 2)), int(sub_image_size / 2))
    cc2, rr2 = disk((rand_x2 + int(sub_image_size / 2), rand_y2 + int(sub_image_size / 2)), int(sub_image_size / 2))
    # cc, rr = ellipse(rand_x + int(sub_image_size / 2), rand_y + int(sub_image_size / 2), int(sub_image_size / 4),
    #                  int(sub_image_size / 2))
    if blob_std == 0.:
        image[cc, rr] = 1.
        image[cc2, rr2] = 1.
    else:
        image[cc, rr] = rn_val[cc, rr]
        image[cc2, rr2] = rn_val[cc, rr]

    if add_noise:
        image, _ = apply_noise(image, noise_std, noise_mean)

    return image


def get_image_for_testing(image_size: int, sub_image_size: int, add_noise: bool = False,
                          blob_std: float = STD, noise_std: float = STD, blob_mean: float = MEAN,
                          noise_mean: float = MEAN, random_choise: bool = True) -> Tuple[np.array, Tuple[int, int], float]:
    image = np.zeros((image_size, image_size))
    # pick random place for a particle in the image
    if random_choise:
        rand_x = random.choice(range(image_size - sub_image_size))
        rand_y = random.choice(range(image_size - sub_image_size))
    else:
        rand_x = int(image_size / 4)
        rand_y = int(image_size / 4)
    rn_val = np.random.normal(blob_mean, blob_std, image.shape)
    cc, rr = disk((rand_x + int(sub_image_size / 2), rand_y + int(sub_image_size / 2)), int(sub_image_size / 2))
    # cc, rr = ellipse(rand_x + int(sub_image_size / 2), rand_y + int(sub_image_size / 2), int(sub_image_size / 4),
    #                  int(sub_image_size / 2))
    if blob_std == 0.:
        image[cc, rr] = 1.
        expected_signal = 1.
    else:
        image[cc, rr] = rn_val[cc, rr]
        expected_signal = np.mean(rn_val[cc, rr]**2)
    expected_noise = 1e-7
    if add_noise:
        image, expected_noise = apply_noise(image, noise_std, noise_mean)

    return image, (rand_x + int(sub_image_size / 2), rand_y + int(sub_image_size / 2)), expected_signal / expected_noise


def create_result_dir(text: str, text2: str='results/', create_sub_folder=False) -> str:
    os.makedirs(text2, exist_ok=True)
    path = text2 + text + '/'
    os.makedirs(path, exist_ok=True)
    if create_sub_folder:
        path = path + f'{np.random.randint(0,100000)}/'
        os.makedirs(path, exist_ok=True)
    return path


def save_plot(image: np.array, filter_size: int, path, snr: float, max_val=False, mark_center: bool = True):
    os.makedirs(path, exist_ok=True)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'filter size - {filter_size}, SNR - {float("{:.3f}".format(snr))}')
    if mark_center:
        if max_val:
            x, y = np.unravel_index(np.argmax(image), image.shape)
        else:
            x, y = np.unravel_index(np.argmin(image), image.shape)
        #todo : check why x and y swapped
        plt.scatter(y, x, color='red')
    plt.colorbar()
    plt.savefig(path + f'filter_size_{filter_size}.png', edgecolor='none')
    plt.close()


def save_double_plot(particle: np.array, random: np.array, filter_size: int, path, snr: float, max_val=False,
                     mark_center: bool = True, particle_count: int = 0):
    os.makedirs(path, exist_ok=True)
    fig = plt.figure()
    fig.suptitle(f'filter size - {filter_size}')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(particle)
    if mark_center:
        if max_val:
            a, b = np.unravel_index(np.argmax(particle), particle.shape)
        else:
            a, b = np.unravel_index(np.argmin(particle), particle.shape)
        ax1.scatter(b, a, color='r')
        # ax1.scatter(x, y, color='red')
    ax1.axis('off')
    ax1.title.set_text(f'Single'
                       f'\nMean: {float("{:.3f}".format(np.mean(particle)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(particle)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(particle)))}')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(random)
    if mark_center:
        if max_val:
            a, b = np.unravel_index(np.argmax(random), random.shape)
        else:
            a, b = np.unravel_index(np.argmin(random), random.shape)
        ax2.scatter(b, a, color='r')
    ax2.axis('off')
    ax2.title.set_text(f'Intersect with {particle_count} particles'
                       f'\nMean: {float("{:.3f}".format(np.mean(random)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(random)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(random)))}')
    # plt.colorbar()
    plt.savefig(path + f'filter_size_{filter_size}.png', edgecolor='none')
    plt.close()


def save_double_plot_top_n(particle: np.array, random: np.array, filter_size: int, path, p1, p2,
                     mark_center: bool = True, particle_count: int = 0):
    os.makedirs(path, exist_ok=True)
    fig = plt.figure()
    fig.suptitle(f'filter size - {filter_size} with top {len(p1[0])} points')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(particle)
    if mark_center:
        for i in range(len(p1[0])):
            ax1.scatter(p1[1][i], p1[0][i], color='r')
        # ax1.scatter(x, y, color='red')

    if len(p1) > 0:
        particle_center_var = np.sum((np.squeeze(p1).T - np.squeeze(p1).mean(axis=1)) ** 2) / len(p1)
        noise_center_var = np.sum((np.squeeze(p2).T - np.squeeze(p2).mean(axis=1)) ** 2) / len(p2)
    else:
        particle_center_var = 0
        noise_center_var = 0

    ax1.axis('off')
    ax1.title.set_text(f'Single'
                       f'\nMean: {float("{:.3f}".format(np.mean(particle)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(particle)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(particle)))}'
                       f'\n Center Var: {float("{:.3f}".format(particle_center_var))}')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(random)
    if mark_center:
        if mark_center:
            for i in range(len(p2[0])):
                ax2.scatter(p2[1][i], p2[0][i], color='r')
    ax2.axis('off')
    ax2.title.set_text(f'Intersect with {particle_count} particles'
                       f'\nMean: {float("{:.3f}".format(np.mean(random)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(random)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(random)))}'
                       f'\n Center Var: {float("{:.3f}".format(noise_center_var))}')
    # plt.colorbar()
    plt.savefig(path + f'filter_size_{filter_size}.png', edgecolor='none')
    plt.close()




def save_double_plot_var(particle: np.array, with_var: np.array, filter_size: int, path, snr: float, max_val=False, mark_center: bool = True, x=None, y=None):
    os.makedirs(path, exist_ok=True)
    fig = plt.figure()
    fig.suptitle(f'filter size - {filter_size}')
    ax2 = fig.add_subplot(1, 3, 3)
    # ax2.imshow(with_var)
    n, bins, patches = ax2.hist(with_var.flatten(), density=True, bins=200)
    if mark_center:
        if max_val:
            a, b = np.unravel_index(np.argmax(with_var), with_var.shape)
        else:
            a, b = np.unravel_index(np.argmin(with_var), with_var.shape)
        ax2.scatter(b, a, color='green')
    #     ax2.scatter(x, y, color='red')
    # ax2.axis('off')
    # ax2.title.set_text(f'With Var patching'

    ax2.title.set_text(f'Noise'
                       f'\nMean: {float("{:.3f}".format(np.mean(with_var)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(with_var)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(with_var)))}')
    ax1 = fig.add_subplot(1, 3, 1)
    # ax1.imshow(particle)
    n2, bins2, patches2 = ax1.hist(particle.flatten(), density=True, bins=bins)
    if mark_center:
        if max_val:
            a, b = np.unravel_index(np.argmax(particle), particle.shape)
        else:
            a, b = np.unravel_index(np.argmin(particle), particle.shape)
        ax1.scatter(b, a, color='green')
        # ax1.scatter(x, y, color='red')
    # ax1.axis('off')
    # ax1.title.set_text(f'Without Var patching'
    ax1.ylim(0, particle.max())
    ax1.title.set_text(f'Particle'
                       f'\nMean: {float("{:.3f}".format(np.mean(particle)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(particle)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(particle)))}')

    ax3 = fig.add_subplot(1, 3, 2)
    ax3.title.set_text(f'W : {float("{:.3f}".format(wasserstein_distance(n, n2)))}'
                       f'\nL2 : {float("{:.3f}".format(np.linalg.norm(n - n2)))}')
    ax3.hist(particle.flatten(), density=True, bins=bins)
    ax3.hist(with_var.flatten(), density=True, bins=bins)
    # plt.colorbar()
    plt.savefig(path + f'filter_size_{filter_size}.png', edgecolor='none')
    plt.close()


def save_double_power_spec(particle: np.array, with_var: np.array, filter_size: int, path, snr: float, max_val=False,
                           mark_center: bool = True):
    os.makedirs(path, exist_ok=True)
    fig = plt.figure()
    fig.suptitle(f'filter size - {filter_size}')
    ax1 = fig.add_subplot(1, 2, 1)
    fr, ps = get_ps(particle)
    fr2, ps2 = get_ps(with_var)
    m = max(ps.max(), ps2.max())
    ax1.plot(fr, ps)
    ax1.set_ylim(0, m)
    ax1.title.set_text(f'Particle'
                       f'\nMean: {float("{:.3f}".format(np.mean(particle)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(particle)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(particle)))}')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(fr2, ps2)
    ax2.set_ylim(0, m)
    ax2.title.set_text(f'Noise'
                       f'\nMean: {float("{:.3f}".format(np.mean(with_var)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(with_var)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(with_var)))}')

    # plt.colorbar()
    plt.savefig(path + f'filter_size_{filter_size}.png', edgecolor='none')
    plt.close()


def save_double_radial_mean(particle: np.array, with_var: np.array, filter_size: int, path, p1, p2, max_val=False,
                           mark_center: bool = True, alpha: int = 3):

    rd_means_particle, shifted_particle = radial_mean_with_center(image=particle, x=p1[0], y=p1[1], alpha=alpha)
    rd_means_noise, shifted_noise = radial_mean_with_center(image=with_var, x=p2[0], y=p2[1], alpha=alpha)
    m1 = min(rd_means_noise.min(), rd_means_particle.min())
    m2 = max(rd_means_noise.max(), rd_means_particle.max())
    os.makedirs(path, exist_ok=True)
    fig = plt.figure()
    fig.suptitle(f'filter size - {filter_size}')
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(rd_means_particle)
    ax1.set_ylim(m1, m2)
    ax1.title.set_text(f'Particle')
                       # f'\nMean: {float("{:.3f}".format(np.mean(particle)))}'
                       # f'\nSTD: {float("{:.3f}".format(np.std(particle)))}'
                       # f'\nDer Sum: {float("{:.3f}".format(get_der_sum(particle)))}')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(rd_means_noise)
    ax2.set_ylim(m1, m2)
    ax2.title.set_text(f'Noise')
                       # f'\nMean: {float("{:.3f}".format(np.mean(with_var)))}'
                       # f'\nSTD: {float("{:.3f}".format(np.std(with_var)))}'
                       # f'\nDer Sum: {float("{:.3f}".format(get_der_sum(with_var)))}')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(shifted_particle, cmap='gray')
    ax3.title.set_text(f'Shifted Particle')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(shifted_noise, cmap='gray')
    ax4.title.set_text(f'Shifted Noise')

    # plt.colorbar()
    plt.savefig(path + f'radial_mean_{filter_size}.png', edgecolor='none')
    # plt.show()
    plt.close()


def save_double_radial_mean_top_n(particle: np.array, with_var: np.array, filtered_particle: np.array,
                                  filtered_with_var: np.array, filter_size: int, path, p1, p2):
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MEAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
    rd_means_particle, shifted_particle = radial_mean_with_center_top_n(image=particle, x=p1[0], y=p1[1],
                                                                        func=get_func('mean'))
    rd_means_noise, shifted_noise = radial_mean_with_center_top_n(image=with_var, x=p2[0], y=p2[1],
                                                                  func=get_func('mean'))
    m1 = min(rd_means_noise.min(), rd_means_particle.min())
    m2 = max(rd_means_noise.max(), rd_means_particle.max())

    rd_means_particle_filtered, shifted_particle_filtered = radial_mean_with_center_top_n(image=filtered_particle,
                                                                                          x=p1[0], y=p1[1],
                                                                                          func=get_func('mean'))
    rd_means_noise_filtered, shifted_noise_filtered = radial_mean_with_center_top_n(image=filtered_with_var, x=p2[0],
                                                                                    y=p2[1], func=get_func('mean'))
    m1_filtered = min(rd_means_noise_filtered.min(), rd_means_particle_filtered.min())
    m2_filtered = max(rd_means_noise_filtered.max(), rd_means_particle_filtered.max())

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VAR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
    rd_var_particle, _ = radial_mean_with_center_top_n(image=particle, x=p1[0], y=p1[1],
                                                                        func=get_func('var'))
    rd_var_noise, _ = radial_mean_with_center_top_n(image=with_var, x=p2[0], y=p2[1],
                                                                  func=get_func('var'))
    m1_var = min(rd_var_noise.min(), rd_var_particle.min())
    m2_var = max(rd_var_noise.max(), rd_var_particle.max())

    rd_var_particle_filtered, _ = radial_mean_with_center_top_n(image=filtered_particle,
                                                                                          x=p1[0], y=p1[1],
                                                                                          func=get_func('var'))
    rd_var_noise_filtered, _ = radial_mean_with_center_top_n(image=filtered_with_var, x=p2[0],
                                                                                    y=p2[1], func=get_func('var'))
    m1_var_filtered = min(rd_var_noise_filtered.min(), rd_var_particle_filtered.min())
    m2_var_filtered = max(rd_var_noise_filtered.max(), rd_var_particle_filtered.max())


    os.makedirs(path, exist_ok=True)
    fig = plt.figure(figsize=(8, 8), dpi=80)
    fig.suptitle(f'filter size - {filter_size}, with mean of top {len(p1[0])} points')
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(rd_means_particle)
    ax1.set_ylim(m1, m2)
    ax1.set(xticklabels=[])
    ax1.tick_params(bottom=False)
    ax1.title.set_text(f'Mean Particle')

    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(rd_means_noise)
    ax2.set_ylim(m1, m2)
    ax2.set(xticklabels=[])
    ax2.tick_params(bottom=False)
    ax2.title.set_text(f'Mean Noise')

    ax3 = fig.add_subplot(4, 2, 3)
    ax3.plot(rd_means_particle_filtered)
    ax3.set_ylim(m1_filtered, m2_filtered)
    ax3.set(xticklabels=[])
    ax3.tick_params(bottom=False)
    ax3.title.set_text(f'Mean Filtered Particle')

    ax4 = fig.add_subplot(4, 2, 4)
    ax4.plot(rd_means_noise_filtered)
    ax4.set_ylim(m1_filtered, m2_filtered)
    ax4.set(xticklabels=[])
    ax4.tick_params(bottom=False)
    ax4.title.set_text(f'Mean Filtered Noise')

    ax5 = fig.add_subplot(4, 2, 5)
    ax5.plot(rd_var_particle)
    ax5.set_ylim(m1_var, m2_var)
    ax5.set(xticklabels=[])
    ax5.tick_params(bottom=False)
    ax5.title.set_text(f'Var Particle')

    ax6 = fig.add_subplot(4, 2, 6)
    ax6.plot(rd_var_noise)
    ax6.set_ylim(m1_var, m2_var)
    ax6.set(xticklabels=[])
    ax6.tick_params(bottom=False)
    ax6.title.set_text(f'Var Noise')

    ax7 = fig.add_subplot(4, 2, 7)
    ax7.plot(rd_var_particle_filtered)
    ax7.set_ylim(m1_var_filtered, m2_var_filtered)
    ax7.title.set_text(f'Var Filtered Particle')

    ax8 = fig.add_subplot(4, 2, 8)
    ax8.plot(rd_var_noise_filtered)
    ax8.set_ylim(m1_var_filtered, m2_var_filtered)
    ax8.title.set_text(f'Var Filtered Noise')

    # ax3 = fig.add_subplot(2, 2, 3)
    # ax3.imshow(shifted_particle, cmap='gray')
    # ax3.title.set_text(f'Shifted Particle')
    #
    # ax4 = fig.add_subplot(2, 2, 4)
    # ax4.imshow(shifted_noise, cmap='gray')
    # ax4.title.set_text(f'Shifted Noise')

    # plt.colorbar()
    plt.savefig(path + f'radial_mean_{filter_size}.png', edgecolor='none')
    # plt.show()
    plt.close()




def save_triple_plot(single_object: np.array, empty: np.array, glued: np.array, filter_size: int,
                     path, snr: float, x=None, y=None, max_val=False, mark_center: bool = True):
    os.makedirs(path, exist_ok=True)

    fig = plt.figure()
    fig.suptitle(f'filter size - {filter_size}, SNR - {float("{:.3f}".format(snr))}')
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(single_object)
    if mark_center:
        if max_val:
            a, b = np.unravel_index(np.argmax(single_object), single_object.shape)
        else:
            a, b = np.unravel_index(np.argmin(single_object), single_object.shape)
        ax1.scatter(b, a, color='green')
        ax1.scatter(x, y, color='red')
    ax1.axis('off')
    ax1.title.set_text(f'With Object'
                       f'\nMean: {float("{:.3f}".format(np.mean(single_object)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(single_object)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(single_object)))}')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(empty)
    ax2.axis('off')
    ax2.title.set_text(f'Without Object'
                       f'\nMean: {float("{:.3f}".format(np.mean(empty)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(empty)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(empty)))}')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(glued)
    ax3.axis('off')
    ax3.title.set_text(f'Glued Object'
                       f'\nMean: {float("{:.3f}".format(np.mean(glued)))}'
                       f'\nSTD: {float("{:.3f}".format(np.std(glued)))}'
                       f'\nDer Sum: {float("{:.3f}".format(get_der_sum(glued)))}')
    # fig.colorbar()
    fig.savefig(path + f'filter_size_{filter_size}.png', edgecolor='none')
    plt.close()


def apply_noise(image: np.ndarray, noise_std: float, noise_mean: float) -> Tuple[np.ndarray, float]:
    noise = np.random.normal(noise_mean, noise_std, image.shape)
    image += noise
    # image += np.random.rand(image.shape[0], image.shape[1])
    # image = np.clip(image, 0, 1)
    # image -= image.min()
    # image /= image.max()
    return image, np.mean(noise**2)


def estimate_noise_histogram(image: np.ndarray):
    circle_radius = int(image.shape[1] / 2)
    cc, rr = disk((circle_radius, circle_radius), circle_radius)
    mask = np.ones(image.shape, dtype=bool)
    mask[cc, rr] = False
    vals = image[mask]
    return vals


def get_patched_var(image: np.ndarray, blob_size: int):
    std_array = np.zeros((blob_size, blob_size))
    for i in range(blob_size):
        for j in range(blob_size):
            std_array[i, j] = image[i:i + image.shape[0] - blob_size, j:j + image.shape[1] - blob_size].var()

    return std_array


def get_ps(df, time_step: float = 1/100):
    ps = np.abs(np.fft.fft2(df))**2
    freqs = np.fft.fftfreq(df.size, time_step)
    idx = np.argsort(freqs)
    ps = ps.flatten()
    return (freqs[idx], ps[idx])


def plot_image_with_marks(graph, coordinates, plot_title, patch_size=100,):
    fig, ax = plt.subplots()
    ax.imshow(graph)
    for c in coordinates:
        p = (c[0] - int(patch_size/2), c[1] - int(patch_size/2))
        rect = Rectangle(p, patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.title(plot_title)
    plt.show()
    plt.close()
# h = get_image_for_testing(100, 11)
# plt.matshow(h)
# plt.show()
# print('end')

import numpy as np
from scipy.signal import convolve2d
from Utils import gaussian_kernel, linear_kernel, circle_filter


def apply_filter(image: np.ndarray, filter_size: int, circle_cut: bool = False) -> np.ndarray:
    filtered_image = image.copy()
    kernel_filter = gaussian_kernel(filter_size, circle_cut=circle_cut)
    # kernel_filter = linear_kernel(filter_size, circle_cut=circle_cut)
    # kernel_filter = circle_filter(filter_size)
    # filtered_image = convolve2d(image, -kernel_filter, mode='same')
    # filtered_image = fft_convolve_2d(image, -kernel_filter)
    filtered_image = fft_convolve_2d(filtered_image, -kernel_filter)
    return filtered_image


def fft_convolve_2d(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    fft_image = np.fft.fft2(image)
    fft_filter = np.fft.fft2(filter, s=image.shape)
    fft_out = np.multiply(fft_image, fft_filter)
    out = np.fft.ifft2(fft_out)
    roll = int(filter.shape[0] / 2)
    return np.roll(out.real, -roll, axis=[0, 1])

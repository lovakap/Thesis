import numpy as np
from Utils import read_mrc
import matplotlib.pyplot as plt
from pyfftw.interfaces.numpy_fft import fft2, ifft2, rfft2, irfft2, fftshift

PATCH_SIZE = 100
HALF_PATH_SIZE = int(PATCH_SIZE / 2)


def radius_norm(n: int, origin=None):
    """
        Create an n(1) x n(2) array where the value at (x,y) is the distance from the
        origin, normalized such that a distance equal to the width or height of
        the array = 1.  This is the appropriate function to define frequencies
        for the fft of a rectangular image.
        For a square array of size n (or [n n]) the following is true:
        RadiusNorm(n) = Radius(n)/n.
        The org argument is optional, in which case the FFT center is used.
        Theta is the angle in radians.
        (Transalted from Matlab RadiusNorm.m)
    """

    if isinstance(n, int):
        n = np.array([n, n])

    if origin is None:
        origin = np.ceil((n + 1) / 2)

    a, b = origin[0], origin[1]
    y, x = np.meshgrid(np.arange(1-a, n[0]-a+1)/n[0],
                       np.arange(1-b, n[1]-b+1)/n[1])  # zero at x,y
    radius = np.sqrt(x ** 2 + y ** 2)

    theta = np.arctan2(x, y)

    return radius, theta


def fortran_to_c(stack):
    """ Convert Fortran-contiguous array to C-contiguous array. """
    return stack.T if stack.flags.f_contiguous else stack


def fctr(n):
    """ Center of an FFT-shifted image. We use this center
        coordinate for all rotations and centering operations. """

    if isinstance(n, int):
        n = np.array([n, n])

    return np.ceil((n + 1) / 2)


def voltage_to_wavelength(voltage):
    # aspire matlab version
    wave_length = 1.22639 / np.sqrt(voltage * 1000 + 0.97845 * voltage ** 2)

    # cov3d matlab version
    # wave_length = 1.22643247 / np.sqrt(voltage * 1000 + 0.978466 * voltage ** 2)
    return wave_length


def get_coordinates(path):
    points = []
    with open(path) as f:
        for line in f:
            x, y = line.split()
            points.append((int(x), int(y)))
    return points


def crop_random_particle(micrograph, coordinates):
    idx = np.random.choice(len(coordinates))
    p = coordinates[idx]
    return micrograph[p[0] - HALF_PATH_SIZE: p[0] + HALF_PATH_SIZE, p[1] - HALF_PATH_SIZE: p[1] + HALF_PATH_SIZE]

file_name = 'Falcon_2012_06_12-14_57_34_0'
file_path = 'Data/mrc_files/' + file_name
full_micrograph = read_mrc(file_path + '.mrc')
coordinates = get_coordinates(file_path + '.coord')
resolution = full_micrograph.shape[0]
voltage = 300
pixel_size = 1.77
wave_length = voltage_to_wavelength(voltage)
square_side = resolution
DefocusU = 30080.150391
DefocusV = 29487.076172
spherical_aberration = 2.0
DefocusAngle = -72.797336
phase_shift = 3.052821
amplitude_contrast = 0.07
bw = 1 / (pixel_size / 10)

s, theta = radius_norm(square_side, origin=fctr(square_side))
astigmatism_angle = np.full(
    shape=theta.shape, fill_value=DefocusAngle, dtype=np.float32)

defocus_sum = np.full(
    shape=theta.shape, fill_value=DefocusU + DefocusV, dtype=np.float32)

defocus = defocus_sum + ((DefocusU - DefocusV) * np.cos(2 * (theta - astigmatism_angle)))

r_ctf = s * (10 / pixel_size)
lmbd = wave_length / 10.0
defocus_factor = np.pi * lmbd * r_ctf * defocus / 2
amplitude_contrast_term = amplitude_contrast / np.sqrt(1 - amplitude_contrast ** 2)

chi = (defocus_factor - np.pi * lmbd ** 3 * spherical_aberration * 1e6 * r_ctf ** 2 / 2 + amplitude_contrast_term)

ctf = -np.sin(chi)

imhat = fft2(full_micrograph)
imhat = np.multiply(imhat, np.sign(ctf))
imhat = fftshift(imhat)
new_graph = irfft2(imhat, s=full_micrograph.shape)
# new_graph = new_graph.real
plt.imshow(new_graph)
plt.show()

cropped_particle = crop_random_particle(micrograph=new_graph, coordinates=coordinates)
plt.imshow(cropped_particle)
plt.show()

cropped_particle = crop_random_particle(micrograph=full_micrograph, coordinates=coordinates)
plt.imshow(cropped_particle)
plt.show()

print('end')
import mrcfile
import numpy as np
import matplotlib.patches as patches
from pyfftw.interfaces.numpy_fft import fft2, ifft2, rfft2, irfft2, fftshift
PATCH_SIZE = 100
HALF_PATH_SIZE = int(PATCH_SIZE / 2)
MRC_CONFIG = {
    'Falcon_2012_06_12-14_57_34_0': {
        'voltage': 300,
        'pixel_size': 1.77,
        'DefocusU': 30080.150391,
        'DefocusV': 29487.076172,
        'spherical_aberration': 2.0,
        'DefocusAngle': -72.797336,
        'phase_shift': -0.348,
        'amplitude_contrast': 0.07
    },
    'Falcon_2012_06_12-14_33_35_0': {
        'voltage': 300,
        'pixel_size': 1.77,
        'DefocusU': 35058.11,
        'DefocusV': 34003.71,
        'spherical_aberration': 2.0,
        'DefocusAngle': -67.312701,
        'phase_shift': 175.128,
        'amplitude_contrast': 0.07
    },
    'Falcon_2012_06_12-15_07_41_0': {
        'voltage': 300,
        'pixel_size': 1.77,
        'DefocusU': 50173.44,
        'DefocusV': 49290.66,
        'spherical_aberration': 2.0,
        'DefocusAngle': -49.72,
        'phase_shift': 9.003,
        'amplitude_contrast': 0.07
    },
    '002': {
        'voltage': 300,
        'pixel_size': 1.77,
        'DefocusU': 31719.705078,
        'DefocusV': 31404.576172,
        'spherical_aberration': 2.0,
        'DefocusAngle': -38.037568,
        'phase_shift': -0.008433,
        'amplitude_contrast': 0.07
    }
}

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
    intersection_particles = contain_particle(p[0], p[1], coordinates)
    return micrograph[p[0] - HALF_PATH_SIZE: p[0] + HALF_PATH_SIZE, p[1] - HALF_PATH_SIZE: p[1] + HALF_PATH_SIZE]\
        , p, intersection_particles


def contain_particle(x, y, coordinates):
    coordinates = np.array(coordinates)
    xs = coordinates[:, 0]
    ys = coordinates[:, 1]
    intersection_x = np.logical_or(
        np.logical_and(xs - HALF_PATH_SIZE < x - HALF_PATH_SIZE, x - HALF_PATH_SIZE < xs + HALF_PATH_SIZE),
        np.logical_and(xs - HALF_PATH_SIZE < x + HALF_PATH_SIZE, x + HALF_PATH_SIZE < xs + HALF_PATH_SIZE))

    intersection_y = np.logical_or(
        np.logical_and(ys - HALF_PATH_SIZE < y - HALF_PATH_SIZE, y - HALF_PATH_SIZE < ys + HALF_PATH_SIZE),
        np.logical_and(ys - HALF_PATH_SIZE < y + HALF_PATH_SIZE, y + HALF_PATH_SIZE < ys + HALF_PATH_SIZE))

    intersection = np.logical_and(intersection_x, intersection_y)

    return np.where(intersection)[0]


def crop_random_patch(micrograph, coordinates):
    x = np.random.randint(0, micrograph.shape[0])
    y = np.random.randint(0, micrograph.shape[1])
    intersection_particles = contain_particle(x, y, coordinates)
    return micrograph[x - HALF_PATH_SIZE: x + HALF_PATH_SIZE, y - HALF_PATH_SIZE: y + HALF_PATH_SIZE], (x, y)\
        , intersection_particles


def add_patches(p, intersections):
    patches_list = []
    for i in intersections:
        x = i[0] - p[0]
        # x = np.clip(p[0] - i[0], 0, PATCH_SIZE)
        # y = np.clip(p[1] - i[1], 0, PATCH_SIZE)
        y = i[1] - p[1]
        # if x == 0 and y == 0:
        #     continue
        patches_list.append(patches.Circle((x + HALF_PATH_SIZE, y + HALF_PATH_SIZE), HALF_PATH_SIZE, linewidth='1', edgecolor='r', facecolor='None'))
        # patches_list.append(patches.Circle((y + HALF_PATH_SIZE, x + HALF_PATH_SIZE), HALF_PATH_SIZE, linewidth='1', edgecolor='r', facecolor='None'))
    return patches_list


def apply_patches(coordinates):
    patches_list = []
    for i in coordinates:
        patches_list.append(patches.Circle(i, HALF_PATH_SIZE, linewidth='0.5', edgecolor='r', facecolor='None'))
    return patches_list


def read_mrc(file_path: str, normalize: bool = True) -> np.ndarray:
    mrc = np.ascontiguousarray(mrcfile.open(file_path).data.T)
    if normalize:
        mrc -= np.min(mrc)
        mrc /= np.max(mrc)
    return mrc


def phase_shift(fimage, dx, dy):
    dims = fimage.shape
    x, y = np.meshgrid(np.arange(-dims[1] / 2, dims[1] / 2), np.arange(dims[0] / 2, dims[0] / 2))
    kx = -1j * 2 * np.pi * x / dims[1]
    ky = -1j * 2 * np.pi * y / dims[0]
    shifted_image = fimage * np.exp(-((kx * dx) + (ky * dy)))
    return shifted_image


def apply_ctf_on(mrc: np.ndarray, file_name: str) -> np.ndarray:
    mrc_info = MRC_CONFIG.get(file_name)
    assert mrc_info is not None, f'No confiig for {file_name}'

    square_side = mrc.shape[0]
    wave_length = voltage_to_wavelength(mrc_info.get('voltage'))
    s, theta = radius_norm(square_side, origin=fctr(square_side))
    astigmatism_angle = np.full(
        shape=theta.shape, fill_value=mrc_info.get('DefocusAngle'), dtype=np.float32)

    defocus_sum = np.full(
        shape=theta.shape, fill_value=mrc_info.get('DefocusU') + mrc_info.get('DefocusV'), dtype=np.float32)

    defocus = defocus_sum + (mrc_info.get('DefocusU') - mrc_info.get('DefocusV') * np.cos(2 * (theta - astigmatism_angle)))

    r_ctf = s * (10 / mrc_info.get('pixel_size'))
    lmbd = wave_length / 10.0
    defocus_factor = np.pi * lmbd * r_ctf * defocus / 2

    amplitude_contrast_term = mrc_info.get('amplitude_contrast') / np.sqrt(1 - mrc_info.get('amplitude_contrast') ** 2)

    chi = (defocus_factor - np.pi * lmbd ** 3 * mrc_info.get('spherical_aberration') * 1e6 * r_ctf ** 2 / 2 + amplitude_contrast_term)

    h = -np.sin(chi)
    imhat = fft2(mrc)
    imhat = np.multiply(imhat, np.sign(h))
    from scipy.ndimage.fourier import fourier_shift
    # imhat = phase_shift(imhat, mrc_info.get('phase_shift'), mrc_info.get('phase_shift'))
    imhat = fourier_shift(imhat, mrc_info.get('phase_shift'))
    new_graph = irfft2(imhat, s=mrc.shape)

    return new_graph

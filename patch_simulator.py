import random
from scipy.stats import norm
import numpy as np
from skimage.draw import ellipse, disk
from skimage.transform import resize
from scipy.signal import convolve2d

from particle_projection import ProjectionSample
from Utils import gaussian_kernel
MEAN = 0.0
STD = 0.5


class PatchSimulator:
    def __init__(self, particle_size: int, particle_ratio: float = None, projection_path: str = None, normalize=False,
                 blob_type='disk', blob_mean: float = MEAN, blob_std: float = STD, noise_mean: float = MEAN,
                 noise_std: float = STD, particle_vals_scale: float = 1.):
        self.particle_size = particle_size
        self.particle_ratio = particle_ratio
        self.patch_size = int(self.particle_size / self.particle_ratio)
        self.normalize = normalize

        self.blob_type = blob_type
        self.blob_mean = blob_mean
        self.blob_std = blob_std

        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.p_sample = None if projection_path is None else ProjectionSample(projection_path)
        self.p_scale = particle_vals_scale
        self.clean_data = None

    def get_patch(self, patch_type: str):

        if patch_type == 'blob':
            patch, center = self.get_blob()
        elif patch_type == 'projection':
            patch, center = self.get_projection()
            patch *= self.p_scale
            patch = np.concatenate([patch, np.zeros(patch.shape)], axis=1)
            # shift = int(self.patch_size * 0.5)
            shift = 0
            patch = patch[:, shift: self.patch_size + shift]
            center = (center[0], center[1] - shift)
        elif patch_type == 'empty':
            patch, center = self.get_empty(), (np.array([0]), np.array([0]))

        if patch_type != 'empty':
            if self.normalize:
                patch /= max(abs(patch))
            self.clean_data = patch

        patch, snr = self.apply_noise(patch)
        return patch, center, snr

    def get_blob(self):
        blob = np.zeros((self.patch_size, self.patch_size))

        # Choose random center
        rand_x = np.random.randint(0, (self.patch_size - self.particle_size))
        rand_y = np.random.randint(0, (self.patch_size - self.particle_size))

        # Generate random values for the blob with given mean and std
        rand_vals = np.random.normal(self.blob_mean, self.blob_std, blob.shape)

        half_blob_size = int(self.particle_size / 2)
        if self.blob_type == 'disk':
            cc, rr = disk((rand_x + half_blob_size, rand_y + half_blob_size), half_blob_size)
        elif self.blob_type == 'ellipse':
            cc, rr = ellipse((rand_x + half_blob_size, rand_y + half_blob_size), int(half_blob_size / 2),
                             half_blob_size)
        else:
            raise Exception(f'There is no blob type : {self.blob_type}')

        if self.blob_std == 0.0:
            blob[cc, rr] = self.blob_mean
        else:
            blob[cc, rr] = rand_vals[cc, rr]

        return blob, (np.array([rand_x + half_blob_size]), np.array([rand_y + half_blob_size]))

    def get_empty(self):
        return np.zeros((self.patch_size, self.patch_size))

    def get_projection(self):
        assert self.p_sample is not None, 'No projection path was given'
        sample, center = self.p_sample.get_sample()
        x_ratio, y_ratio = sample.shape[0]/self.patch_size, sample.shape[1]/self.patch_size
        approx_x, approx_y = int(center[0] / x_ratio), int(center[1] / y_ratio)

        # Applying deblurring based on resize of the particle
        deblurring_ratio = 1. - (sample.shape[0] / self.patch_size)
        sample = resize(sample, (self.patch_size, self.patch_size))
        sample = convolve2d(sample, gaussian_kernel(11, steepness=deblurring_ratio*0.25), mode='same')

        return sample, (np.array([approx_x]), np.array([approx_y]))

    def apply_noise(self, patch):
        # noise = np.random.normal(self.noise_mean, self.noise_std, patch.shape)
        # noise = np.random.uniform(-self.noise_std, self.noise_std, patch.shape)
        # noise = random.gauss(self.noise_mean, self.noise_std, patch.shape)
        noise = norm.rvs(self.noise_mean, self.noise_std, patch.shape)
        snr = np.mean(patch**2) / np.mean(noise**2)
        return patch + noise, snr

import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.measurements import center_of_mass


class ProjectionSample:
    def __init__(self, file_path: str, normalize: bool = False):
        self.volume = np.load(file_path)
        self.shape = self.volume.shape

    def get_sample(self, angles=None):
        if angles is None:
            angles = (int(np.random.random(1) * 360), int(np.random.random(1) * 360))

        first_rotation = rotate(self.volume, axes=(0, 1), angle=angles[0])
        second_rotation = rotate(first_rotation, axes=(1, 2), angle=angles[1])

        if second_rotation.shape != self.shape:
            delta_x = np.max(int((second_rotation.shape[0] - self.shape[0]) / 2), 0)
            delta_y = np.max(int((second_rotation.shape[1] - self.shape[1]) / 2), 0)
            delta_z = np.max(int((second_rotation.shape[2] - self.shape[2]) / 2), 0)
            second_rotation = second_rotation[delta_x:delta_x + self.shape[0], delta_y:delta_y + self.shape[1],
                              delta_z:delta_z + self.shape[2]]
        c_of_mass = center_of_mass(second_rotation)
        return second_rotation.mean(axis=-1), np.round(c_of_mass).astype(np.int)[:-1]

import numpy as np
from scipy import signal


def get_1d_signal(length: int) -> np.array:
    return signal.square(length)
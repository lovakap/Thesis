from typing import List
import numpy as np
import os


def get_filter_sizes(ratios : List[float], patch_size: int):
    filter_sizes = [int(patch_size * i) for i in ratios]
    filter_sizes = [f_size - 1 if f_size % 2 == 0 else f_size for f_size in filter_sizes]
    return filter_sizes


def get_n_points(patch: np.ndarray, n: int = 1, max_values: bool = True):
    if max_values:
        points = np.unravel_index(np.argsort(patch.flatten())[-n:], patch.shape)
    else:
        points = np.unravel_index(np.argsort(patch.flatten())[:n], patch.shape)
    return points


def create_result_dir(text: str, text2: str = 'results/', create_sub_folder=False) -> str:
    os.makedirs(text2, exist_ok=True)
    path = text2 + text + '/'
    os.makedirs(path, exist_ok=True)
    if create_sub_folder:
        path = path + f'{np.random.randint(0,100000)}/'
        os.makedirs(path, exist_ok=True)
    return path


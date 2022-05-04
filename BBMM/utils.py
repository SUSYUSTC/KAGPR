import typing as tp
import numpy as np
from collections.abc import Iterable

T = tp.TypeVar('T')
general_float = tp.Union[float, np.float]


def make_desired_size(value: tp.Union[T, tp.List[T]], n: int) -> tp.List[T]:
    result: tp.List[T]
    if isinstance(value, Iterable):
        result = tp.cast(tp.List[T], value)
    else:
        result = [value] * n
    assert len(result) == n
    return result


try:
    import cupy as cp
    gpu_available = True
except BaseException:
    gpu_available = False


def get_array_module(x):
    if gpu_available:
        return cp.get_array_module(x)
    else:
        return np


def print_dict(d: tp.Dict[tp.Any, tp.Any], level=1, **printoptions):
    for key in d:
        if isinstance(d[key], dict):
            print_dict(d[key], level=level + 1, **printoptions)
        else:
            print(' ' * level * 4, key, d[key], **printoptions)


def split_by_onetime_number(X: np.ndarray, onetime_number: int):
    lengths = [len(item) for item in X]
    N = len(lengths)
    total = 0
    current_index = 0
    current_total = 0
    result = []
    for i in range(len(lengths)):
        total += lengths[i]
        if total - current_total > onetime_number:
            result.append(slice(current_index, i + 1))
            current_index = i + 1
            current_total = total
    if current_index != N:
        result.append(slice(current_index, N))
    return result


def apply_recursively(method, X):
    if isinstance(X, list):
        return [apply_recursively(method, item) for item in X]
    elif isinstance(X, tuple):
        return tuple([apply_recursively(method, item) for item in X])
    elif isinstance(X, np.ndarray) and X.dtype == object:
        return [apply_recursively(method, item) for item in X]
    elif isinstance(X, np.ndarray):
        return method(X)
    elif gpu_available and isinstance(X, cp.ndarray):
        return method(X)
    else:
        assert False

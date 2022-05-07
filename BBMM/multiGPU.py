import cupy as cp
import numpy as np
from . import utils


class copyed_array:
    def __init__(self, data, nGPUs):
        self.nGPUs = nGPUs
        self.length = len(data)
        self.data = [None for i in range(nGPUs)]
        for i in range(self.nGPUs):
            with cp.cuda.Device(i):
                self.data[i] = utils.apply_recursively(cp.asarray, data)

    def __len__(self):
        return self.length


class splited_array:
    def __init__(self, data, nGPUs):
        self.nGPUs = nGPUs
        self.length = len(data)
        self.division = utils.make_slices(self.length, (self.length-1) // nGPUs + 1)
        self.data = [None for i in range(nGPUs)]
        for i, d in enumerate(self.division):
            with cp.cuda.Device(i):
                self.data[i] = utils.apply_recursively(cp.asarray, data[d])

    def __len__(self):
        return self.length


def clear_cache(nGPUs):
    for i in range(nGPUs):
        with cp.cuda.Device(i):
            cp._default_memory_pool.free_all_blocks()

import typing as tp
from .. import utils
class Param(object):
    def __init__(self, name: str, value: utils.general_float):
        self.name = name
        self.value = float(value)


def group_params(params: tp.List[Param]):
    n = len(params)
    unique_params: tp.List[Param] = []
    indices = []
    for i in range(n):
        position = utils.where_is(params[i], unique_params)
        if position == -1:
            unique_params.append(params[i])
            indices.append([i])
        else:
            indices[position].append(i)
    return unique_params, indices

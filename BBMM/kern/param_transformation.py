import typing as tp
import numpy as np
import warnings


class Transformation(object):
    def __init__(self) -> None:
        pass

    def __call__(self, x: float) -> float:
        raise NotImplementedError

    def d(self, x: float) -> float:
        raise NotImplementedError

    def inv(self, x: float) -> float:
        raise NotImplementedError


class Linear(Transformation):
    def __init__(self) -> None:
        pass

    def __call__(self, x: float) -> float:
        return x

    def d(self, x: float) -> float:
        return 1.0

    def inv(self, x: float) -> float:
        return x


class Log(Transformation):
    def __init__(self) -> None:
        pass

    def __call__(self, x: float) -> float:
        return float(np.log(x))

    def d(self, x: float) -> float:
        return 1.0 / x

    def inv(self, x: float) -> float:
        return float(np.exp(x))


linear = Linear()
log = Log()


class Group(object):
    def __init__(self, group: tp.List[Transformation]) -> None:
        self.group = group
        self.n = len(group)

    def __call__(self, x: tp.List[float]) -> tp.List[float]:
        assert len(x) == self.n
        return [self.group[i](x[i]) for i in range(self.n)]

    def d(self, x: tp.List[float]) -> tp.List[float]:
        assert len(x) == self.n
        return [self.group[i].d(x[i]) for i in range(self.n)]

    def inv(self, x: tp.List[float]) -> tp.List[float]:
        assert len(x) == self.n
        return [self.group[i].inv(x[i]) for i in range(self.n)]

    def transform_bounds(self, x: tp.List[tp.Tuple[float, float]]) -> tp.List[tp.Tuple[float, float]]:
        assert len(x) == self.n
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in log")
            return [tuple([self.group[i](item) for item in x[i]]) for i in range(self.n)]

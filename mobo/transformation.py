from typing import List, Tuple

from abc import ABC, abstractmethod
from bayes_optim.search_space.search_space import SearchSpace
from bayes_optim.search_space.variable import Real

import numpy as np
from sklearn.preprocessing import StandardScaler

"""
Data transformations (normalizations) for fitting surrogate model
"""


### 1-dim scaler


class Scaler(ABC):
    def fit(self, X):
        return self

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def inverse_transform(self, X):
        pass


class BoundedScaler(Scaler):
    """
    Scale data to [0, 1] according to bounds
    """

    def __init__(self, bounds):
        self.bounds = bounds

    def transform(self, X):
        return np.clip((X - self.bounds[0]) / (self.bounds[1] - self.bounds[0]), 0, 1)

    def inverse_transform(self, X):
        return np.clip(X, 0, 1) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]


class MockupScaler(Scaler):
    """Empty scaler that does nothing"""

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


### 2-dim transformation


class Transformation:
    def __init__(self, x_scaler, y_scaler):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def fit(self, x, y):
        self.x_scaler = self.x_scaler.fit(x)
        self.y_scaler = self.y_scaler.fit(y)

    def do(self, x=None, y=None):
        assert x is not None or y is not None
        if x is not None:
            x_res = self.x_scaler.transform(np.atleast_2d(x))
            if len(np.array(x).shape) < 2:
                x_res = x_res.squeeze()

        if y is not None:
            y_res = self.y_scaler.transform(np.atleast_2d(y))
            if len(np.array(y).shape) < 2:
                y_res = y_res.squeeze()

        if x is not None and y is not None:
            return x_res, y_res
        elif x is not None:
            return x_res
        elif y is not None:
            return y_res

    def undo(self, x=None, y=None):
        assert x is not None or y is not None
        if x is not None:
            x_res = self.x_scaler.inverse_transform(np.atleast_2d(x))
            if len(np.array(x).shape) < 2:
                x_res = x_res.squeeze()

        if y is not None:
            y_res = self.y_scaler.inverse_transform(np.atleast_2d(y))
            if len(np.array(y).shape) < 2:
                y_res = y_res.squeeze()

        if x is not None and y is not None:
            return x_res, y_res
        elif x is not None:
            return x_res
        elif y is not None:
            return y_res


class StandardTransform(Transformation):
    def __init__(self, x_bound):
        super().__init__(BoundedScaler(x_bound), StandardScaler())


class NonStandardTransform(Transformation):
    def __init__(self):
        super().__init__(MockupScaler(), StandardScaler())


class SearchSpaceTransform(Transformation):
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space
        self.y_scaler = StandardScaler()

    def fit(self, _, y):
        self.y_scaler = self.y_scaler.fit(y)

    def do(
        self, x: np.ndarray = None, y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert x is not None or y is not None
        if x is not None:
            x_res = np.atleast_2d(x).astype(float)
            for i, var in enumerate(self.search_space.data):
                if isinstance(var, Real):
                    x_res[:, i] = getattr(var, "_trans")(x_res[:, i])

            if len(np.array(x).shape) < 2:
                x_res = x_res.squeeze()

        if y is not None:
            y_res = super().do(y=y)

        if x is not None and y is not None:
            return x_res, y_res
        elif x is not None:
            return x_res
        elif y is not None:
            return y_res

    def undo(
        self, x: np.ndarray = None, y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if x is not None:
            x = self.search_space.to_linear_scale(x)
        if y is not None:
            y = super().undo(y=y)

        if x is not None and y is not None:
            return x, y
        elif x is not None:
            return x
        elif y is not None:
            return y

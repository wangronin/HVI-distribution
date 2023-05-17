from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from pymoo.factory import get_performance_indicator
from scipy.stats import norm

from .hv_improvement import HypervolumeImprovement

"""
Acquisition functions that define the objectives for surrogate multi-objective problem
"""


def find_pareto_front(Y, return_index=False):
    """
    Find pareto front (undominated part) of the input performance data.
    """
    if len(Y) == 0:
        return np.array([])
    sorted_indices = np.argsort(Y.T[0])
    pareto_indices = []
    for idx in sorted_indices:
        # check domination relationship
        a = np.all(Y <= Y[idx], axis=1)
        b = np.any(Y < Y[idx], axis=1)
        if not np.any(np.logical_and(a, b)):
            pareto_indices.append(idx)
    pareto_front = Y[pareto_indices].copy()

    if return_index:
        return pareto_front, pareto_indices
    else:
        return pareto_front


class Acquisition(ABC):
    """
    Base class of acquisition function
    """

    requires_std = (
        False  # whether requires std output from surrogate model, set False to avoid unnecessary computation
    )

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, Y, ref):
        """
        Fit the parameters of acquisition function from data
        """

        pass

    @abstractmethod
    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        """
        Evaluate the output from surrogate model using acquisition function
        Input:
            val: output from surrogate model, storing mean and std of prediction, and their derivatives
            val['F']: mean, shape (N, n_obj)
            val['dF']: gradient of mean, shape (N, n_obj, n_var)
            val['hF']: hessian of mean, shape (N, n_obj, n_var, n_var)
            val['S']: std, shape (N, n_obj)
            val['dS']: gradient of std, shape (N, n_obj, n_var)
            val['hS']: hessian of std, shape (N, n_obj, n_var, n_var)
        Output:
            F: acquisition value, shape (N, n_obj)
            dF: gradient of F, shape (N, n_obj, n_var)
            hF: hessian of F, shape (N, n_obj, n_var, n_var)
        """
        pass


class Epsilon_PoI(Acquisition):
    """at least epsilon PoI"""

    """ search for $\epsilon \%$ HVI improvement
    """
    """Naive Upper Confidence Bound"""

    requires_std = True

    def __init__(self, *args, **kwargs):
        self.n_sample = None

    def fit(self, X, Y):
        epsilon = 0.05
        self.pf = find_pareto_front(Y, return_index=False) - epsilon
        if self.rf is "dynamic":
            self.rf = np.max(self.pf, axis=0) + 1
        self.pf_shape = self.pf.shape

    def transform_pf(self, mu, sigma):
        transformed_pf = np.zeros(self.pf_shape)
        for i in range(self.pf_shape[0]):
            for j in range(self.pf_shape[1]):
                transformed_pf[i, j] = norm.cdf((self.pf[i, j] - mu[j]) / sigma[j])
        return transformed_pf

    def _evaluate_one(self, i) -> float:
        mu, sigma = self.val["F"][i, :], self.val["S"][i, :]
        transformed_pf = self.transform_pf(mu, sigma)
        hv = get_performance_indicator("hv", ref_point=[1, 1])
        F = 1 - hv.calc(transformed_pf)
        return F, None, None

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        self.val = val
        N = len(val["S"])
        F = np.atleast_2d(Parallel(n_jobs=7)(delayed(self._evaluate_one)(i) for i in range(N)))
        return -F[:, 0], F[:, 1], F[:, 2]


class PoHVI(Acquisition):
    r"""Upper Confidence Bound of the hypervolume improvement

    TODO: add the reference to our paper once it is accepted
    """
    # search for the individual x of which cdf value is nearest to the defined epsilon of HVI
    requires_std = True

    def __init__(self, tol: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.n_sample: int = 0
        self.n0: int = 0
        self.tol: float = tol

    def fit(self, X, Y) -> PoHVI:
        self.n_sample = X.shape[0]
        self.pf = find_pareto_front(Y, return_index=False)
        if self.rf is "dynamic":
            self.rf = np.max(self.pf, axis=0) + 1

    def delta_hvi(self, maxHVI) -> float:
        t = self.n_sample - self.n0
        a = 1
        b = 0.02 / a
        y = (1 / np.exp(t**a)) ** b
        return y * maxHVI

    def _evaluate_one(self, i) -> Tuple[float, float]:
        mu, sigma = self.val["F"][i, :], self.val["S"][i, :]
        hvi = HypervolumeImprovement(self.pf, self.rf, mu, sigma)
        x = self.delta_hvi(1 * 0.05)
        out = hvi.cdf(x) - 1
        out[out > 0] = 0
        return [float(out), float(x)]  # - CDF in non-dominate space, a

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        self.val = val
        N = len(val["S"])
        if N <= 50:
            F = np.atleast_2d([self._evaluate_one(i) for i in range(N)])
        else:
            F = np.atleast_2d(Parallel(n_jobs=7)(delayed(self._evaluate_one)(i) for i in range(N)))
        return F[:, 0], F[:, 1], None  # hvi.cdf(x) - 1 --> to minimize, a

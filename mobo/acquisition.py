from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from pymoo.factory import get_performance_indicator
from scipy.optimize import newton
from scipy.stats import norm

from .hv_improvement import HypervolumeImprovement
from .utils import expand, find_pareto_front, safe_divide
from scipy import optimize

"""
Acquisition functions that define the objectives for surrogate multi-objective problem
"""


class Acquisition(ABC):
    """
    Base class of acquisition function
    """

    requires_std = (
        False  # whether requires std output from surrogate model, set False to avoid unnecessary computation
    )

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, Y):
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

class NUCB(Acquisition):
    """Naive Upper Confidence Bound"""
    requires_std = True

    def __init__(self, *args, **kwargs):
        self.n_sample = None

    def fit(self, X, Y):
        self.n_sample = X.shape[0]
        self.pf = find_pareto_front(Y, return_index=False)
        if (self.rf is 'dynamic'):
            self.rf = np.max(self.pf, axis=0) + 1

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        lamda = np.sqrt(np.log(self.n_sample) / self.n_sample)

        y_mean, y_std = val["F"], val["S"]
        F = y_mean - lamda * y_std

        hv = get_performance_indicator("hv", ref_point=self.rf)
        hv_current = hv.calc(self.pf)

        FF = np.array([float(0)] * len(F))
        for i in range(0, len(F)):
            FF[i] = hv.calc(np.vstack([self.pf, F[i]])) - hv_current

        dF, hF = None, None
        dy_mean, hy_mean, dy_std, hy_std = val["dF"], val["hF"], val["dS"], val["hS"]

        if calc_gradient or calc_hessian:
            dF_y_mean = np.ones_like(y_mean)
            dF_y_std = -lamda * np.ones_like(y_std)

            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)

        if calc_gradient:
            dF = dF_y_mean * dy_mean + dF_y_std * dy_std

        if calc_hessian:
            hF_y_mean = 0
            hF_y_std = 0

            dy_mean, dy_std = expand(dy_mean), expand(dy_std)
            dy_mean_T, dy_std_T = dy_mean.transpose(0, 1, 3, 2), dy_std.transpose(0, 1, 3, 2)
            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)

            hF = (
                dF_y_mean * hy_mean
                + dF_y_std * hy_std
                + hF_y_mean * dy_mean * dy_mean_T
                + hF_y_std * dy_std * dy_std_T
            )

        return -FF, dF, hF

class UCB(Acquisition):
    r"""Upper Confidence Bound of the hypervolume improvement"""
    # search for the individual x of which hvi is largest and the cdf value is nearest to the defined CI (beta)
    requires_std = True

    def __init__(self, tol: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.n_sample: int = 0
        self.n0: int = 0
        self.tol: float = tol

    def fit(self, X, Y) -> UCB:
        self.n_sample = X.shape[0]
        self.pf = find_pareto_front(Y, return_index=False)
        if (self.rf is 'dynamic'):
            self.rf = np.max(self.pf, axis=0) + 1

    def get_beta(self) -> float:
        t = self.n_sample - self.n0 + 1
        return norm.cdf(0.55 * np.sqrt(np.log(t * 25)))

    def _evaluate_one(self, i) -> Tuple[float, float]:
        mu, sigma = self.val["F"][i, :], self.val["S"][i, :]
        hvi = HypervolumeImprovement(self.pf, self.rf, mu, sigma)
        
        # probability for the quantile
        beta = self.get_beta()
        if beta <= 1 - hvi.dominating_prob:
            return 1, 0, beta
        func = lambda x: hvi.cdf(x) - beta
        
        # sample 100 evenly-spaced points in log-10 scale to approximate the quantile
        x = 10 ** np.linspace(-1, np.log10(hvi.max_hvi), 100)
        v = np.abs(func(x))
        idx = np.argmin(v)
        out = x[idx]
        
        # if the precision of above approximation is not enough
        if not np.isclose(v[idx], 0, rtol=self.tol, atol=self.tol):
            # refine the quantile value
            out_ = newton(func, x0=out, fprime=hvi.pdf, tol=self.tol, maxiter=20, disp=False)
            if out > 0:
                out = out_
        return [float(v[idx]), float(out), float(beta)]  # abs(CDF-CI), HVI, beta 

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        self.val = val
        N = len(val["S"])
        F = np.atleast_2d(Parallel(n_jobs=7)(delayed(self._evaluate_one)(i) for i in range(N)))
        # To maximize the HVI value under the same confidence bound
        return -F[:, 1], F[:, 0], F[:, 2]  # HVI, abs(CDF-CI), none

class Epsilon_PoI(Acquisition):
    """at least epsilon PoI """
    """ search for $\epsilon \%$ HVI improvement
    """
    """Naive Upper Confidence Bound"""

    requires_std = True

    def __init__(self, *args, **kwargs):
        self.n_sample = None

    def fit(self, X, Y):
        epsilon = 0.05
        self.pf = find_pareto_front(Y, return_index=False) - epsilon
        if (self.rf is 'dynamic'):
            self.rf = np.max(self.pf, axis=0) + 1
        self.pf_shape = self.pf.shape

    def transform_pf(self,mu, sigma):
        transformed_pf = np.zeros(self.pf_shape)
        for i in range(self.pf_shape[0]):
            for j in range(self.pf_shape[1]): 
                transformed_pf[i,j] = norm.cdf((self.pf[i,j]-mu[j])/sigma[j])
        return transformed_pf

    def _evaluate_one(self, i) -> float:
        mu, sigma = self.val["F"][i, :], self.val["S"][i, :]
        transformed_pf = self.transform_pf(mu, sigma)
        hv = get_performance_indicator("hv", ref_point=self.rf)
        F = 1 - hv.calc(transformed_pf)

        return F, None, None

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        dF, hF = None, None
        self.val = val
        N = len(val["S"])

        # F = np.array([[float(0)] * 3] * N)
        # for i in range(N):
        #     F[i] = self._evaluate_one(i)
        F = np.atleast_2d(Parallel(n_jobs=7)(delayed(self._evaluate_one)(i) for i in range(N)))

        return -F[:, 0], F[:, 1], F[:, 2] 

class Epsilon_PoI_Cut(Acquisition):
    
    """at least epsilon PoI """
    """ search for $\epsilon HVI$ improvement
    """
    """Naive Upper Confidence Bound"""

    requires_std = True

    def __init__(self, tol: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.n_sample: int = 0
        self.n0: int = 0
        self.tol: float = tol


    def fit(self, X, Y):

        self.pf = find_pareto_front(Y, return_index=False) 
        if (self.rf is 'dynamic'):
            self.rf = np.max(self.pf, axis=0) + 1
        hv = get_performance_indicator("hv", ref_point=np.ravel(self.rf))

        func = lambda x: hv.calc(self.pf - x) - hv.calc(self.pf) - self.delta_hvi(0.05)   
        solution = optimize.root(func, 0.05, method='lm')
        
        self.pf = self.pf - solution.x
        
        self.pf_shape = self.pf.shape

    def delta_hvi(self, maxHVI) -> float:
        t = self.n_sample - self.n0
        a = 1
        b = 0.02 / a
        y = (1 / np.exp(t ** a)) ** b
        return y * maxHVI

    def transform_data(self, data, mu, sigma):
        transformed_data = np.zeros(data.shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]): 
                transformed_data[i,j] = norm.cdf((data[i,j]-mu[j])/sigma[j])
        return transformed_data


    def _evaluate_one(self, i) -> float:
        mu, sigma = self.val["F"][i, :], self.val["S"][i, :]
        transformed_pf = self.transform_data(self.pf, mu, sigma)

        # cut the space by using reference point: 
        self.tf_rf = np.ravel(self.transform_data(np.atleast_2d(self.rf), mu, sigma))

        hv = get_performance_indicator("hv", ref_point=self.tf_rf)
        F = (np.prod(self.tf_rf) - hv.calc(transformed_pf)) / np.prod(self.tf_rf)

        return F, None, None

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        dF, hF = None, None
        self.val = val
        N = len(val["S"])
        
        F = np.array([[float(0)] * 3] * N)

        # for i in range(N):
        #     F[i] = self._evaluate_one(i)

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
        # self.rf = np.max(Y, axis=0) + 1
        if (self.rf is 'dynamic'):
            self.rf = np.max(self.pf, axis=0) + 1 

    def delta_hvi(self, maxHVI) -> float:
        # n = self.n_sample - self.n0+1
        # # return maxHVI * np.true_divide(1,n)**0.2
        # return maxHVI - n * maxHVI / 171
        t = self.n_sample - self.n0
        a = 1
        b = 0.02 / a
        y = (1 / np.exp(t ** a)) ** b
        return y * maxHVI

    def _evaluate_one(self, i) -> Tuple[float, float]:
        mu, sigma = self.val["F"][i, :], self.val["S"][i, :]
        hvi = HypervolumeImprovement(self.pf, self.rf, mu, sigma)
        x = self.delta_hvi(1 * 0.05)
        out = hvi.cdf(x) - 1

        return [float(out), float(x)]  # - CDF in non-dominate space, a

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        self.val = val
        N = len(val["S"])
        F = np.atleast_2d(Parallel(n_jobs=7)(delayed(self._evaluate_one)(i) for i in range(N)))

        return F[:, 0], F[:, 1], None  # hvi.cdf(x) - 1 --> to minimize, a

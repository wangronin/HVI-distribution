from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .hv_improvement import HypervolumeImprovement
from .utils import expand, find_pareto_front, safe_divide
# import matplotlib.pyplot as plt
# from pynverse import inversefunc
from pymoo.factory import get_performance_indicator
from joblib import Parallel, delayed
# import timeit

"""
Acquisition functions that define the objectives for surrogate multi-objective problem
"""


class Acquisition(ABC):
    """
    Base class of acquisition function
    """

    requires_std = False  # whether requires std output from surrogate model, set False to avoid unnecessary computation

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


class IdentityFunc(Acquisition):
    """
    Identity function
    """

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        F, dF, hF = val["F"], val["dF"], val["hF"]
        return F, dF, hF


class PI(Acquisition):
    """
    Probability of Improvement
    """

    requires_std = True

    def __init__(self, *args, **kwargs):
        self.y_min = None

    def fit(self, X, Y):
        self.y_min = np.min(Y, axis=0)

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        y_mean, y_std = val["F"], val["S"]
        z = safe_divide(self.y_min - y_mean, y_std)
        cdf_z = norm.cdf(z)
        F = -cdf_z

        dF, hF = None, None
        dy_mean, hy_mean, dy_std, hy_std = val["dF"], val["hF"], val["dS"], val["hS"]

        if calc_gradient or calc_hessian:
            dz_y_mean = -safe_divide(1, y_std)
            dz_y_std = -safe_divide(self.y_min - y_mean, y_std ** 2)

            pdf_z = norm.pdf(z)
            dF_y_mean = -pdf_z * dz_y_mean
            dF_y_std = -pdf_z * dz_y_std

            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)

        if calc_gradient:
            dF = dF_y_mean * dy_mean + dF_y_std * dy_std

        if calc_hessian:
            dpdf_z_z = -z * pdf_z
            dpdf_z_y_mean = dpdf_z_z * dz_y_mean
            dpdf_z_y_std = dpdf_z_z * dz_y_std
            hz_y_std = safe_divide(self.y_min - y_mean, y_std ** 3)

            hF_y_mean = -dpdf_z_y_mean * dz_y_mean
            hF_y_std = -dpdf_z_y_std * dz_y_std - pdf_z * hz_y_std

            dy_mean, dy_std = expand(dy_mean), expand(dy_std)
            dy_mean_T, dy_std_T = dy_mean.transpose(0, 1, 3, 2), dy_std.transpose(0, 1, 3, 2)
            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)
            hF_y_mean, hF_y_std = expand(hF_y_mean, (-1, -2)), expand(hF_y_std, (-1, -2))

            hF = (
                dF_y_mean * hy_mean
                + dF_y_std * hy_std
                + hF_y_mean * dy_mean * dy_mean_T
                + hF_y_std * dy_std * dy_std_T
            )

        return F, dF, hF


class EI(Acquisition):
    """
    Expected Improvement
    """

    requires_std = True

    def __init__(self, *args, **kwargs):
        self.y_min = None

    def fit(self, X, Y):
        self.y_min = np.min(Y, axis=0)

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        y_mean, y_std = val["F"], val["S"]
        z = safe_divide(self.y_min - y_mean, y_std)
        pdf_z = norm.pdf(z)
        cdf_z = norm.cdf(z)
        F = -(self.y_min - y_mean) * cdf_z - y_std * pdf_z

        dF, hF = None, None
        dy_mean, hy_mean, dy_std, hy_std = val["dF"], val["hF"], val["dS"], val["hS"]

        if calc_gradient or calc_hessian:
            dz_y_mean = -safe_divide(1, y_std)
            dz_y_std = -safe_divide(self.y_min - y_mean, y_std ** 2)
            dpdf_z_z = -z * pdf_z

            dF_y_mean = (
                cdf_z - (self.y_min - y_mean) * pdf_z * dz_y_mean - y_std * dpdf_z_z * dz_y_mean
            )
            dF_y_std = (
                (self.y_min - y_mean) * pdf_z * dz_y_std + pdf_z + y_std * dpdf_z_z * dz_y_std
            )

            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)

        if calc_gradient:
            dF = dF_y_mean * dy_mean + dF_y_std * dy_std

        if calc_hessian:
            dpdf_z_y_mean = dpdf_z_z * dz_y_mean
            dpdf_z_y_std = dpdf_z_z * dz_y_std
            ddpdf_z_z_y_mean = -z * dpdf_z_y_mean - dz_y_mean * pdf_z
            ddpdf_z_z_y_std = -z * dpdf_z_y_std - dz_y_std * pdf_z
            ddz_y_std_y_std = safe_divide(self.y_min - y_mean, y_std ** 3)

            hF_y_mean = (
                -pdf_z * dz_y_mean
                - dz_y_mean * pdf_z
                + (self.y_min - y_mean) * dpdf_z_z * dz_y_mean ** 2
                + y_std * dz_y_mean * ddpdf_z_z_y_mean
            )
            hF_y_std = (
                (self.y_min - y_mean) * (dz_y_std * dpdf_z_y_std + pdf_z * ddz_y_std_y_std)
                + dpdf_z_y_std
                + dpdf_z_z * dz_y_std
                + y_std * dz_y_std * ddpdf_z_z_y_std
                + y_std * dpdf_z_z * ddz_y_std_y_std
            )

            dy_mean, dy_std = expand(dy_mean), expand(dy_std)
            dy_mean_T, dy_std_T = dy_mean.transpose(0, 1, 3, 2), dy_std.transpose(0, 1, 3, 2)
            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)
            hF_y_mean, hF_y_std = expand(hF_y_mean, (-1, -2)), expand(hF_y_std, (-1, -2))

            hF = (
                dF_y_mean * hy_mean
                + dF_y_std * hy_std
                + hF_y_mean * dy_mean * dy_mean_T
                + hF_y_std * dy_std * dy_std_T
            )

        return F, dF, hF


class UCB(Acquisition):
    """Naive Upper Confidence Bound"""

    requires_std = True

    def __init__(self, *args, **kwargs):
        self.n_sample = None

    def fit(self, X, Y):
        self.n_sample = X.shape[0]
        self.pf = find_pareto_front(Y, return_index=False)
        self.rf = np.max(Y, axis=0) + 1

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        lamda = np.sqrt(np.log(self.n_sample) / self.n_sample)

        y_mean, y_std = val["F"], val["S"]
        F = y_mean - lamda * y_std


        hv = get_performance_indicator('hv', ref_point=self.rf)
        hv_current = hv.calc(self.pf)


        FF = np.array([float(0)] * len(F))
        for i in range(0,len(F)):
            FF[i] = hv_current - hv.calc(np.vstack([self.pf, F[i]]))


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

        return FF, dF, hF


class HVI_UCB(Acquisition):
    r"""Upper Confidence Bound of the hypervolume improvement

    TODO: add the reference to our paper once it is accepted
    """

    requires_std = True

    def __init__(self, quantile: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.n_sample = None
        self.quantile = quantile

    def fit(self, X, Y):
        self.n_sample = X.shape[0]
        self.pf = find_pareto_front(Y, return_index=False)
        self.rf = np.max(Y, axis=0) + 1
        # self.rf = [15, 15]

    def optimize_function(self, i):
        pf = self.pf
        mu, sigma = self.val["F"], self.val["S"]
        rf = self.rf

        t = self.n_sample
        lamda = (0.7 - np.sqrt(np.log(t) / t))

        hvi = HypervolumeImprovement(pf,  rf,  mu[i,:],  sigma[i,:])

        func = lambda x: abs(lamda - hvi.cdf(x))
        sol = minimize(func, 1e-10, method="CG", options={"maxiter": 50})

        p = sol.fun
        a = sol.x

        return float(p), float(a)

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        dF, hF = None, None
        self.val = val
        N = len(val["S"])
        F = np.array([float(0)] * N)
        dF = np.array([float(0)] * N)



        res = Parallel(n_jobs=7)(
            delayed(self.optimize_function)(i) for i in range(N))

        F = np.array([res[i][0] for i in range(N)])
        dF = np.array([res[i][1] for i in range(N)])

        return F, dF, hF

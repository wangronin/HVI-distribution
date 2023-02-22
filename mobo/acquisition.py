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

            dF_y_mean = cdf_z - (self.y_min - y_mean) * pdf_z * dz_y_mean - y_std * dpdf_z_z * dz_y_mean
            dF_y_std = (self.y_min - y_mean) * pdf_z * dz_y_std + pdf_z + y_std * dpdf_z_z * dz_y_std

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

        hv = get_performance_indicator("hv", ref_point=self.rf)
        hv_current = hv.calc(self.pf)

        FF = np.array([float(0)] * len(F))
        for i in range(0, len(F)):
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
    r"""Upper Confidence Bound of the hypervolume improvement"""
    # search for the individual x of which cdf value is nearest to the defined CI (beta)
    requires_std = True

    def __init__(self, tol: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.n_sample: int = 0
        self.n0: int = 0
        self.tol: float = tol

    def fit(self, X, Y) -> HVI_UCB_M2:
        self.n_sample = X.shape[0]
        self.pf = find_pareto_front(Y, return_index=False)
        self.rf = np.max(Y, axis=0) + 1
        return self

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
        return [float(v[idx]), float(out), float(beta)]  # abs(CDF-CI), a

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        self.val = val
        N = len(val["S"])
        F = np.atleast_2d(Parallel(n_jobs=7)(delayed(self._evaluate_one)(i) for i in range(N)))
        return -F[:, 1], F[:, 0], F[:, 2]  # a, abs(CDF-CI), none


class HVI_UCB_M1(Acquisition):
    r"""Upper Confidence Bound of the hypervolume improvement

    TODO: add the reference to our paper once it is accepted
    """
    # search for the individual x of which cdf value is nearest to the defined CI (beta)
    requires_std = True

    def __init__(self, tol: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.n_sample: int = 0
        self.n0: int = 0
        self.tol: float = tol

    def fit(self, X, Y) -> HVI_UCB_M1:
        self.n_sample = X.shape[0]
        self.pf = find_pareto_front(Y, return_index=False)
        self.rf = np.max(Y, axis=0) + 1
        return self

    def get_beta(self) -> float:
        t = self.n_sample - self.n0 + 1
        return norm.cdf(0.55 * np.sqrt(np.log(t * 25)))
        # return 1 - (1 - min_prob) / n ** 1.5
        # c = (1 - min_prob) / np.sqrt(np.log(2) / 2)
        # return 1 - c * np.sqrt(np.log(t + 1) / (t + 1))

        # if n < 2:
        #     return 0.1 + 0.9 * n / 170
        # else:
        #     return 1 - (1 - min_prob) / (n+1) ** 1.5

        # c = (1 - min_prob) / np.sqrt(np.log(2) / 2)
        # return 1 - c * np.sqrt(np.log(n + 1) / (n + 1))

        # return 0.01 + 0.99 * n / 170

    def _evaluate_one(self, i) -> Tuple[float, float]:

        mu, sigma = self.val["F"][i, :], self.val["S"][i, :]
        hvi = HypervolumeImprovement(self.pf, self.rf, mu, sigma)
        # probability for the quantile
        beta = self.get_beta()
        if beta <= 1 - hvi.dominating_prob:
            return 1, -hvi.dominating_prob, beta
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
        return [float(v[idx]), float(out), float(beta)]  # abs(CDF-CI), a

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        self.val = val
        N = len(val["S"])

        # # test
        # F = np.array([[float(0)] * 3] * N)
        # for i in range(N):
        #     F[i] = self._evaluate_one(i)
        
        F = np.atleast_2d(Parallel(n_jobs=7)(delayed(self._evaluate_one)(i) for i in range(N)))

        return F[:, 0], F[:, 1], F[:, 2]  # abs(CDF-CI), a, beta


class HVI_UCB_M2(Acquisition):
    r"""Upper Confidence Bound of the hypervolume improvement

    TODO: add the reference to our paper once it is accepted
    """
    # search for the individual x of which cdf value is nearest to the defined CI (beta)
    requires_std = True

    def __init__(self, tol: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.n_sample: int = 0
        self.n0: int = 0
        self.tol: float = tol

    def fit(self, X, Y) -> HVI_UCB_M2:
        self.n_sample = X.shape[0]
        self.pf = find_pareto_front(Y, return_index=False)
        self.rf = np.max(Y, axis=0) + 1
        return self

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
        return [float(v[idx]), float(out), float(beta)]  # abs(CDF-CI), a

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        self.val = val
        N = len(val["S"])
        F = np.atleast_2d(Parallel(n_jobs=7)(delayed(self._evaluate_one)(i) for i in range(N)))

        return -F[:, 1], F[:, 0], F[:, 2]  # a, abs(CDF-CI), none


class HVI_UCB_M3(Acquisition):
    r"""Upper Confidence Bound of the hypervolume improvement

    TODO: add the reference to our paper once it is accepted
    """
    # search for the individual x of which cdf value is nearest to the defined CI (beta)
    requires_std = True

    def __init__(self, tol: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.n_sample: int = 0
        self.n0: int = 0
        self.tol: float = tol

    def fit(self, X, Y) -> HVI_UCB_M3:
        self.n_sample = X.shape[0]
        self.pf = find_pareto_front(Y, return_index=False)
        # self.rf = np.max(Y, axis=0) + 1
        self.rf = [15, 15]
        return self

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

        # x = self.delta_hvi(hvi.max_hvi * 0.382)
        # x = self.delta_hvi(hvi.max_hvi * hvi.dominating_prob)
        # x = self.delta_hvi(hvi.max_hvi * 0.618)
        x = self.delta_hvi(1 * 0.05)
        out = -(1 - hvi.cdf(x))

        return [float(out), float(x)]  # - CDF in non-dominate space, a

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        self.val = val
        N = len(val["S"])
        F = np.atleast_2d(Parallel(n_jobs=7)(delayed(self._evaluate_one)(i) for i in range(N)))

        return F[:, 0], F[:, 1], None  # abs(CDF-CI), a


class HVI_UCB_M3_EPSILON(Acquisition):
    r"""Upper Confidence Bound of the hypervolume improvement

    TODO: add the reference to our paper once it is accepted
    """
    # search for the individual x of which cdf value is nearest to the defined CI (beta)
    requires_std = True

    def __init__(self, tol: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.n_sample: int = 0
        self.n0: int = 0
        self.tol: float = tol

    def fit(self, X, Y) -> HVI_UCB_M3_EPSILON:
        self.n_sample = X.shape[0]
        epsilon = 0.05
        self.pf_org = find_pareto_front(Y, return_index=False)
        self.pf = self.pf_org - epsilon
        self.rf = [15, 15]
        self.pf_shape = self.pf.shape
        return self

    def delta_hvi(self) -> float:
        # n = self.n_sample - self.n0+1
        # # return maxHVI * np.true_divide(1,n)**0.2
        # return maxHVI - n * maxHVI / 171
        num_pf = self.pf.shape[0]
        hvi_pf = np.array([0] * num_pf)


        hv = get_performance_indicator("hv", ref_point=self.rf)
        hv_current = hv.calc(self.pf_org)

        hvi_pf = np.array([float(0)] * num_pf)
        for i in range(0, num_pf):
            hvi_pf[i] = hv.calc(np.vstack([self.pf_org, self.pf[i]])) - hv_current

        rst = np.min(hvi_pf)
        return rst
        # return rst if rst < maxHVI else maxHVI

    def _evaluate_one(self, i) -> Tuple[float, float]:
        mu, sigma = self.val["F"][i, :], self.val["S"][i, :]
        hvi = HypervolumeImprovement(self.pf, self.rf, mu, sigma)

        # x = self.delta_hvi(hvi.max_hvi * 0.382)
        # x = self.delta_hvi(hvi.max_hvi * hvi.dominating_prob)
        # x = self.delta_hvi(hvi.max_hvi * 0.618)
        x = self.delta_hvi()
        out = -(1 - hvi.cdf(x))
        return [float(out), float(x)]  # - CDF in non-dominate space, a

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        self.val = val
        N = len(val["S"])

        # F = np.array([[float(0)] * 3] * N)
        # for i in range(N):
        #     F[i] = self._evaluate_one(i)
        F = np.atleast_2d(Parallel(n_jobs=7)(delayed(self._evaluate_one)(i) for i in range(N)))

        return F[:, 0], F[:, 1], None  # abs(CDF-CI), a



class Epsilon_PoI(Acquisition):
    from scipy.stats import norm
    """at least epsilon PoI """
    # search for the solution with at least Epsilon PoI
    # currently only works for bio-objective optimization
    """Naive Upper Confidence Bound"""

    requires_std = True

    def __init__(self, *args, **kwargs):
        self.n_sample = None

    def fit(self, X, Y):
        epsilon = 0.05
        self.pf = find_pareto_front(Y, return_index=False) - epsilon
        self.rf = np.array([1,1])
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
    from scipy.stats import norm
    """at least epsilon PoI """
    # search for the solution with at least Epsilon PoI
    # currently only works for bio-objective optimization
    """Naive Upper Confidence Bound"""

    requires_std = True

    def __init__(self, tol: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.n_sample: int = 0
        self.n0: int = 0
        self.tol: float = tol


    def fit(self, X, Y):
        self.pf = find_pareto_front(Y, return_index=False) 
        self.rf = np.array([[15,15]])
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
        self.tf_rf = np.ravel(self.transform_data(self.rf, mu, sigma))

        hv = get_performance_indicator("hv", ref_point=self.tf_rf)
        F = (np.prod(self.tf_rf) - hv.calc(transformed_pf)) / np.prod(self.tf_rf)

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


class HVI_UCB_M4(Acquisition):
    r"""Upper Confidence Bound of the hypervolume improvement of using dynamic reference point

    TODO: add the reference to our paper once it is accepted
    """
    # search for the individual x of which cdf value is nearest to the defined CI (beta)
    requires_std = True
    def __init__(self, tol: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.n_sample: int = 0
        self.n0: int = 0
        self.tol: float = tol


class HVI_UCB_M3_EPSILON_DR(Acquisition):
    r"""Upper Confidence Bound of the hypervolume improvement of using dynamic reference point

    TODO: add the reference to our paper once it is accepted
    """
    # search for the individual x of which cdf value is nearest to the defined CI (beta)
    requires_std = True

    def __init__(self, tol: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.n_sample: int = 0
        self.n0: int = 30               #TODO: import from Acuiqistion function
        self.window_size: int = 10      #TODO: import from Acuiqistion function
        self.tol: float = tol

    def fit(self, X, Y) -> HVI_UCB_M3_EPSILON_DR:
        self.n_sample = X.shape[0]
        epsilon = 0.05          #TODO: import from Acquisition function 
        self.rf = [15, 15]      #TODO: import from Acuiqistion function
        if self.n_sample - self.window_size < self.n0: 
            self.extreme_point_impr_prob = 1
        else: 
            old_pf_extreme = Y[0:self.n_sample - self.window_size].min(axis=0)
            y_in_window = Y[self.n_sample - self.window_size: self.n_sample]
            rst = (y_in_window < old_pf_extreme).any(axis=1)
            self.extreme_point_impr_prob = np.sum(rst) / rst.shape[0]
        



        self.pf_org = find_pareto_front(Y, return_index=False)
        self.pf = self.pf_org - epsilon
        
        self.pf_shape = self.pf.shape
        return self

    def delta_hvi(self) -> float:
        # n = self.n_sample - self.n0+1
        # # return maxHVI * np.true_divide(1,n)**0.2
        # return maxHVI - n * maxHVI / 171
        num_pf = self.pf.shape[0]
        hvi_pf = np.array([0] * num_pf)

        hv = get_performance_indicator("hv", ref_point=self.rf)
        hv_current = hv.calc(self.pf_org)

        hvi_pf = np.array([float(0)] * num_pf)
        for i in range(0, num_pf):
            hvi_pf[i] = hv.calc(np.vstack([self.pf_org, self.pf[i]])) - hv_current

        rst = np.min(hvi_pf)
        return rst
        # return rst if rst < maxHVI else maxHVI

    def _evaluate_one(self, i) -> Tuple[float, float]:
        mu, sigma = self.val["F"][i, :], self.val["S"][i, :]
        hvi = HypervolumeImprovement(self.pf, self.rf, mu, sigma, self.extreme_point_impr_prob)

        x = self.delta_hvi()
        out = -(1 - hvi.cdf(x))
        return [float(out), float(x)]  # - CDF in non-dominate space, a

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        self.val = val
        N = len(val["S"])

        # F = np.array([[float(0)] * 2] * N)
        # for i in range(N):
        #     F[i] = self._evaluate_one(i)
        F = np.atleast_2d(Parallel(n_jobs=7)(delayed(self._evaluate_one)(i) for i in range(N)))

        return F[:, 0], F[:, 1], None  # abs(CDF-CI), a

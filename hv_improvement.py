from __future__ import annotations

import warnings
from typing import Callable, List, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from numba import cfunc, jit
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable
from scipy.integrate import quad
from scipy.special import binom, factorial
from scipy.stats import norm

from hypervolume import hypervolume as hv

warnings.simplefilter("error")

__authors__ = ["Kaifeng Yang", "Hao Wang"]


def jit_integrand(integrand_function):
    jitted_function = jit(integrand_function, nopython=True)

    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(_, xx):
        return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5])

    return LowLevelCallable(wrapped.ctypes)


def D(L, U, loc, scale):
    return norm.cdf(U, loc, scale) - norm.cdf(L, loc, scale)


@jit_integrand
def integrand_eq4(*args):
    x, mu0, mu1, sigma0, sigma1, p = args
    return np.exp(-0.5 * (((x - mu0) / sigma0) ** 2 + ((p / x - mu1) / sigma1) ** 2)) / x


def integrand_eq7(x, m, n, sigma, p):
    return x ** (2 * m - n - 1) * np.exp(-0.5 * ((x / sigma[0]) ** 2 + (p / x / sigma[1]) ** 2))


def pdf_product_of_truncated_gaussian(
    p: float,
    mean: List[float],
    sigma: List[float],
    lower: List[float],
    upper: List[float],
    normalizer: float,
    taylor_expansion: bool = False,
    taylor_order: int = 5,
    fac: List[float] = None,
    bc: List[List[float]] = None,
) -> float:
    (L1, L2), (U1, U2) = lower, upper
    if L1 * U2 > U1 * L2:  # swap y_1' and y_2'
        (L2, L1), (U2, U1) = lower, upper
        mean = mean[1], mean[0]
        sigma = sigma[1], sigma[0]

    if L1 * L2 <= p < L1 * U2:
        alpha = L1
        beta = p / L2
    elif L1 * U2 <= p < U1 * L2:
        alpha = p / U2
        beta = p / L2
    elif U1 * L2 <= p <= U1 * U2:
        alpha = p / U2
        beta = U1
    else:
        return 0

    if alpha == beta:
        return 0

    if not taylor_expansion:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                out = quad(
                    integrand_eq4, alpha, beta, args=(mean[0], mean[1], sigma[0], sigma[1], p)
                )[0]
            except Warning:
                out = 0
    else:
        eta = np.log(sigma[1]) - np.log(sigma[0] * p)
        # range of the integration
        L, U = 2 * np.log(alpha) + eta, 2 * np.log(beta) + eta

        C1 = p / np.prod(sigma)
        C2 = np.array([(2 * m - n) / 2 for n in range(taylor_order) for m in range(n + 1)])
        C = np.exp(-0.5 * (mean[0] ** 2 / sigma[0] ** 2 + mean[1] ** 2 / sigma[1] ** 2))
        mn = np.array([(m, n) for n in range(taylor_order) for m in range(n + 1)])
        term1 = np.array(
            [
                (
                    p ** (n - m)
                    / fac[n]
                    * bc[n][m]
                    * (mean[0] / sigma[0] ** 2) ** m
                    * (mean[1] / sigma[1] ** 2) ** (n - m)
                )
                for m, n in mn
            ]
        )
        term2 = np.array([0.5 * (p * sigma[0] / sigma[1]) ** ((2 * m - n) / 2) for m, n in mn])
        bs = (U - L) / 5
        breaks = [(L + bs * i, L + bs * (i + 1)) for i in range(5)]
        out = np.zeros(len(C2))
        for l, u in breaks:
            # expand the integrand at the mid point
            x = (l + u) / 2
            f = np.exp(-C1 * np.cosh(x) + C2 * x)  # the integrand
            a = f * (C2 - C1 * np.sinh(x))  # first-order derivative
            b = f * ((C2 - C1 * np.sinh(x)) ** 2 - C1 * np.cosh(x))  # second-order derivative
            # c = (
            #     f * (C1 * np.sinh(x) - 2 * C1 * np.cosh(x) * (C2 - C1 * np.cosh(x)))
            #     + ((C2 - C1 * np.sinh(x)) ** 2 - C1 * np.cosh(x)) * a
            # )  # third-order derivative

            # second-order Taylor approximation of the integral
            out += (
                (u - l) * f
                + ((u - x) ** 2 - (l - x) ** 2) * a / 2
                + ((u - x) ** 3 - (l - x) ** 3) * b / 6
                # + ((u - x) ** 4 - (l - x) ** 4) * c / 24
            )
        out = C * (term1 * term2 * out).sum()

    return out / normalizer


class HypervolumeImprovement:
    r"""Class to computer the Hypervolume Improvement and the distribution thereof"""

    def __init__(
        self,
        pareto_front: np.ndarray,
        r: Union[List, np.ndarray],
        mu: List[float],
        sigma: List[float],
    ):
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        assert len(self.mu) == len(self.sigma)
        # 6-sigma corresponds to ~1.973175e-09 significance
        self.neg_inf: float = self.mu - 6.0 * self.sigma
        self.r = r
        self.pareto_front = pareto_front
        self.set_cells(self.pareto_front)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        self.dim = len(r)
        self._r = np.asarray(r)

    @property
    def pareto_front(self):
        return self._pareto_front

    @pareto_front.setter
    def pareto_front(self, pareto_front):
        pareto_front = np.vstack(
            (pareto_front, [self.neg_inf[0], self.r[1]], [self.r[0], self.neg_inf[1]])
        )
        idx = pareto_front[:, 0].argsort()
        self._pareto_front = pareto_front[idx]
        self.N = pareto_front.shape[0] - 1

    def set_cells(self, pareto_front: np.ndarray):
        self.cells_lb = np.full((self.N, self.N, self.dim), np.nan)
        self.cells_ub = np.full((self.N, self.N, self.dim), np.nan)
        self.transformed_lb = np.full((self.N, self.N, self.dim), np.nan)
        self.transformed_ub = np.full((self.N, self.N, self.dim), np.nan)
        self.normalizer = np.full((self.N, self.N), np.nan)
        for i in range(self.N):
            for j in range(self.N - i):
                self.cells_lb[i, j] = [pareto_front[i, 0], pareto_front[self.N - j, 1]]
                self.cells_ub[i, j] = [pareto_front[i + 1, 0], pareto_front[self.N - 1 - j, 1]]
                self.transformed_lb[i, j] = [
                    pareto_front[self.N - j, 0] - pareto_front[i + 1, 0],
                    pareto_front[i, 1] - pareto_front[self.N - 1 - j, 1],
                ]
                self.transformed_ub[i, j] = [
                    pareto_front[self.N - j, 0] - pareto_front[i, 0],
                    pareto_front[i, 1] - pareto_front[self.N - j, 1],
                ]
                mu_prime = self.mu_prime(i, j)
                self.normalizer[i, j] = (
                    2
                    * np.pi
                    * np.prod(self.sigma)
                    * np.prod(
                        [
                            D(
                                self.transformed_lb[i, j, k],
                                self.transformed_ub[i, j, k],
                                mu_prime[k],
                                self.sigma[k],
                            )
                            for k in range(self.dim)
                        ]
                    )
                )

    def mu_prime(self, i: int, j: int) -> List[float]:
        return [
            self.pareto_front[self.N - j, 0] - self.mu[0],
            self.pareto_front[i, 1] - self.mu[1],
        ]

    def gamma(self, i: int, j: int) -> float:
        return self.improvement(self.cells_ub[i, j]) - np.prod(self.transformed_lb[i, j])

    def prob_in_cell(self, i: int, j: int) -> float:
        """The probability of a Gaussian random point falling in a cell, in which independent
        marginals are assumed.

        Parameters
        ----------
        i : int
            the cell's row index
        j : int
            the cell's column index

        Returns
        -------
        float
            the probability
        """
        return np.prod(
            [
                D(self.cells_lb[i, j][k], self.cells_ub[i, j][k], self.mu[k], self.sigma[k])
                for k in range(self.dim)
            ]
        )

    def improvement(
        self, new: List[float], pareto_front: np.ndarray = None, r: np.ndarray = None
    ) -> float:
        pareto_front = pareto_front if pareto_front else self.pareto_front
        r = r if r else self.r
        return hv(np.vstack([pareto_front, new]).tolist(), r) - hv(pareto_front.tolist(), r)

    def _check_input(
        self, v: Union[float, List[float], np.ndarray], taylor_expansion: bool, taylor_order: int
    ) -> np.ndarray:
        if isinstance(v, (int, float)):
            v = [v]
        v = np.array(v)
        if taylor_expansion:
            self.fac = [factorial(i) for i in range(taylor_order)]
            self.bc = [[binom(i, j) for j in range(i + 1)] for i in range(taylor_order)]
        else:
            self.fac, self.bc = None, None
        return v

    def __internal_loop_over_cells(
        self,
        v: Union[float, List[float], np.ndarray],
        func: Callable,
        parallel: bool = False,
        n_jobs: int = 6,
        **kwargs,
    ) -> Tuple[float, float]:
        if isinstance(v, (int, float)):
            v = [v]
        v = np.array(v)
        # loop over all the cells
        terms = np.zeros((self.N, self.N, len(v)))
        ij = [(i, j) for i in range(self.N) for j in range(self.N - i)]
        prob = 0  # probability in the dominating region w.r.t. the attainment boundary
        res = (
            Parallel(n_jobs=n_jobs)(delayed(func)(v, i, j, **kwargs) for i, j in ij)
            if parallel
            else [func(v, i, j, **kwargs) for i, j in ij]
        )
        for k, (i, j) in enumerate(ij):
            prob_ij = self.prob_in_cell(i, j)
            terms[i, j, :] = res[k] * prob_ij
            prob += prob_ij
        return terms.sum(axis=(0, 1)), prob

    def pdf_conditional(
        self, v: np.ndarray, i: int, j: int, taylor_expansion: bool = False, taylor_order: int = 25
    ) -> np.ndarray:
        """Conditional PDF of hypervolume when restricting the objective point in the cell (i, j)

        Parameters
        ----------
        v : np.ndarray
            the hypervolume values
        i : int
            cell's row index
        j : int
            cell's column index
        taylor_expansion : bool, optional
            whether using Taylor expansion to computate the conditional density, by default False
        taylor_order : int, optional
            the order of the Taylor expansion, by default 25

        Returns
        -------
        np.ndarray
            the conditional probability density at volume `v`
        """
        par = (
            self.mu_prime(i, j),
            self.sigma,
            self.transformed_lb[i, j],
            self.transformed_ub[i, j],
            self.normalizer[i, j],
            taylor_expansion,
            taylor_order,
            self.fac,
            self.bc,
        )
        return np.array([pdf_product_of_truncated_gaussian(p, *par) for p in v - self.gamma(i, j)])

    def pdf(
        self,
        v: Union[float, List[float], np.ndarray],
        taylor_expansion: bool = False,
        taylor_order: int = 6,
    ) -> np.ndarray:
        """PDF of the hypervolume

        Parameters
        ----------
        v : Union[float, List[float], np.ndarray]
            the hypervolume values
        taylor_expansion : bool, optional
            whether using Taylor expansion to computate the conditional density, by default False
        taylor_order : int, optional
            the order of the Taylor expansion, by default 25

        Returns
        -------
        np.ndarray
            the probability density at volume `v`
        """
        v = self._check_input(v, taylor_expansion, taylor_order)
        idx = v == 0
        res, prob = self.__internal_loop_over_cells(
            v,
            self.pdf_conditional,
            taylor_expansion=taylor_expansion,
            taylor_order=taylor_order,
        )
        # NOTE: the density at zero volume is a Dirac delta
        if np.any(idx):
            res = res.astype(object)
            _ = res[idx]
            res[idx] = f"{1 - prob} * delta" if _ == 0 else f"{1 - prob} * delta + {_}"
        return res

    def cdf_conditional(
        self,
        v: np.ndarray,
        i: int,
        j: int,
        taylor_expansion: bool = False,
        taylor_order: int = 25,
    ) -> np.ndarray:
        """Conditional CDF of hypervolume when restricting the objective point in the cell (i, j)

        Parameters
        ----------
        v : np.ndarray
            the hypervolume values
        i : int
            cell's row index
        j : int
            cell's column index
        taylor_expansion : bool, optional
            whether using Taylor expansion to computate the conditional density, by default False
        taylor_order : int, optional
            the order of the Taylor expansion, by default 25

        Returns
        -------
        np.ndarray
            the cumulative probability at volume `v`
        """
        L, U = np.prod(self.transformed_lb[i, j]), np.prod(self.transformed_ub[i, j])
        args = (
            self.mu_prime(i, j),
            self.sigma,
            self.transformed_lb[i, j],
            self.transformed_ub[i, j],
            self.normalizer[i, j],
            taylor_expansion,
            taylor_order,
            self.fac,
            self.bc,
        )
        idx = v.argsort()
        out = np.zeros(len(v))
        v = np.clip(v[idx] - self.gamma(i, j), L, U)
        bounds = [(L if k == 0 else v[k - 1], vv) for k, vv in enumerate(v)]
        func = lambda l, u: quad(pdf_product_of_truncated_gaussian, l, u, args=args)[0]
        out[idx] = np.cumsum([func(*b) for b in bounds])
        return out

    def cdf(
        self,
        v: Union[float, List[float], np.ndarray],
        taylor_expansion: bool = False,
        taylor_order: int = 6,
    ) -> np.ndarray:
        """CDF of the hypervolume

        Parameters
        ----------
        v : Union[float, List[float], np.ndarray]
            the hypervolume values
        taylor_expansion : bool, optional
            whether using Taylor expansion to computate the conditional density, by default False
        taylor_order : int, optional
            the order of the Taylor expansion, by default 25

        Returns
        -------
        np.ndarray
            the cumulative probability at volume `v`
        """
        v = self._check_input(v, taylor_expansion, taylor_order)
        if taylor_expansion:
            self.fac = [factorial(i) for i in range(taylor_order)]
            self.bc = [[binom(i, j) for j in range(i + 1)] for i in range(taylor_order)]
        else:
            self.fac, self.bc = None, None
        res, prob = self.__internal_loop_over_cells(
            v, self.cdf_conditional, taylor_expansion=taylor_expansion, taylor_order=taylor_order
        )
        return res + (1 - prob)

    def cdf_monte_carlo(
        self,
        v: Union[float, List[float], np.ndarray],
        n_sample: int = 1e5,
        eval_sd: bool = False,
        n_boostrap: bool = 1e2,
    ) -> np.ndarray:
        """Monte-Carlo approximation to the CDF of hypervolume

        Parameters
        ----------
        v : Union[float, List[float], np.ndarray]
            the hypervolume values
        n_sample : int, optional
            the sample size, by default 1e5
        eval_sd: bool, optional
            whether to estimate the standard deviation of the estimate via boostrapping,
            by default False
        n_boostrap: bool, optional
            the boostrap size, by default 1e2

        Returns
        -------
        np.ndarray
            the cumulative probability at volume `v`
        """
        if isinstance(v, (int, float)):
            v = [v]
        sample = self.mu + self.sigma * np.random.randn(int(n_sample), self.dim)
        fun = lambda x: hv(np.vstack([self.pareto_front.tolist(), x]), self.r)
        mc_fun = lambda v, delta: np.sum(delta <= v) / (1.0 * n_sample)
        delta = np.array(list(map(fun, sample))) - hv(self.pareto_front.tolist(), self.r)
        estimate = np.array([mc_fun(_, delta) for _ in v])
        if eval_sd:
            bs_sample = np.random.choice(delta, size=(int(n_boostrap), len(delta)))
            v = np.array([[mc_fun(_, s) for _ in v] for s in bs_sample])
            sd = v.std(axis=0)
        return (estimate, sd) if eval_sd else estimate

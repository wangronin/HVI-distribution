from __future__ import annotations

import warnings
from typing import Callable, List, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from scipy.integrate import quad
from scipy.special import binom, factorial

from .hypervolume import hypervolume as hv
from .special import D2, D, cdf_product_of_truncated_gaussian, pdf_product_of_truncated_gaussian

import timeit

np.seterr(divide="ignore", invalid="ignore")
warnings.simplefilter("ignore")


__authors__ = ["Hao Wang", "Kaifeng Yang"]


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
        self.neg_inf: List[float] = self.mu - 6.0 * self.sigma
        self.r = r
        self.pareto_front = pareto_front
        (
            self.cells_lb,
            self.cells_ub,
            self.cells_dist,
            self.transformed_lb,
            self.transformed_ub,
            self.normalizer,
        ) = HypervolumeImprovement.__set_cells(
            self.pareto_front, self.N, self.dim, self.mu, self.sigma
        )
        (
            self.prob_in_cell,
            self.dominating_prob,
        ) = HypervolumeImprovement.__compute_probability_in_cell(
            self.dim, self.N, self.cells_lb, self.cells_ub, self.mu, self.sigma
        )
        self.ij = self.get_integral_box_index()
        self.hv_value = hv(pareto_front.tolist(), r) # for HVI calculation by using HV method

    @property
    def max_hvi(self):
        return self.improvement(self.neg_inf, 0, 0)

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
        pareto_front = np.atleast_2d(pareto_front)
        _min = pareto_front.min(axis=0) - 1
        self.neg_inf = [min(self.neg_inf[0], _min[0]), min(self.neg_inf[1], _min[1])]
        pareto_front = np.vstack(
            [pareto_front, [self.neg_inf[0], self.r[1]], [self.r[0], self.neg_inf[1]]]
        )
        idx = pareto_front[:, 0].argsort()
        self._pareto_front = pareto_front[idx]
        self.N = pareto_front.shape[0] - 1

    @njit
    def __set_cells(
        pareto_front: np.ndarray, N: int, dim: int, mu: List[float], sigma: List[float]
    ):
        cells_lb = np.full((N, N, dim), np.nan)
        cells_ub = np.full((N, N, dim), np.nan)
        cells_dist = np.full((N, N, dim), np.nan)
        transformed_lb = np.full((N, N, dim), np.nan)
        transformed_ub = np.full((N, N, dim), np.nan)
        normalizer = np.full((N, N), np.nan)

        for i in range(N):
            for j in range(N - i):
                cells_lb[i, j] = [pareto_front[i, 0], pareto_front[N - j, 1]]
                cells_ub[i, j] = [pareto_front[i + 1, 0], pareto_front[N - 1 - j, 1]]
                transformed_lb[i, j] = [
                    pareto_front[N - j, 0] - pareto_front[i + 1, 0],
                    pareto_front[i, 1] - pareto_front[N - 1 - j, 1],
                ]
                transformed_ub[i, j] = [
                    pareto_front[N - j, 0] - pareto_front[i, 0],
                    pareto_front[i, 1] - pareto_front[N - j, 1],
                ]
                mu_prime = [
                    pareto_front[N - j, 0] - mu[0],
                    pareto_front[i, 1] - mu[1],
                ]
                normalizer[i, j] = D2(
                    transformed_lb[i, j, 0],
                    transformed_ub[i, j, 0],
                    mu_prime[0],
                    sigma[0],
                ) * D2(
                    transformed_lb[i, j, 1],
                    transformed_ub[i, j, 1],
                    mu_prime[1],
                    sigma[1],
                )
                
        cells_dist = cells_ub - cells_lb
        
        return cells_lb, cells_ub, cells_dist, transformed_lb, transformed_ub, normalizer

    @njit
    def __compute_probability_in_cell(dim, N, cells_lb, cells_ub, mu, sigma):
        ij = [(i, j) for i in range(N) for j in range(N - i)]
        _prob_in_cell = np.zeros((N, N, 1))
        for i, j in ij:
            p1 = D2(cells_lb[i, j][0], cells_ub[i, j][0], mu[0], sigma[0])
            p2 = D2(cells_lb[i, j][1], cells_ub[i, j][1], mu[1], sigma[1])
            _prob_in_cell[i, j, ...] = p1 * p2
        # probability in the dominating region w.r.t. the attainment boundary
        dominating_prob = np.nansum(_prob_in_cell)
        return _prob_in_cell, dominating_prob

    @njit
    def mu_prime(pareto_front: np.ndarray, N: int, mu: List[float], i: int, j: int) -> List[float]:
        return [
            pareto_front[N - j, 0] - mu[0],
            pareto_front[i, 1] - mu[1],
        ]

    def gamma(self, i: int, j: int) -> float:
        return self.improvement(self.cells_ub[i, j], i, j) - np.prod(self.transformed_lb[i, j])

    def improvement(
        self, new: List[float], ii: int, jj: int, pareto_front: np.ndarray = None, r: np.ndarray = None
    ) -> float:
        pareto_front = pareto_front if pareto_front else self.pareto_front
        r = r if r else self.r  
        
        return np.sum([np.prod(self.cells_dist[i][j]) for i in range(ii+1, self.N) for j in range(jj+1, self.N-i) if i+j < self.N ])
        # return hv(np.vstack([pareto_front, new]).tolist(), r) - self.hv_value

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

    def get_integral_box_index(
        self,
    ):
        n_sigma = 3
        points_y1 = np.append(self.cells_lb[:, 0, 0], self.cells_ub[:, 0, 0][-1])
        points_y2 = np.append(self.cells_lb[0, :, 1], self.cells_ub[0, :, 1][-1])
        points_y1[0] = -np.inf
        points_y2[0] = -np.inf
        points_y1[-1] = np.inf
        points_y2[-1] = np.inf

        n_y1 = len(points_y1) - 1
        n_y2 = len(points_y2) - 1

        i_start = [
            (i)
            for i in range(n_y1)
            if points_y1[i] <= self.mu[0] - n_sigma * self.sigma[0] < points_y1[i + 1]
        ]
        i_end = [
            (i + 1)
            for i in range(n_y1)
            if points_y1[i] <= self.mu[0] + n_sigma * self.sigma[0] < points_y1[i + 1]
        ]
        j_start = [
            (i)
            for i in range(n_y2)
            if points_y2[i] <= self.mu[1] - n_sigma * self.sigma[1] < points_y2[i + 1]
        ]
        j_end = [
            (i + 1)
            for i in range(n_y2)
            if points_y2[i] <= self.mu[1] + n_sigma * self.sigma[1] < points_y2[i + 1]
        ]

        ij = [
            (i, j)
            for i in range(i_start[0], i_end[0] + 1)
            for j in range(j_start[0], j_end[0] + 1)
            if i + j < self.N
        ]
        # ij = [(i, j) for i in range(self.N) for j in range(self.N - i)]
        return ij

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

        if self.ij == []:
            return np.zeros(len(v))
        else:
            res = (
                Parallel(n_jobs=n_jobs)(delayed(func)(v, i, j, **kwargs) for i, j in self.ij)
                if parallel
                else [func(v, i, j, **kwargs) for i, j in self.ij]
            )
            for k, (i, j) in enumerate(self.ij):
                terms[i, j, :] = res[k]
            return np.nansum(terms * self.prob_in_cell, axis=(0, 1))

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
            HypervolumeImprovement.mu_prime(self.pareto_front, self.N, self.mu, i, j),
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
        res = self.__internal_loop_over_cells(
            v,
            self.pdf_conditional,
            taylor_expansion=taylor_expansion,
            taylor_order=taylor_order,
        )
        # NOTE: the density at zero volume is a Dirac delta
        if np.any(idx):
            res = res.astype(object)
            _ = res[idx]
            prob = 1 - self.dominating_prob
            res[idx] = f"{prob} * delta" if _ == 0 else f"{prob} * delta + {_}"
        return res

    def cdf_conditional(
        self,
        v: np.ndarray,
        i: int,
        j: int,
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
        Returns
        -------
        np.ndarray
            the cumulative probability at volume `v`
        """
        L, U = np.prod(self.transformed_lb[i, j]), np.prod(self.transformed_ub[i, j])
        args = (
            HypervolumeImprovement.mu_prime(self.pareto_front, self.N, self.mu, i, j),
            self.sigma,
            self.transformed_lb[i, j],
            self.transformed_ub[i, j],
            self.normalizer[i, j],
        )
        a = np.clip(v - self.gamma(i, j), L, U)
        out = np.array([cdf_product_of_truncated_gaussian(v, *args) for v in a])
        return out

    def cdf(self, v: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """Exact CDF of the hypervolume
        Parameters
        ----------
        v : Union[float, List[float], np.ndarray]
            the hypervolume values
        Returns
        -------
        np.ndarray
            the cumulative probability at volume `v`
        """
        res = self.__internal_loop_over_cells(v, self.cdf_conditional)
        res[res == np.inf] = 0
        return res + (1 - self.dominating_prob)

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
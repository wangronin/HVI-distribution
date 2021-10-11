from __future__ import annotations

import warnings
from typing import Callable, List, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from numba import njit

from .hypervolume import hypervolume as hv
from .special import D2, cdf_product_of_truncated_gaussian, pdf_product_of_truncated_gaussian

np.seterr(divide="ignore", invalid="ignore")
warnings.simplefilter("ignore")


__authors__ = ["Hao Wang", "Kaifeng Yang"]


@njit
def _set_cells(pareto_front: np.ndarray, N: int, dim: int, mu: List[float], sigma: List[float]):
    cells_volume = np.zeros((N, N))
    cells_lb = np.full((N, N, dim), np.nan)
    cells_ub = np.full((N, N, dim), np.nan)
    mu_prime = np.full((N, N, dim), np.nan)

    transformed_lb = np.full((N, N, dim), np.nan)
    transformed_ub = np.full((N, N, dim), np.nan)
    normalizer = np.full((N, N), np.nan)

    for i in range(N):
        for j in range(N - i):
            cells_lb[i, j] = [pareto_front[i, 0], pareto_front[N - j, 1]]
            cells_ub[i, j] = [pareto_front[i + 1, 0], pareto_front[N - 1 - j, 1]]
            cells_volume[i, j] = np.prod(cells_ub[i, j] - cells_lb[i, j])
            transformed_lb[i, j] = [
                pareto_front[N - j, 0] - pareto_front[i + 1, 0],
                pareto_front[i, 1] - pareto_front[N - 1 - j, 1],
            ]
            transformed_ub[i, j] = [
                pareto_front[N - j, 0] - pareto_front[i, 0],
                pareto_front[i, 1] - pareto_front[N - j, 1],
            ]
            mu_prime[i, j] = [
                pareto_front[N - j, 0] - mu[0],
                pareto_front[i, 1] - mu[1],
            ]
            normalizer[i, j] = D2(
                transformed_lb[i, j, 0],
                transformed_ub[i, j, 0],
                mu_prime[i, j, 0],
                sigma[0],
            ) * D2(
                transformed_lb[i, j, 1],
                transformed_ub[i, j, 1],
                mu_prime[i, j, 1],
                sigma[1],
            )

    for l in range(N):
        for i, j in [(k, N - l - 1 - k) for k in range(N - l)]:
            term1 = cells_volume[i, j + 1] if j + 1 < N else 0
            term2 = cells_volume[i + 1, j] if i + 1 < N else 0
            term3 = cells_volume[i + 1, j + 1] if i + 1 < N and j + 1 < N else 0
            cells_volume[i, j] += term1 + term2 - term3

    return (
        cells_lb,
        cells_ub,
        cells_volume,
        mu_prime,
        transformed_lb,
        transformed_ub,
        normalizer,
    )


@njit
def _compute_probability_in_cell(
    dim: int,
    N: int,
    cells_lb: List[float],
    cells_ub: List[float],
    mu: List[float],
    sigma: List[float],
) -> Tuple[np.ndarray, float]:
    ij = [(i, j) for i in range(N) for j in range(N - i)]
    # the probability of the Gaussian objective point lying in each cell
    prob_in_cell = np.zeros((N, N, 1))
    for i, j in ij:
        # p = np.array(
        #     [D2(cells_lb[i, j][k], cells_ub[i, j][k], mu[k], sigma[k]) for k in range(dim)]
        # )
        p1 = D2(cells_lb[i, j][0], cells_ub[i, j][0], mu[0], sigma[0])
        p2 = D2(cells_lb[i, j][1], cells_ub[i, j][1], mu[1], sigma[1])
        prob_in_cell[i, j, ...] = p1 * p2
    # probability in the dominating region w.r.t. the attainment boundary
    dominating_prob = np.nansum(prob_in_cell)
    if dominating_prob > 1: dominating_prob = 1
    return prob_in_cell, dominating_prob


@njit
def _gamma(cells_volume, transformed_lb, N, i: int, j: int) -> float:
    v = cells_volume[i + 1, j + 1] if i + 1 < N and j + 1 < N else 0
    return v - np.prod(transformed_lb[i, j])


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
            self.cells_volume,
            self.mu_prime,
            self.transformed_lb,
            self.transformed_ub,
            self.normalizer,
        ) = _set_cells(self.pareto_front, self.N, self.dim, self.mu, self.sigma)
        self.prob_in_cell, self.dominating_prob = _compute_probability_in_cell(
            self.dim, self.N, self.cells_lb, self.cells_ub, self.mu, self.sigma
        )
        self.ij = self._make_index()

    @property
    def max_hvi(self):
        return self.cells_volume[0, 0]

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

    def _check_input(self, v: Union[float, List[float], np.ndarray]) -> np.ndarray:
        if isinstance(v, (int, float)):
            v = [v]
        v = np.array(v)
        return v

    def _make_index(self) -> List[Tuple[int, int]]:
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

    def pdf_conditional(self, v: np.ndarray, i: int, j: int) -> np.ndarray:
        """Conditional PDF of hypervolume when restricting the objective point in the cell (i, j)
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
            the conditional probability density at volume `v`
        """
        par = (
            self.mu_prime[i, j],
            self.sigma,
            self.transformed_lb[i, j],
            self.transformed_ub[i, j],
            self.normalizer[i, j],
        )
        gamma = _gamma(self.cells_volume, self.transformed_lb, self.N, i, j)
        return np.array([pdf_product_of_truncated_gaussian(p, *par) for p in v - gamma])

    def pdf(self, v: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """PDF of the hypervolume
        Parameters
        ----------
        v : Union[float, List[float], np.ndarray]
            the hypervolume values
        Returns
        -------
        np.ndarray
            the probability density at volume `v`
        """
        v = self._check_input(v)
        idx = v == 0
        res = self.__internal_loop_over_cells(v, self.pdf_conditional)
        # NOTE: the density at zero volume is the Dirac delta
        if np.any(idx):
            res = res.astype(object)
            _ = res[idx]
            prob = 1 - self.dominating_prob
            res[idx] = f"{prob} * delta" if _ == 0 else f"{prob} * delta + {_}"
        return res

    def cdf_conditional(self, v: np.ndarray, i: int, j: int) -> np.ndarray:
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
            self.mu_prime[i, j],
            self.sigma,
            self.transformed_lb[i, j],
            self.transformed_ub[i, j],
            self.normalizer[i, j],
        )
        a = np.clip(v - _gamma(self.cells_volume, self.transformed_lb, self.N, i, j), L, U)
        return np.array([cdf_product_of_truncated_gaussian(v, *args) for v in a])

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

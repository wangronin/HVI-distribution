from __future__ import annotations

import warnings
from typing import Callable, List, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import quad
from scipy.special import binom, factorial

from .hypervolume import hypervolume as hv
from .special import D, cdf_product_of_truncated_gaussian, pdf_product_of_truncated_gaussian

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
        self.__set_cells(self.pareto_front)
        self.__compute_probability_in_cell()
        # self.fac: List[int] = None
        # self.bc: List[float] = None

    @property
    def max_hvi(self):
        return self.improvement(self.neg_inf)

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

    def __set_cells(self, pareto_front: np.ndarray):
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
                self.normalizer[i, j] = np.prod(
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

    def __compute_probability_in_cell(self):
        ij = [(i, j) for i in range(self.N) for j in range(self.N - i)]
        self._prob_in_cell = np.zeros((self.N, self.N))
        for i, j in ij:
            self._prob_in_cell[i, j] = self.prob_in_cell(i, j)
        # probability in the dominating region w.r.t. the attainment boundary
        self.dominating_prob = np.nansum(self._prob_in_cell)

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
        ij = self.get_integral_box_index()

        if ij == []:
            return np.zeros(len(v)), 0
        else:
            res = (
                Parallel(n_jobs=n_jobs)(delayed(func)(v, i, j, **kwargs) for i, j in ij)
                if parallel
                else [func(v, i, j, **kwargs) for i, j in ij]
            )
            for k, (i, j) in enumerate(ij):
                terms[i, j, :] = res[k]
            return np.nansum(terms * self._prob_in_cell, axis=(0, 1))

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
            self.mu_prime(i, j),
            self.sigma,
            self.transformed_lb[i, j],
            self.transformed_ub[i, j],
            self.normalizer[i, j],
        )
        a = np.clip(v - self.gamma(i, j), L, U)
        out = np.array([cdf_product_of_truncated_gaussian(v, *args) for v in a])
        return out

    def cdf_conditional_(
        self,
        v: np.ndarray,
        i: int,
        j: int,
    ) -> np.ndarray:
        """This is the old, deprecated code...
        Conditional CDF of hypervolume when restricting the objective point in the cell (i, j)
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
            self.mu_prime(i, j),
            self.sigma,
            self.transformed_lb[i, j],
            self.transformed_ub[i, j],
            self.normalizer[i, j],
            self.fac,
            self.bc,
        )
        idx = v.argsort()
        out = np.zeros(len(v))
        v = np.clip(v[idx] - self.gamma(i, j), L, U)
        bounds = [(L if k == 0 else v[k - 1], vv) for k, vv in enumerate(v)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            func = lambda l, u: quad(pdf_product_of_truncated_gaussian, l, u, args=args, limit=20)[
                0
            ]
            out[idx] = np.cumsum([func(*b) for b in bounds])
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

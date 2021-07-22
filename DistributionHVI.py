import math
from typing import List, Union

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from hypervolume import hypervolume as hv

__authors__ = ["Kaifeng Yang", "Hao Wang"]


def D(L, U, loc, scale):
    return norm.cdf(U, loc, scale) - norm.cdf(L, loc, scale)


def integrand_eq4(x, mu, sigma, p):
    return (
        np.exp(-0.5 * ((x - mu[0]) ** 2 / sigma[0] ** 2 + (p / x - mu[1]) ** 2 / sigma[1] ** 2))
        / x
    )


def integrand_eq6(x, sigma, p, m, n):
    return x ** (2 * m - n - 1) * np.exp(
        -0.5 * (x ** 2 / sigma[0] ** 2 + p ** 2 / (x ** 2 * sigma[1] ** 2))
    )


def pdf_product_of_truncated_gaussian(
    p: float,
    mean: List[float],
    sigma: List[float],
    lower: List[float],
    upper: List[float],
    taylor_expansion: bool = False,
    taylor_order: int = 25,
) -> float:
    (L1, L2), (U1, U2) = lower, upper
    if L1 * U2 > U1 * L2:  # swap y_1' and y_2'
        (L2, L1), (U2, U1) = lower, upper
        mean = mean[1], mean[0]
        sigma = sigma[1], sigma[0]

    D1, D2 = D(L1, U1, mean[0], sigma[0]), D(L2, U2, mean[1], sigma[1])
    if L1 * L2 <= p < L1 * U2:
        alpha = L1
        belta = p / L2
    elif L1 * U2 <= p < U1 * L2:
        alpha = p / U2
        belta = p / L2
    elif U1 * L2 <= p <= U1 * U2:
        alpha = p / U2
        belta = U1
    else:
        print("error in lb and ub")

    if not taylor_expansion:
        out = quad(
            integrand_eq4,
            alpha,
            belta,
            args=(mean, sigma, p),
            limit=1000,
            epsabs=1e-30,
            epsrel=1e-10,
        )[0]
    else:
        K = taylor_order
        faInCell = np.zeros((K, K + 1))
        term1 = np.exp(-0.5 * (mean[0] ** 2 / sigma[0] ** 2 + mean[1] ** 2 / sigma[1] ** 2))
        for n in range(K):
            for m in range(n + 1):
                res = quad(
                    integrand_eq6,
                    alpha,
                    belta,
                    args=(sigma, p, m, n),
                )[0]

                faInCell[n, m] = (
                    p ** (n - m)
                    / math.factorial(n)
                    * (math.factorial(n) / math.factorial(m) / math.factorial(n - m))
                    * (mean[0] / sigma[0] ** 2) ** m
                    * (mean[1] / sigma[1] ** 2) ** (n - m)
                    * res
                )
        out = term1 * np.nansum(faInCell)

    return out / (2 * np.pi * sigma[0] * sigma[1] * D1 * D2)


class HypervolumeImprovement:
    def __init__(self, pareto_front: np.ndarray, r: Union[List, np.ndarray]):
        self.lbInf: float = -4
        self.r = r
        self.pareto_front = pareto_front
        self.set_cells(self.pareto_front)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        self.dim = len(r)
        self._r = r

    @property
    def pareto_front(self):
        return self._pareto_front

    @pareto_front.setter
    def pareto_front(self, pareto_front):
        pareto_front = np.vstack((pareto_front, [self.lbInf, self.r[1]], [self.r[0], self.lbInf]))
        idx = pareto_front[:, 0].argsort()
        self._pareto_front = pareto_front[idx]
        self.N = pareto_front.shape[0] - 1

    def set_cells(self, pareto_front: np.ndarray):
        self.cells_lb = np.full((self.N, self.N, self.dim), np.nan)
        self.cells_ub = np.full((self.N, self.N, self.dim), np.nan)
        self.truncated_lb = np.full((self.N, self.N, self.dim), np.nan)
        self.truncated_ub = np.full((self.N, self.N, self.dim), np.nan)
        for i in range(self.N):
            for j in range(self.N - i):
                self.cells_lb[i, j] = [pareto_front[i, 0], pareto_front[self.N - j, 1]]
                self.cells_ub[i, j] = [pareto_front[i + 1, 0], pareto_front[self.N - 1 - j, 1]]
                self.truncated_lb[i, j] = [
                    pareto_front[self.N - j, 0] - pareto_front[i + 1, 0],
                    pareto_front[i, 1] - pareto_front[self.N - 1 - j, 1],
                ]
                self.truncated_ub[i, j] = [
                    pareto_front[self.N - j, 0] - pareto_front[i, 0],
                    pareto_front[i, 1] - pareto_front[self.N - j, 1],
                ]

    def improvement(
        self, new: List[float], pareto_front: np.ndarray = None, r: np.ndarray = None
    ) -> float:
        pareto_front = pareto_front if pareto_front else self.pareto_front
        r = r if r else self.r
        return hv(np.vstack(pareto_front, new).T.tolist(), r) - hv(pareto_front.T.tolist(), r)

    def pdf_conditional(self) -> float:
        pass

    def pdf(self) -> float:
        pass

    def cdf_conditional(
        self,
        a: float,
        mu: List[float],
        sigma: List[float],
        i: int,
        j: int,
        taylor_expansion: bool = False,
        taylor_order: int = 25,
    ) -> float:
        L1, L2 = self.truncated_lb[i, j]
        U1, U2 = self.truncated_ub[i, j]
        mu_ = [self.pareto_front[0, self.N - j] - mu[0], self.pareto_front[1, i] - mu[1]]
        gamma = self.improvement(self.cells_ub[i, j]) - L1 * L2

        a_ = min(a - gamma, U1 * U2)
        if a_ < L1 * L2:
            return 0

        prob_ij = D(self.cells_lb[i, j][0], self.cells_ub[i, j][0], mu[0], sigma[0]) * D(
            self.cells_lb[i, j][1], self.cells_ub[i, j][1], mu[0], sigma[1]
        )
        if taylor_expansion:
            out = quad(
                pdf_product_of_truncated_gaussian,
                L1 * L2,
                a_,
                args=(mu_, sigma, [L1, L2], [U1, U2], True, taylor_order),
            )[0]
        else:
            out = quad(
                pdf_product_of_truncated_gaussian,
                L1 * L2,
                a_,
                args=(mu_, sigma, [L1, L2], [U1, U2]),
                limit=50,
            )[0]
        return out * prob_ij

    def cdf_monte_carlo(
        self, a: float, mu: List[float], sigma: List[float], N: int = 1e5
    ) -> float:
        mu, sigma = np.array(mu), np.array(sigma)
        sample = mu + sigma * np.random.randn(N, self.dim)
        fun = lambda x: hv(np.r_[self.pareto_front, x].T.tolist(), self.r)
        delta = np.array(list(map(fun, sample))) - hv(self.pareto_front.T.tolist(), self.r)
        return np.sum(np.bitwise_and(delta > 0, delta <= a)) / (1.0 * N)

    def cdf(
        self,
        a: float,
        mu: List[float],
        sigma: List[float],
        taylor_expansion: bool = False,
        taylor_order: int = 25,
    ) -> float:
        terms = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N - i):
                terms[i, j] = self.cdf_conditional(
                    a, mu, sigma, i, j, taylor_expansion, taylor_order
                )
        return np.sum(terms)

    # def density_cosh(self, x, mean, variance, p, m, n):
    #     return (
    #         0.5
    #         * (np.sqrt(variance[0] / variance[1] * p) ** (m - n / 2))
    #         * (np.exp(-p / np.sqrt(variance[0] * variance[1]) * np.cosh(x) + (m - n / 2) * x))
    #     )

    # def computeTaylorSeries_PDF(self, p, mean, variance, truncatedLB, truncatedUB):
    #     nTaylor = self.nTaylor

    #     L1, L2 = truncatedLB
    #     U1, U2 = truncatedUB

    #     if L1 * U2 > U1 * L2:  # swap y_1' and y_2'
    #         L2, L1 = truncatedLB
    #         U2, U1 = truncatedUB
    #         mean = mean[1], mean[0]
    #         variance = variance[1], variance[0]
    #     # else:
    #     #     print('test')

    #     D1, D2 = self.get_D(L1, U1, mean[0], variance[0] ** 0.5), self.get_D(
    #         L2, U2, mean[1], variance[1] ** 0.5
    #     )

    #     if L1 * L2 <= p < L1 * U2:
    #         alpha = L1
    #         belta = p / L2
    #     elif L1 * U2 <= p < U1 * L2:
    #         alpha = p / U2
    #         belta = p / L2
    #     elif U1 * L2 <= p <= U1 * U2:
    #         alpha = p / U2
    #         belta = U1
    #     else:
    #         print("error in lb and ub")

    #     faInCell = np.zeros((nTaylor, nTaylor + 1))

    #     theta = np.log(np.sqrt(variance[1])) - np.log(np.sqrt(variance[0])) * p
    #     term1 = np.exp(-0.5 * (mean[0] ** 2 / variance[0] + mean[1] ** 2 / variance[1]))

    #     lb = 2 * np.log(alpha) + theta
    #     ub = 2 * np.log(belta) + theta
    #     for n in range(nTaylor):
    #         for m in range(n + 1):

    #             # tmp = quad(
    #             #     density_cosh,
    #             #     lb,
    #             #     ub,
    #             #     args=(mean, variance, p, m, n),
    #             # )

    #             tmp = quad(
    #                 self.density_eq7left,
    #                 alpha,
    #                 belta,
    #                 args=(mean, variance, p, m, n),
    #             )

    #             faInCell[n, m] = (
    #                 p ** (n - m)
    #                 / math.factorial(n)
    #                 * (math.factorial(n) / math.factorial(m) / math.factorial(n - m))
    #                 * (mean[0] / variance[0]) ** m
    #                 * (mean[1] / variance[1]) ** (n - m)
    #                 * tmp[0]
    #             )

    #     return (
    #         term1
    #         * np.nansum(faInCell)
    #         / (2 * np.pi * np.sqrt(variance[0]) * np.sqrt(variance[1]) * D1 * D2)
    #     )

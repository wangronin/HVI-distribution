import sys
import warnings
from typing import List

import numpy as np
from numba import carray, cfunc, jit
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable
from scipy.integrate import quad
from scipy.stats import norm, truncnorm


def jit_integrand(integrand_function):
    jitted_function = jit(integrand_function, nopython=True, error_model="numpy")

    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx, n)
        return jitted_function(values)

    return LowLevelCallable(wrapped.ctypes)


@jit(nopython=True, error_model="numpy")
def erf(x: float) -> float:
    """faster approximation of Gaussian error function"""
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    x_ = x if x >= 0 else -1.0 * x
    t = 1 / (1 + p * x_)
    out = 1 - (a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5) * np.exp(
        -1.0 * x_ ** 2
    )
    return out if x >= 0 else -1.0 * out


@jit(nopython=True, error_model="numpy")
def dnorm(x: float, mean: float, sd: float) -> float:
    return np.exp(-0.5 * (x - mean) ** 2 / sd ** 2) / (np.sqrt(2 * np.pi) * sd)


@jit(nopython=True, error_model="numpy")
def pnorm(x: float, loc: float, scale: float) -> float:
    return 0.5 * (1 + erf((x - loc) / scale / np.sqrt(2)))


@jit(nopython=True, error_model="numpy")
def D2(L: float, U: float, loc: float, scale: float) -> float:
    return 0.5 * (erf((U - loc) / scale / np.sqrt(2)) - erf((L - loc) / scale / np.sqrt(2)))


def D(L, U, loc, scale):
    out = D2(L, U, loc, scale)
    if out == 0:
        val = np.mean(
            norm.pdf(np.linspace(L, U, 10), loc=loc, scale=scale)
            / truncnorm.pdf(
                np.linspace(L, U, 10), (L - loc) / scale, (U - loc) / scale, loc=loc, scale=scale
            )
        )
        if not np.isnan(val):
            out = val
    return out


@jit_integrand
def integrand_eq4(args):
    x, mu0, mu1, sigma0, sigma1, p = args
    out = np.exp(-0.5 * (((x - mu0) / sigma0) ** 2 + ((p / x - mu1) / sigma1) ** 2)) / x
    return out


def integrand_eq7(x, m, n, sigma, p):
    return x ** (2 * m - n - 1) * np.exp(-0.5 * ((x / sigma[0]) ** 2 + (p / x / sigma[1]) ** 2))


@jit_integrand
def integrand_cdf(args):
    x, a, mu0, mu1, sigma0, sigma1 = args
    return 0.5 * (1 + erf((a / x - mu1) / sigma1 / np.sqrt(2))) * dnorm(x, mu0, sigma0)


def cdf_product_of_truncated_gaussian(
    p: float,
    mean: List[float],
    sigma: List[float],
    lower: List[float],
    upper: List[float],
    normalizer: float,
):
    if normalizer == 0:
        return 0

    (L1, L2), (U1, U2) = lower, upper
    if L1 * U2 > U1 * L2:  # swap y_1' and y_2'
        (L2, L1), (U2, U1) = lower, upper
        mean = mean[1], mean[0]
        sigma = sigma[1], sigma[0]

    l, u = max(L1, p / U2), min(U1, p / L2)
    term1 = max(D2(L1, p / U2, mean[0], sigma[0]), 0) * D2(L2, U2, mean[1], sigma[1])
    term2 = D2(l, u, mean[0], sigma[0]) * 0.5 * (1 + erf((L2 - mean[1]) / sigma[1] / np.sqrt(2)))
    term3 = quad(integrand_cdf, l, u, args=(p, mean[0], mean[1], sigma[0], sigma[1]))[0]
    return (term1 - term2 + term3) / normalizer


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
    if normalizer == 0:
        return 0
    normalizer *= 2 * np.pi * np.prod(sigma)
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
                if alpha <= 0 <= beta:
                    out = (
                        quad(
                            integrand_eq4,
                            alpha,
                            -1 * sys.float_info.min,
                            args=(mean[0], mean[1], sigma[0], sigma[1], p),
                        )[0]
                        + quad(
                            integrand_eq4,
                            sys.float_info.min,
                            beta,
                            args=(mean[0], mean[1], sigma[0], sigma[1], p),
                        )[0]
                    )
                else:
                    out = quad(
                        integrand_eq4,
                        alpha,
                        beta,
                        args=(mean[0], mean[1], sigma[0], sigma[1], p),
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

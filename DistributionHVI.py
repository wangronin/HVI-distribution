#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:33:54 2021

@author: kaifengyang
"""

import math

import numpy as np

# import pygmo as pg
from scipy.integrate import quad
from scipy.stats import norm

import hypervolume as hv

__authors__ = ["Kaifeng Yang", "Hao Wang"]


class DistributionHVI(object):
    def __init__(self, pf, r, nTaylor):
        self.pf = pf
        self.r = r
        self.lbInf = -4
        self.nTaylor = nTaylor

    def original(self, x, mean, variance, p):
        return (
            np.exp(
                -0.5 * ((x - mean[0]) ** 2 / variance[0] + (p / x - mean[1]) ** 2 / variance[1])
            )
            / x
        )

    def get_D(self, L, U, loc, scale):
        return norm.cdf(U, loc, scale) - norm.cdf(L, loc, scale)

    def density_eq7left(self, x, mean, variance, p, m, n):
        return x ** (2 * m - n - 1) * np.exp(
            -0.5 * (x ** 2 / variance[0] + p ** 2 / (x ** 2 * variance[1]))
        )

    def method_without_taylor(self, p, mean, variance, truncatedLB, truncatedUB):
        L1, L2 = truncatedLB
        U1, U2 = truncatedUB

        if L1 * U2 > U1 * L2:  # swap y_1' and y_2'
            L2, L1 = truncatedLB
            U2, U1 = truncatedUB
            mean = mean[1], mean[0]
            variance = variance[1], variance[0]

        D1, D2 = self.get_D(L1, U1, mean[0], variance[0] ** 0.5), self.get_D(
            L2, U2, mean[1], variance[1] ** 0.5
        )
        # D1, D2 = 1, 1

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

        res = quad(
            self.original,
            alpha,
            belta,
            args=(mean, variance, p),
            limit=int(1e6),
            epsabs=1e-30,
            epsrel=1e-10,
        )[0]
        return res / (2 * np.pi * np.sqrt(variance[0]) * np.sqrt(variance[1]) * D1 * D2)

    def density_cosh(self, x, mean, variance, p, m, n):
        return (
            0.5
            * (np.sqrt(variance[0] / variance[1] * p) ** (m - n / 2))
            * (np.exp(-p / np.sqrt(variance[0] * variance[1]) * np.cosh(x) + (m - n / 2) * x))
        )

    def rcond_norm(self, N, loc, scale, L, U):
        alpha = norm.cdf(L, loc, scale)
        beta = norm.cdf(U, loc, scale)
        Z = beta - alpha
        return norm.ppf(np.random.rand(int(N)) * Z + alpha, loc, scale)

    def hvi(self, pf, r, point_add):  # get HVI value
        pfs = pf.tolist()
        # HV value for origianl PF in the transposed space
        hvPrevious = hv.hypervolume(pfs, r)
        pf_temp = np.vstack((pfs, point_add)).tolist()
        # HV values of adding u in PF in the transposed sapce
        hvAdded = hv.hypervolume(pf_temp, r)
        return hvAdded - hvPrevious

    def computeTaylorSeries(self, p, mean, variance, truncatedLB, truncatedUB):
        nTaylor = self.nTaylor
        L1, L2 = truncatedLB
        U1, U2 = truncatedUB

        if L1 * U2 > U1 * L2:  # swap y_1' and y_2'
            L2, L1 = truncatedLB
            U2, U1 = truncatedUB
            mean = mean[1], mean[0]
            variance = variance[1], variance[0]

        if p < L1 * L2:
            p = L1 * L2

        D1, D2 = self.get_D(L1, U1, mean[0], variance[0] ** 0.5), self.get_D(
            L2, U2, mean[1], variance[1] ** 0.5
        )

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

        faInCell = np.zeros((nTaylor, nTaylor + 1))

        theta = np.log(np.sqrt(variance[1])) - np.log(np.sqrt(variance[0])) * p
        term1 = np.exp(-0.5 * (mean[0] ** 2 / variance[0] + mean[1] ** 2 / variance[1]))

        lb = 2 * np.log(alpha) + theta
        ub = 2 * np.log(belta) + theta
        for n in range(nTaylor):
            for m in range(n + 1):

                # tmp = quad(
                #     density_cosh,
                #     lb,
                #     ub,
                #     args=(mean, variance, p, m, n),
                # )

                tmp = quad(
                    self.density_eq7left,
                    alpha,
                    belta,
                    args=(mean, variance, p, m, n),
                )

                faInCell[n, m] = (
                    p ** (n - m)
                    / math.factorial(n)
                    * (math.factorial(n) / math.factorial(m) / math.factorial(n - m))
                    * (mean[0] / variance[0]) ** m
                    * (mean[1] / variance[1]) ** (n - m)
                    * tmp[0]
                )

        return (
            term1
            * np.nansum(faInCell)
            / (2 * np.pi * np.sqrt(variance[0]) * np.sqrt(variance[1]) * D1 * D2)
        )

    def computeTaylorSeries_PDF(self, p, mean, variance, truncatedLB, truncatedUB):
        nTaylor = self.nTaylor

        L1, L2 = truncatedLB
        U1, U2 = truncatedUB

        if L1 * U2 > U1 * L2:  # swap y_1' and y_2'
            L2, L1 = truncatedLB
            U2, U1 = truncatedUB
            mean = mean[1], mean[0]
            variance = variance[1], variance[0]
        # else:
        #     print('test')

        D1, D2 = self.get_D(L1, U1, mean[0], variance[0] ** 0.5), self.get_D(
            L2, U2, mean[1], variance[1] ** 0.5
        )

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

        faInCell = np.zeros((nTaylor, nTaylor + 1))

        theta = np.log(np.sqrt(variance[1])) - np.log(np.sqrt(variance[0])) * p
        term1 = np.exp(-0.5 * (mean[0] ** 2 / variance[0] + mean[1] ** 2 / variance[1]))

        lb = 2 * np.log(alpha) + theta
        ub = 2 * np.log(belta) + theta
        for n in range(nTaylor):
            for m in range(n + 1):

                # tmp = quad(
                #     density_cosh,
                #     lb,
                #     ub,
                #     args=(mean, variance, p, m, n),
                # )

                tmp = quad(
                    self.density_eq7left,
                    alpha,
                    belta,
                    args=(mean, variance, p, m, n),
                )

                faInCell[n, m] = (
                    p ** (n - m)
                    / math.factorial(n)
                    * (math.factorial(n) / math.factorial(m) / math.factorial(n - m))
                    * (mean[0] / variance[0]) ** m
                    * (mean[1] / variance[1]) ** (n - m)
                    * tmp[0]
                )

        return (
            term1
            * np.nansum(faInCell)
            / (2 * np.pi * np.sqrt(variance[0]) * np.sqrt(variance[1]) * D1 * D2)
        )

    def get_q(self, pf, r):
        ## ---- grid the non-dominated sapce -------
        # Pareto-front approximation set with reference point
        # pfr = np.vstack((pf,[-math.inf, r[1]], [r[0], -math.inf])) ## BUG: lower bound can be -infi
        pfr = np.vstack((pf, [self.lbInf, r[1]], [r[0], self.lbInf]))
        pfr_test = np.transpose(pfr)
        # sorted pfr
        q = np.transpose(
            pfr[
                pfr[:, 0].argsort(),
            ]
        )
        return q

    def get_cell_LUB(self, q):
        num = q.shape[1] - 1
        # upperbound of each cell
        cellUB = np.array(
            [[("nan", "nan") for j in range(0, num)] for i in range(0, num)], dtype=float
        )
        # lowerbound of each cell
        cellLB = np.array(
            [[("nan", "nan") for j in range(0, num)] for i in range(0, num)], dtype=float
        )

        for i in range(0, num):
            for j in range(0, num - i):
                cellLB[i, j] = [q[0, i], q[1, num - j]]
                cellUB[i, j] = [q[0, i + 1], q[1, num - 1 - j]]
        self.cellLB = cellLB
        self.cellUB = cellUB

    def initialization(self, mu, ss, a):
        self.q = self.get_q(self.pf, self.r)
        self.get_cell_LUB(self.q)

        self.mu = mu
        self.variance = ss

        m = self.q.shape[0]
        self.m = m
        num = self.q.shape[1] - 1
        self.num = num

        self.gamma = np.array([[(0) for j in range(0, num)] for i in range(0, num)], dtype=float)
        self.truncatedLB = np.array(
            [[("nan", "nan") for j in range(0, num)] for i in range(0, num)], dtype=float
        )
        self.truncatedUB = np.array(
            [[("nan", "nan") for j in range(0, num)] for i in range(0, num)], dtype=float
        )
        # initialized D:
        self.Dset = np.array(
            [[("nan",) * m for j in range(0, num)] for i in range(0, num)], dtype=float
        )

        self.muPrime = np.array(
            [[("nan",) * m for j in range(0, num)] for i in range(0, num)], dtype=float
        )

        # initialized HV for each cell
        self.HVIDistinCell = np.array(
            [["nan" for j in range(0, num)] for i in range(0, num)], dtype=float
        )
        self.HVIDistinCell2 = np.array(
            [["nan" for j in range(0, num)] for i in range(0, num)], dtype=float
        )

        self.pInCell = np.array([[(0) for j in range(0, num)] for i in range(0, num)], dtype=float)
        self.aInCell = np.array([[(0) for j in range(0, num)] for i in range(0, num)], dtype=float)

    def compute_MC(self, pf, r, mu, ss, a):
        return 0

    def compute_without_taylor(self, pf, r, mu, ss, a):
        return 0

    def compute_HVI_dist(self, method, mu, ss, a):
        self.initialization(mu, ss, a)
        for i in range(0, self.num):
            for j in range(0, self.num - i):
                if i + j < self.num - 1:
                    self.truncatedLB[i, j] = [
                        self.q[0, self.num - j] - self.cellUB[i, j][0],
                        self.q[1, i] - self.cellUB[i, j][1],
                    ]
                    self.truncatedUB[i, j] = [
                        self.q[0, self.num - j] - self.cellLB[i, j][0],
                        self.q[1, i] - self.cellLB[i, j][1],
                    ]
                else:
                    self.truncatedLB[i, j] = [0, 0]
                    self.truncatedUB[i, j] = [
                        self.cellUB[i, j][0] - self.cellLB[i, j][0],
                        self.cellUB[i, j][1] - self.cellLB[i, j][1],
                    ]

                self.muPrime[i, j][0] = self.q[0, self.num - j] - self.mu[0]
                self.muPrime[i, j][1] = self.q[1, i] - self.mu[1]

                self.Dset[i, j] = [
                    norm.cdf(
                        self.truncatedUB[i, j][k], loc=self.muPrime[i, j][k], scale=np.sqrt(ss[k])
                    )
                    - norm.cdf(
                        self.truncatedLB[i, j][k], loc=self.muPrime[i, j][k], scale=np.sqrt(ss[k])
                    )
                    for k in range(0, self.m)
                ]

                self.gamma[i, j] = (
                    self.hvi(self.pf, self.r, self.cellUB[i, j])
                    - self.truncatedLB[i, j][0] * self.truncatedLB[i, j][1]
                )

                # area = (self.cellUB[i, j][0] - self.cellLB[i, j][0]) * (self.cellUB[i, j][1] - self.cellLB[i, j][1])
                # if a > area:
                #     self.aInCell[i, j] = area
                # else:
                #     self.aInCell[i, j] = a

                # aInCell[i,j] = a

                # self.pInCell[i, j] = self.aInCell[i, j] - self.gamma[i, j]

                L1, L2 = self.truncatedLB[i, j]
                U1, U2 = self.truncatedUB[i, j]
                muP = [self.muPrime[i, j][0], self.muPrime[i, j][1]]
                a_ = a - self.gamma[i, j]
                if a_ < L1 * L2 or a_ > U1 * U2:
                    self.HVIDistinCell[i, j] = 0
                    continue

                # self.aInCell[i, j] = L1 * L2
                # self.pInCell[i, j] = U1 * U2

                # if self.pInCell[i, j] < L1 * L2:
                #     self.pInCell[i, j] = L1 * L2
                # print('error in Cell(%d,%d)' %(i,j))

                if method == "Taylor":
                    # self.HVIDistinCell[i,j] = self.computeTaylorSeries(self.pInCell[i, j],
                    #                                                    muP, ss,
                    #                                                    [L1, L2],
                    #                                                    [U1, U2],
                    #                                                    )

                    rst = quad(
                        self.computeTaylorSeries,
                        L1 * L2,
                        a_,
                        args=(muP, ss, [L1, L2], [U1, U2]),
                    )[0]

                    ProInCell = self.get_D(
                        self.cellLB[i, j][0], self.cellUB[i, j][0], mu[0], ss[0] ** 0.5
                    ) * self.get_D(self.cellLB[i, j][1], self.cellUB[i, j][1], mu[1], ss[1] ** 0.5)

                    self.HVIDistinCell[i, j] = rst * ProInCell
                elif method == "withoutTaylor":
                    rst = quad(
                        self.method_without_taylor,
                        L1 * L2,
                        a_,
                        args=(muP, ss, [L1, L2], [U1, U2]),
                        epsabs=1e-30,
                        epsrel=1e-10,
                    )[0]

                    ProInCell = self.get_D(
                        self.cellLB[i, j][0], self.cellUB[i, j][0], mu[0], ss[0] ** 0.5
                    ) * self.get_D(self.cellLB[i, j][1], self.cellUB[i, j][1], mu[1], ss[1] ** 0.5)
                    # ProInCell = 1
                    self.HVIDistinCell[i, j] = rst * ProInCell
                    # self.HVIDistinCell[i, j] = self.method_without_taylor(
                    #     muP, ss, [L1, L2], [U1, U2], self.pInCell[i, j]
                    # )

                    # pvals = np.linspace(L1 * L2, self.pInCell[i, j], 500)
                    # delta = (self.pInCell[i, j] - L1 * L2) / 500
                    # hist_pdf = [
                    #     self.method_without_taylor(muP, ss, [L1, L2], [U1, U2], p) for p in pvals
                    # ]

                    # ProInCell = self.get_D(
                    #     self.cellLB[i, j][0], self.cellUB[i, j][0], mu[0], ss[0] ** 0.5
                    # ) * self.get_D(self.cellLB[i, j][1], self.cellUB[i, j][1], mu[1], ss[1] ** 0.5)

                    # print('ProInCell is %f', ProInCell)
                    # if ProInCell > 1:
                    # print("error")
                    # self.HVIDistinCell[i, j] = np.sum(delta * np.array(hist_pdf)) * ProInCell

        return np.nansum(self.HVIDistinCell)

    def MC_approx(self, mean, variance, a, n_mc):
        mc_rst = np.zeros((n_mc, 1))
        evaluated_points = np.zeros((n_mc, 2))
        pf_list = self.pf.tolist()
        hvPrevious = hv.hypervolume(pf_list, self.r)
        evaluated_points[:, 0] = np.random.normal(
            loc=mean[0], scale=np.sqrt(variance[0]), size=n_mc
        )
        evaluated_points[:, 1] = np.random.normal(
            loc=mean[1], scale=np.sqrt(variance[1]), size=n_mc
        )

        n_pf = len(pf_list)
        countHVIlessA = 0
        for i in range(n_mc):
            added = [evaluated_points[i, 0], evaluated_points[i, 1]]
            pf_list = self.pf.tolist()
            pf_list.append(added)

            # ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = pf_list)
            # # if len(ndf[0]) <= n_pf:
            # if 1>11:
            #     mc_rst[i] = 0
            # else:
            hvi = hv.hypervolume(pf_list, self.r) - hvPrevious
            if hvi <= a and hvi > 0:
                countHVIlessA = countHVIlessA + 1
                mc_rst[i] = 1
        return countHVIlessA / n_mc

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:38:22 2021

@author: kaifengyang
"""
import matplotlib.pyplot as plt
import numpy as np

from DistributionHVI import DistributionHVI

r = np.array([6, 6])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])

mu = np.array([2, 3])  # mean of f1 and f2
ss = np.array([1, 1])  # variance, not std
a = 3  # input to CDF of HVI
nTaylor = 25
n_mc = int(1e5)

lbInf = -50

hvi_dist = DistributionHVI(pf, r, nTaylor)

# rst_Taylor = hvi_dist.compute_HVI_dist("withoutTaylor", mu, ss, a)
# mc_rst = hvi_dist.MC_approx(mu, ss, a, n_mc)
# rst_noTaylor = hvi_dist.compute_HVI_dist("Taylor", mu, ss, a)

# print("MC approximation is %f", mc_rst)
# print("Exact (without Taylor) result is %f", rst_Taylor)
# print("with Taylor result is %f", rst_noTaylor)

avals = np.linspace(0.1, 15, 10)

rst_all_ex = [hvi_dist.compute_HVI_dist("withoutTaylor", mu, ss, a) for a in avals]
rst_all_mc = [hvi_dist.MC_approx(mu, ss, a, n_mc) for a in avals]

plt.plot(avals, rst_all_ex, "r-")
plt.plot(avals, rst_all_mc, "bo")
plt.show()
breakpoint()

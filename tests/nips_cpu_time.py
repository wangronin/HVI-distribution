import sys

sys.path.insert(0, "./")

import pickle
import timeit

import numpy as np
from mobo.hv_improvement import HypervolumeImprovement

r = np.array([12, 12])
cpu_time = dict()

for n in [10, 20, 40, 80]:
    # linear Pareto front
    x = np.linspace(0, 6.5, 80)
    y = -0.5 * x + 5.5
    pf = np.c_[x, y]

    mu = np.array([2, 2])  # mean of f1 and f2
    sigma = np.array([4, 4])  # standard deviation
    avals = np.linspace(-20, 20, 80)
    hvi = HypervolumeImprovement(pf, r, mu, sigma)

    def exact():
        hvi.cdf(avals)

    def MC():
        hvi.cdf_monte_carlo(avals, n_sample=1e4, eval_sd=False)

    t = timeit.Timer("exact()", setup="from __main__ import exact")
    exact = t.repeat(repeat=100, number=1)

    t = timeit.Timer("MC()", setup="from __main__ import MC")
    MC = t.repeat(repeat=100, number=1)
    cpu_time[n] = np.atleast_2d([exact, MC])

with open("data.pkl", "wb") as f:
    pickle.dump(cpu_time, f)

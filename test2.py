import time

import matplotlib.pyplot as plt
import numpy as np

from mobo.hv_improvement import HypervolumeImprovement

r = np.array([15, 15])

# Pareto-front approximation set
pf = np.array(
    [
        [-0.03127299, 7.3938895],
        [0.0365997, 3.17223445],
        [0.18005575, 2.71554077],
        [0.30909125, 0.88363543],
        [0.73037724, 0.68023894],
        [0.75325258, 0.30688668],
    ]
)

mu = np.array([0.6530602, 0.79768873])  # mean of f1 and f2
sigma = np.array([0.00058944, 0.03164462])  # standard deviation

avals = 10 ** np.linspace(-3, -1, 100)
# avals = np.linspace(-10, 10, 100)
hvi = HypervolumeImprovement(pf, r, mu, sigma)

# v = hvi.cdf(0.00509414)
t = time.time_ns()
rst_all_ex = hvi.cdf(avals)
# rst_all_pdf = hvi.pdf(avals)
print(time.time_ns() - t)
# print(rst_all_ex)

t = time.time_ns()
rst_all_mc = hvi.cdf_monte_carlo(avals, n_sample=1e4, eval_sd=False)
print(time.time_ns() - t)
# print(rst_all_mc)

plt.plot(avals, rst_all_ex, "r-", alpha=0.6)
plt.plot(avals, rst_all_mc, "k--", mfc="none", alpha=0.6)
# plt.plot(avals, rst_all_pdf, "r--", mfc="none")
# plt.plot(avals, rst_all_mc + 3 * sd, "b-", alpha=0.5)
# plt.plot(avals, rst_all_mc - 3 * sd, "b-", mfc="none", alpha=0.5)
plt.xscale("log")
plt.yscale("linear")
plt.show()

breakpoint()

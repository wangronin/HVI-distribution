import matplotlib.pyplot as plt
import numpy as np
from mobo.hv_improvement import HypervolumeImprovement

r = np.array([6, 6])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])

mu = np.array([2, 3])  # mean of f1 and f2
sigma = np.array([1, 1])  # standard deviation

avals = 10 ** np.linspace(-10, np.log10(10), 100)
hvi = HypervolumeImprovement(pf, r, mu, sigma)

rst_all_ex = hvi.cdf(avals)
rst_all_ex.sort()
print(rst_all_ex)

rst_all_mc, sd = hvi.cdf_monte_carlo(avals, n_sample=1e5, eval_sd=True)
print(rst_all_mc)

plt.plot(avals, rst_all_ex, "ro", alpha=0.4)
plt.plot(avals, rst_all_mc, "k--", mfc="none")
plt.plot(avals, rst_all_mc + 3 * sd, "b-", alpha=0.5)
plt.plot(avals, rst_all_mc - 3 * sd, "b-", mfc="none", alpha=0.5)
plt.xscale("log")
plt.yscale("log")
plt.show()

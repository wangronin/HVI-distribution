import matplotlib.pyplot as plt
import numpy as np

from DistributionHVI import HypervolumeImprovement

r = np.array([6, 6])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])

mu = np.array([2, 3])  # mean of f1 and f2
sigma = np.array([5, 5])  # standard deviation

avals = 10 ** np.linspace(-10, np.log10(60), 30)
hvi = HypervolumeImprovement(pf, r, mu, sigma)

rst_all_ex = hvi.cdf(avals[::-1])
rst_all_ex.sort()
print(rst_all_ex)

rst_all_mc, sd = hvi.cdf_monte_carlo(avals, n_sample=5e5, eval_sd=True)
print(rst_all_mc)

plt.semilogx(avals, rst_all_ex, "r-")
plt.semilogx(avals, rst_all_mc, "bs", mfc="none")
plt.semilogx(avals, rst_all_mc + 1.96 * sd, "b-", alpha=0.5)
plt.semilogx(avals, rst_all_mc - 1.96 * sd, "b-", mfc="none", alpha=0.5)
plt.show()

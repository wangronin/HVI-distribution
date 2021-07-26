import matplotlib.pyplot as plt
import numpy as np

from DistributionHVI import HypervolumeImprovement

r = np.array([6, 6])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])

mu = np.array([2, 3])  # mean of f1 and f2
sigma = np.array([2, 2])  # standard deviation

avals = np.linspace(0.1, 15, 10)
hvi = HypervolumeImprovement(pf, r)

rst_all_ex = hvi.cdf(avals, mu, sigma)
rst_all_mc = hvi.cdf_monte_carlo(avals, mu, sigma)

print(rst_all_ex)
print(rst_all_mc)

plt.plot(avals, rst_all_ex, "r-")
plt.plot(avals, rst_all_mc, "bo")
plt.show()

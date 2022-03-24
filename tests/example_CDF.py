import sys

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import numpy as np
from mobo.hv_improvement import HypervolumeImprovement

plt.style.use("ggplot")

r = np.array([12, 12])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])
mu = np.array([3, 4])  # mean of f1 and f2
sigma = np.array([4, 4])  # standard deviation
hvi = HypervolumeImprovement(pf, r, mu, sigma)

avals = np.r_[np.linspace(-400, 0, 100), np.linspace(0.1, hvi.max_hvi, 100)]

res = hvi.cdf(avals)
res_mc, sd = hvi.cdf_monte_carlo(avals, n_sample=5e5, eval_sd=True)

print(1 - hvi.prob_in_ndom)
print(res[99])
print(res_mc[99])

plt.plot(avals, res, "r-")

plt.plot(avals, res_mc, "k-", marker="+")
# plt.plot(avals, res_mc + 3 * sd, "b-", alpha=0.5)
# plt.plot(avals, res_mc - 3 * sd, "b-", mfc="none", alpha=0.5)

plt.xscale("linear")
plt.yscale("log")
plt.xlabel("Hypervolume Improvement")
plt.ylabel("Probability")
plt.show()

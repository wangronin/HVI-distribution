import matplotlib.pyplot as plt
import numpy as np
from mobo.hv_improvement import HypervolumeImprovement

plt.style.use("ggplot")

r = np.array([12, 12])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])

mu = np.array([3, 4])  # mean of f1 and f2
sigma = np.array([2, 2])  # standard deviation

# avals = 10 ** np.linspace(-5, np.log10(10), 100)
avals = np.linspace(-20, 20, 80)
hvi = HypervolumeImprovement(pf, r, mu, sigma)
rst_all_ex = hvi.cdf(avals)
# rst_all_pdf = hvi.pdf(avals)
print(rst_all_ex)

rst_all_mc, sd = hvi.cdf_monte_carlo(avals, n_sample=2e3, eval_sd=True)
print(rst_all_mc)

plt.plot(
    avals,
    rst_all_ex,
    "r-",
)
plt.plot(avals, rst_all_mc, "k--")
# plt.plot(avals, rst_all_pdf, "r--", mfc="none")
plt.plot(avals, rst_all_mc + 5 * sd, "b-", alpha=0.5)
plt.plot(avals, rst_all_mc - 5 * sd, "b-", mfc="none", alpha=0.5)
plt.xscale("linear")
plt.yscale("linear")
plt.xlabel("Hypervolume Improvement")
plt.ylabel("Probability")
plt.show()

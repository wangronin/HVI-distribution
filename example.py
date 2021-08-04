import matplotlib.pyplot as plt
import numpy as np

from hv_improvement import HypervolumeImprovement

r = np.array([6, 6])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])

mu = np.array([0, 0])  # mean of f1 and f2
sigma = np.array([0.3, 0.3])  # standard deviation

avals = 10 ** np.linspace(1.5, np.log10(50), 50)
hvi = HypervolumeImprovement(pf, r, mu, sigma)

rst_all_ex = hvi.cdf(avals)
# pdf_ex = hvi.pdf(avals, taylor_expansion=True, taylor_order=60)
# pdf_ex2 = hvi.pdf(1e-10, taylor_expansion=False)
# rst_all_ex.sort()
# print(pdf_ex2)
print(rst_all_ex)

rst_all_mc, sd = hvi.cdf_monte_carlo(avals, n_sample=1e5, eval_sd=True)
print(rst_all_mc)

# print(avals)
# print(rst_all_ex - rst_all_mc)

# plt.loglog(avals, np.abs(pdf_ex - pdf_ex2) / pdf_ex2, color="r", ls="-", marker="o", mfc="none")
plt.loglog(avals, rst_all_ex, color="r", ls="-", marker="o", mfc="none")
plt.loglog(avals, rst_all_mc, color="b", ls="--", marker="s", mfc="none")
# plt.loglog(avals, np.abs(rst_all_ex - rst_all_mc), color="r", ls="-", marker="o", mfc="none")
plt.loglog(avals, rst_all_mc + 3 * sd, "b-", alpha=0.5)
plt.loglog(avals, rst_all_mc - 3 * sd, "b-", mfc="none", alpha=0.5)
plt.show()

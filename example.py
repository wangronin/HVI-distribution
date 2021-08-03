import matplotlib.pyplot as plt
import numpy as np

from hv_improvement import HypervolumeImprovement

r = np.array([6, 6])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])

mu = np.array([2, 3])  # mean of f1 and f2
sigma = np.array([1, 1])  # standard deviation

avals = 10 ** np.linspace(-1, np.log10(50), 50)
hvi = HypervolumeImprovement(pf, r, mu, sigma)

rst_all_ex = hvi.cdf(avals, taylor_expansion=False, taylor_order=10)
# pdf_ex = hvi.pdf(avals, taylor_expansion=True, taylor_order=10)
# pdf_ex2 = hvi.pdf(avals, taylor_expansion=False)
# rst_all_ex.sort()
print(rst_all_ex)

rst_all_mc = hvi.cdf_monte_carlo(avals, n_sample=1e5, eval_sd=False)
print(rst_all_mc)

# print(avals)
# print(rst_all_ex - rst_all_mc)

# plt.loglog(avals, np.abs(pdf_ex - pdf_ex2) / pdf_ex2, color="r", ls="-", marker="o", mfc="none")
# plt.loglog(avals, rst_all_ex, color="r", ls="-", marker="o", mfc="none")
# plt.loglog(avals, rst_all_mc, color="b", ls="--", marker="s", mfc="none")
plt.loglog(avals, np.abs(rst_all_ex - rst_all_mc), color="r", ls="-", marker="o", mfc="none")
# plt.semilogx(avals, rst_all_mc + 3 * sd, "b-", alpha=0.5)
# plt.semilogx(avals, rst_all_mc - 3 * sd, "b-", mfc="none", alpha=0.5)
plt.show()

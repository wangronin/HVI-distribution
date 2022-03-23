import sys

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import numpy as np
from mobo.hv_improvement import HypervolumeImprovement

plt.style.use("ggplot")

r = np.array([12, 12])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])
mu = np.array([2, 2])  # mean of f1 and f2
sigma = np.array([4, 4])  # standard deviation
hvi = HypervolumeImprovement(pf, r, mu, sigma)

avals = np.linspace(-50, 50, 100)

res = hvi.cdf(avals)
res_pdf = hvi.pdf(avals)
res_mc1, sd = hvi.cdf_monte_carlo(avals, n_sample=100, eval_sd=True)
res_mc2, sd = hvi.cdf_monte_carlo(avals, n_sample=500, eval_sd=True)
res_mc3, sd = hvi.cdf_monte_carlo(avals, n_sample=2500, eval_sd=True)

line = []
line.append(plt.plot(avals, res, ls="-", color="r")[0])
# line.append(plt.plot(avals, res_pdf, ls="-", color="green", alpha=0.7)[0])
line.append(plt.plot(avals, res_mc1, ls="dotted", color="k", alpha=0.7)[0])
line.append(plt.plot(avals, res_mc2, ls="-.", color="k", alpha=0.7)[0])
line.append(plt.plot(avals, res_mc3, ls="--", color="k", alpha=0.7)[0])
line.append(plt.fill_between(avals, res_mc3 - 2 * sd, res_mc3 + 2 * sd, alpha=0.3, color="cyan"))


plt.legend(line, ["exact CDF", "MC-100", "MC-500", "MC-2500", "95% CI of MC-2500"])

plt.xscale("linear")
plt.yscale("linear")
plt.xlabel(r"$\Delta(\mathbf{y})$")
plt.annotate(r"$y_1\sim \mathcal{N}(2, 16), y_2 \sim \mathcal{N}(2, 16)$", xy=(11, 0.03))
plt.ylabel("Probability")
plt.savefig("example.pdf")
plt.show()

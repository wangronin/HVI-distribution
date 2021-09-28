import matplotlib.pyplot as plt
import numpy as np

from hv_improvement import HypervolumeImprovement

r = np.array([6, 6])

# Pareto-front approximation set
# pf = np.array([[3, 4], [1, 5], [5, 1], [4,2]])

pf = np.array(
[[0.0000000000,1.0000000000],
[0.1618789945,0.9737951912],
[0.2428184950,0.9410391785],
[0.3102680810,0.9037337179],
[0.3703861900,0.8628140703],
[0.4256253904,0.8188430271],
[0.4772800193,0.7722037832],
[0.5261394407,0.7231772889],
[0.5727302472,0.6719800640],
[0.6174260050,0.6187851283],
[0.6605040066,0.5637344573],
[0.7021772485,0.5069471117],
[0.7426138660,0.4485246460],
[0.7819495726,0.3885548659],
[0.8202959154,0.3271146112],
[0.8577459707,0.2642718498],
[0.8943784801,0.2000871342],
[0.9302607811,0.1346148791],
[0.9654510582,0.0679042543],
[1.0000000000,0.0000000000]]) # ZDT2

mu = np.array([0.97, 0.05])  # mean of f1 and f2
sigma = np.array([0.3, 0.3])*0.1  # standard deviation


# r = np.array([1.99012962, 9.52036267])

# # Pareto-front approximation set
# pf = np.array([[0.01829378, 2.49050436],
#                [0.63943971, 1.62811579]])

# mu = np.array([0.01829383, 2.49050436])  # mean of f1 and f2
# sigma = np.array([5.35550221e-05, 1e-5])  # standard deviation

avals = 10 ** np.linspace(-2, np.log10(50), 50)
hvi = HypervolumeImprovement(pf, r, mu, sigma)

rst_all_ex = hvi.cdf(avals)
# pdf_ex = hvi.pdf(avals, taylor_expansion=True, taylor_order=60)
# pdf_ex2 = hvi.pdf(1, taylor_expansion=False)
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

plt.scatter(pf[:,0], pf[:,1], color="r")
plt.scatter(mu[0], mu[1], color="b")
plt.show()

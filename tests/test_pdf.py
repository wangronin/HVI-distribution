import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

from mobo.special import D, pdf_product_of_truncated_gaussian


def rcond_norm(N, loc, scale, L=1, U=2):
    return truncnorm.rvs((L - loc) / scale, (U - loc) / scale, loc, scale, int(N))


# reference point
r = np.array([6, 6])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])

mu = np.array([2, 3])  # mean of f1 and f2
ss = np.array([0.5, 1])  # std.
L1, L2, U1, U2 = 0.5, 0.3, 5, 7

pvals = np.clip(10 ** np.linspace(np.log10(L1 * L2), np.log10(U1 * U2), 200), L1 * L2, U1 * U2)
lower, upper = np.array([L1, L2]), np.array([U1, U2])
normalizer = np.prod([D(lower[k], upper[k], mu[k], ss[k]) for k in range(len(ss))]) * 2 * np.pi * np.prod(ss)
rst_new = [pdf_product_of_truncated_gaussian(p, mu, ss, lower, upper, normalizer) for p in pvals]

x = rcond_norm(1e7, mu[0], ss[0], L1, U1)
y = rcond_norm(1e7, mu[1], ss[1], L2, U2)

a = x * y
a = a[~np.isinf(a)]
n, bins, patches = plt.hist(a, 100, density=True, facecolor="b", alpha=0.4)
plt.plot(pvals, np.array(rst_new), "r-")
plt.show()

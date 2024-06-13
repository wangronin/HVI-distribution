import sys

import numpy as np

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sympy import symbols

from hvi.level_curve import HypervolumeLevelCurve

plt.style.use("ggplot")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 14
rcParams["ytick.labelsize"] = 14
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

pareto_front = np.array([[1, 4], [2, 3], [3, 2], [4, 1]])

lc = HypervolumeLevelCurve(pareto_front=pareto_front, reference_point=[0, 0])
curves = lc.compute(0.1)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect("equal")
plt.subplots_adjust(top=0.96, bottom=0.12, right=0.99, left=0.06, hspace=0.11, wspace=0.25)

ax.plot(pareto_front[:, 0], pareto_front[:, 1], "k.", ms=10)
ax.hlines(pareto_front[:, 1], xmin=0, xmax=7, linestyles="dashed", colors="k", alpha=0.3)
ax.vlines(pareto_front[:, 0], ymin=0, ymax=7, linestyles="dashed", colors="k", alpha=0.3)

x = symbols("x")
for x0, x1, expr in curves:
    print(f"{expr}, x \in [{x0:.3f}, {x1:.3f}]")
    x0 = max(1e-1, x0)
    x1 = min(x1, 7)
    xx = np.linspace(x0, x1, num=50)
    y = [expr.subs(x, _) for _ in xx]
    ax.plot(xx, y, "r-")

ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
plt.show()

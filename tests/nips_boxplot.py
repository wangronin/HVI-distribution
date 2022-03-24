import pickle

import matplotlib.pyplot as plt
from pylab import boxplot, setp

plt.style.use("ggplot")

# function for setting the colors of the box plots pairs
def setBoxColors(bp, color):
    setp(bp["boxes"], color=color)
    setp(bp["caps"], color=color)
    setp(bp["whiskers"], color=color)
    setp(bp["fliers"], color=color)
    setp(bp["medians"], color=color)


with open("data.pkl", "rb") as f:
    data = pickle.load(f)

x = data.keys()
fig = plt.figure()

for i, k in enumerate(x):
    X = data[k]
    bp1 = boxplot([data[0, :]], positions=[i * 3 + 1], widths=0.6, sym="+")
    bp2 = boxplot([data[1, :]], positions=[i * 3 + 2], widths=0.6, sym="+")
    setBoxColors(bp1, "red")
    setBoxColors(bp2, "blue")


plt.xticks([1.5 + i * 3 for i in range(4)])
ax = plt.gca()
ax.set_xticklabels(list(map(str, x)))

plt.yscale("log")
plt.annotate(r"$y_1\sim \mathcal{N}(2, 16), y_2 \sim \mathcal{N}(2, 16)$", xy=(6, 0.02))
plt.xlabel(r"$n=|\mathcal{P}|$")
plt.ylabel("CPU time (sec)")

(hB,) = plt.plot([1, 1], "r-")
(hR,) = plt.plot([1, 1], "b-")
plt.legend((hB, hR), ["Exact CDF", "MC-1e4"])
hB.set_visible(False)
hR.set_visible(False)
plt.savefig("CPU_time.pdf")
plt.show()

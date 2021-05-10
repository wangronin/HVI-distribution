import math

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

import hypervolume as hv
import matplotlib.pyplot as plt



def integral1(s, a, b, p):
    return quad(
        lambda x: x ** (s - 1) * np.exp(-0.5 * x ** 2 / a - 0.5 * p ** 2 * x ** (-2) / b),
        0,
        1,
        limit=100,
        epsabs=1e-20,
        epsrel=1e-10,
    )[0]


def integrand1(x, a, b, s, p):
    return x ** (s - 1) * np.exp(-0.5 * x ** 2 / a - 0.5 * p ** 2 * x ** (-2) / b)

# Eq.(4)
def original(x, mean, variance, p):
    return np.exp(-0.5 * ( (x-mean[0]) ** 2 / variance[0] + (p / x - mean[1]) ** 2 / variance[1]) ) / x


# transformed form
def integrand2(x, a, b, s, p):
    return (
        0.5
        * (p * np.sqrt(a / b)) ** (s / 2)
        * np.exp(-p / np.sqrt(a * b) * np.cosh(x) + x * s / 2)
    )





def computeTaylorSeries(nTaylor, mean, variance, p):
    #  mu: muHat[i,j]
    #  s: ssHat[i,j]
    # faInCell = np.array(
    #     [[("nan") for j in range(0, nTaylor)] for i in range(0, nTaylor)], dtype=float
    # )

    faInCell = np.zeros((nTaylor, nTaylor + 1))
    TMP = np.zeros((nTaylor, nTaylor + 1))
    error = np.zeros((nTaylor, nTaylor + 1))

    for n in range(nTaylor):
        for m in range(n + 1):
            # hao
            tmp = quad(
                integrand1,
                p,
                1.0,
                args=(variance[0], variance[1], 2 * m - n, p),

                # points=(0.005, 0.03),
                # weight= 'cauchy',
                # wvar=0

                # points=(0.003, 0.03),
                # limit=1000,
                # epsabs=1e-30,
                # epsrel=1e-10,
                # weight="cauchy",
                # wvar=0,

            )
            error[n, m] = tmp[1]
            TMP[n, m] = tmp[0]

            # if n == 10 and m == 1:
            #     breakpoint()

            faInCell[n, m] = (
                p ** (n - m)
                / math.factorial(n)
                * (math.factorial(n) / math.factorial(m) / math.factorial(n - m))
                * (mean[0] / variance[0]) ** m
                * (mean[1] / variance[1]) ** (n - m)
                # * integral1(2 * m - n, variance[0], variance[1], p)
                * tmp[0]
            )

    faInCell[np.isnan(faInCell)] = 0
    tmp = np.cumsum(np.sum(faInCell, axis=0))
    breakpoint()
    return np.nansum(faInCell)


if 11 < 2:
    res = list(
        map(
            lambda p: computeTaylorSeries(
                10, (-0.00798403, -0.00598802), (7.96809574e-06, 1.19521436e-05), p
            ),
            10 ** np.linspace(-20, 0, 50),
            # [0.00019306977288832455],
            # 10 ** np.linspace(-20, -1, 50),
            # [0.00019306977288832455],

        )
    )

 

    plt.semilogx(10 ** np.linspace(-20, -1, 50), res)
    plt.show()

    # breakpoint()


def affineTransform(x, lb, ub):
    return (x - lb) / (ub - lb)


def hviPrime(pf, r, l, u):
    # transformed u
    uT = [1, 1]
    # transformed reference point
    rT = np.array([affineTransform(r[0], l[0], u[0]), affineTransform(r[1], l[1], u[1])])
    # transformed pf in the first coordinate
    pfT1 = [affineTransform(pf[k, 0], l[0], u[0]) for k in range(0, num - 1)]
    # transformed pf in the second coordinate
    pfT2 = [affineTransform(pf[k, 1], l[1], u[1]) for k in range(0, num - 1)]
    # transformed pf
    pfT = np.transpose(np.vstack([pfT1, pfT2])).tolist()
    # HV value for origianl PF in the transposed space
    hvPrevious = hv.hypervolume(pfT, rT)
    pfT.append(uT)
    # HV values of adding u in PF in the transposed sapce
    hvAdded = hv.hypervolume(pfT, rT)

    return hvAdded - hvPrevious


def MC_approximate (mean, variance, pf, r, n_mc, a, cellLB, cellUB):
    mc_rst = np.zeros((n_mc,1))
    evaluated_points = np.zeros((n_mc,2))
    pf_list = pf.tolist()
    hvPrevious = hv.hypervolume(pf_list, r)
    evaluated_points[:,0] = np.random.normal(loc=mean[0], scale=np.sqrt(variance[0]), size=n_mc)
    evaluated_points[:,1] = np.random.normal(loc=mean[1], scale=np.sqrt(variance[1]), size=n_mc)
    
    count = 0
    for i in range(n_mc):
        if cellLB[0] <= evaluated_points[i,0] <= cellUB[0] and cellLB[1] <= evaluated_points[i,1] <= cellUB[1]:
            added = [evaluated_points[i,0], evaluated_points[i,1]]
            pf_list = pf.tolist()
            pf_list.append(added)
            hvi = hv.hypervolume(pf_list, r) - hvPrevious
            if hvi <= a:
                mc_rst[i] = hvi
                count = count + 1
            
    return np.sum(mc_rst)/n_mc, count/n_mc
        

def compute_f (Dset, c, muHat, ssHat, pInCell, nTaylor):
    productInDset = Dset[0] * Dset[1]
    rst = (
            c ** (-1) 
            * (2 * np.pi * np.sqrt(ssHat[0] * ssHat[1]) * productInDset) ** (-1)
            * np.exp( -0.5 * (muHat[0] ** 2 / ssHat[0] + muHat[1] ** 2 / ssHat[1]))
            * computeTaylorSeries(nTaylor, muHat, ssHat, pInCell)
            )  
    return rst   

def compute_f_original (Dset, c, muHat, ssHat, pInCell, nTaylor, productInDset):
    
    rst = (
          c ** (-1) 
          * (2 * np.pi * np.sqrt(ssHat[0] * ssHat[1]) * productInDset) ** (-1)
          * quad(original, 0, 1, args=(muHat, ssHat, pInCell))[0]
          )
            
    return rst   
    
    
    
    
    

# reference point
r = np.array([6, 6])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])

mu = np.array([2, 3])  # mean of f1 and f2
ss = np.array([1, 1])  # variance, not std
m = 2
a = 1  # input to CDF of HVI
nTaylor = 50
lbInf = -500


## ---- grid the non-dominated sapce -------
# Pareto-front approximation set with reference point
# pfr = np.vstack((pf,[-math.inf, r[1]], [r[0], -math.inf])) ## BUG: lower bound can be -infi
pfr = np.vstack((pf, [lbInf, r[1]], [r[0], lbInf]))
pfr_test = np.transpose(pfr)
# sorted pfr
q = np.transpose(
    pfr[
        pfr[:, 0].argsort(),
    ]
)


# number of PF + 1
num = q.shape[1] - 1
# upperbound of each cell
cellUB = np.array([[("nan", "nan") for j in range(0, num)] for i in range(0, num)], dtype=float)
# lowerbound of each cell
cellLB = np.array([[("nan", "nan") for j in range(0, num)] for i in range(0, num)], dtype=float)

for i in range(0, num):
    for j in range(0, num - i):
        cellLB[i, j] = [q[0, i], q[1, num - j]]
        cellUB[i, j] = [q[0, i + 1], q[1, num - 1 - j]]


# hvpf = hv.hypervolume(pf.tolist(),r)


# initialized HV for each cell
HVinCell = np.array([["nan" for j in range(0, num)] for i in range(0, num)], dtype=float)
# initialized c_{i,j} for each cell
c = np.array([["nan" for j in range(0, num)] for i in range(0, num)], dtype=float)
# initialized D:
Dset = np.array([[("nan",) * m for j in range(0, num)] for i in range(0, num)], dtype=float)

alpha_012 = np.array([[("nan",) * 3 for j in range(0, num)] for i in range(0, num)], dtype=float)
muHat = np.array([[("nan",) * m for j in range(0, num)] for i in range(0, num)], dtype=float)
ssHat = np.array([[("nan",) * m for j in range(0, num)] for i in range(0, num)], dtype=float)
# lastTerm = np.array([[(0) for j in range(0,num)] for i in range(0,num)], dtype=float)
pInCell = np.array([[(0) for j in range(0, num)] for i in range(0, num)], dtype=float)


# transformed HVI
tranformedHVIinCell = np.array([[("nan",) * 1 for j in range(0, num)] for i in range(0, num)], dtype=float)

q1_prime = np.array([[("nan",) * 1 for j in range(0, num)] for i in range(0, num)], dtype=float)
q2_prime = np.array([[("nan",) * 1 for j in range(0, num)] for i in range(0, num)], dtype=float)
alpha = np.array([[("nan",) * 1 for j in range(0, num)] for i in range(0, num)], dtype=float)
alpha2 = np.array([[("nan",) * 1 for j in range(0, num)] for i in range(0, num)], dtype=float)
# n+1 = num
for i in range(0, num):
    for j in range(0, num - i):
        
        # tranposed alpha in the paper,
        # tranformedHVIinCell[i, j] = hviPrime(pf, r, cellLB[i, j], cellUB[i, j])
        
        # # HVI(u')
        # alpha_012[i, j][0] = tranformedHVIinCell[i, j]
        
        # # q_1^(n+1-i)' - 1
        # alpha_012[i, j][1] = (q[0, num - j] - cellLB[i, j][0]) / (
        #     cellUB[i, j][0] - cellLB[i, j][0]
        # ) - 1
         
        # # q_2^(i)' - 1
        # alpha_012[i, j][2] = (q[1, i] - cellLB[i, j][1]) / (cellUB[i, j][1] - cellLB[i, j][1]) - 1
        
        # muHat[i, j][0] = (
        #     (mu[0] - cellLB[i, j][0]) / (cellUB[i, j][0] - cellLB[i, j][0])
        #     - alpha_012[i, j][1]
        #     - 1
        # )
        
        # muHat[i, j][1] = (
        #     (mu[1] - cellLB[i, j][1]) / (cellUB[i, j][1] - cellLB[i, j][1])
        #     - alpha_012[i, j][2]
        #     - 1
        # )
        # alpha[i,j] = alpha_012[i, j][0] - alpha_012[i, j][1] * alpha_012[i, j][2]
        
        
        
        tranformedHVIinCell[i, j] = hviPrime(pf, r, cellLB[i, j], cellUB[i, j])
        q1_prime[i,j] = affineTransform(q[0, num - j], cellLB[i, j][0], cellUB[i, j][0])
        q2_prime[i,j] = affineTransform(q[1, i], cellLB[i, j][1], cellUB[i, j][1])
        muHat[i, j][0] = affineTransform(mu[0], cellLB[i, j][0], cellUB[i, j][0]) - q1_prime[i,j]
        muHat[i, j][1] = affineTransform(mu[1], cellLB[i, j][1], cellUB[i, j][1]) - q2_prime[i,j]
        alpha[i, j] = tranformedHVIinCell[i, j] - (q1_prime[i,j] - 1) * (q2_prime[i,j] - 1)
        
        ssHat[i,j] = [
            np.sqrt(ss[k]) / (cellUB[i, j][k] - cellLB[i, j][k]) ** 2 for k in range(0, m)
        ]

        Dset[i,j] = [
            norm.cdf(1, loc=muHat[i, j][k], scale=np.sqrt(ssHat[i, j][k])) - 
            norm.cdf(0, loc=muHat[i, j][k], scale=np.sqrt(ssHat[i, j][k]))
            for k in range(0, m)
        ]

        c[i,j] = (cellUB[i, j][0] - cellLB[i, j][0]) * (cellUB[i, j][1] - cellLB[i, j][1])

        pInCell[i,j] = a / c[i,j] - alpha[i,j] 
       
        
        # TODO: uncomment for the final version 
        # productInDset = Dset[i, j][0] * Dset[i, j][1]
        # HVinCell[i, j] = (
        #     (1 / c[i, j])
        #     / (2 * np.pi * np.sqrt(ssHat[i, j][0] * ssHat[i, j][1]) * productInDset)
        #     * np.exp(
        #         -0.5
        #         * (muHat[i, j][0] ** 2 / ssHat[i, j][0] + muHat[i, j][1] ** 2 / ssHat[i, j][1])
        #     )
        #     * computeTaylorSeries(nTaylor, muHat[i, j], ssHat[i, j], pInCell[i, j])
        # )  # Taylor parts, 
        # HVinCell[i, j] = compute_f (Dset[i,j], c[i,j], muHat[i,j], ssHat[i,j], pInCell[i,j], nTaylor)


# print("F(A(y)) < %f is %f" % (a, np.nansum(HVinCell)))




# plot Monte Carlo method approximation with explicit formula results 
i = 1
j = 1

# Explict results
rst_formula = compute_f(Dset[i,j], c[i,j], muHat[i,j], ssHat[i,j], pInCell[i,j], nTaylor)
# eq. (4) 
productInDset = Dset[i, j][0] * Dset[i, j][1]
rst_original = compute_f_original(Dset[i,j], c[i,j], muHat[i,j], ssHat[i,j], pInCell[i,j], nTaylor, productInDset)


n_mc = 10000
rst_mc, mc_pro = MC_approximate(mu, ss, pf, r, n_mc, a, cellLB[i,j], cellUB[i,j])


tmp = np.linspace(10, 60, 6).astype(int)
if 11 < 2:
    res = list(
        map(
            lambda n: compute_f(
                Dset[i,j], c[i,j], muHat[i,j], ssHat[i,j], pInCell[i,j], n
            ),
            tmp,
            # [0.00019306977288832455],
        )
    )

 

    plt.plot(tmp, res)

print("Result by using Eq.(4) is %f" %rst_original)
print("Explicit formula result is %f" %rst_formula)
print("MC result is (%f,%f)" %(rst_mc, mc_pro))


# #
# 1. pInCell[i,j] may be bigger than 1 if a is very large or variances are large
# 2. mc_rst and rst_formula are not consistent. 



        





# # This is the test of two functions 
# print("Test of two functions")
# p = 0.1
# ss1 = np.exp(2)
# ss2 = np.pi * 2
# print(quad(integrand1, 0, 1, args=(ss1, ss2, nTaylor, p)))
# print(
#     quad(
#         integrand2,
#         -500,
#         np.log(np.sqrt(ss2)) - np.log(p * np.sqrt(ss1)),
#         args=(ss1, ss2, nTaylor, p),
#     )
# )

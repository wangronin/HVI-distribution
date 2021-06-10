#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 06:58:25 2021

@author: kaifengyang
"""
import math

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

import hypervolume as hv
import matplotlib.pyplot as plt

# Eq.(4)
def original(x, mean, variance, p):
    return np.exp(-0.5 * ( (x-mean[0]) ** 2 / variance[0] + (p / x - mean[1]) ** 2 / variance[1]) ) / x



def hvi(pf, r, point_add):
    pfs = pf.tolist()
    # HV value for origianl PF in the transposed space
    hvPrevious = hv.hypervolume(pfs, r)
    pf_temp = np.vstack((pfs, point_add)).tolist()
    # HV values of adding u in PF in the transposed sapce
    hvAdded = hv.hypervolume(pf_temp, r)

    return hvAdded - hvPrevious


def MC_approximate (mean, variance, pf, r, n_mc, a, cellLB, cellUB):
    mc_rst = np.zeros((n_mc,1))
    evaluated_points = np.zeros((n_mc,2))
    pf_list = pf.tolist()
    hvPrevious = hv.hypervolume(pf_list, r)
    evaluated_points[:,0] = np.random.normal(loc=mean[0], scale=np.sqrt(variance[0]), size=n_mc)
    evaluated_points[:,1] = np.random.normal(loc=mean[1], scale=np.sqrt(variance[1]), size=n_mc)
    
    
    countYinCell = 0
    countHVIlessA = 0
    for i in range(n_mc):
        if cellLB[0] < evaluated_points[i,0] <= cellUB[0] and cellLB[1] < evaluated_points[i,1] <= cellUB[1]:
            countYinCell = countYinCell + 1
            added = [evaluated_points[i,0], evaluated_points[i,1]]
            pf_list = pf.tolist()
            pf_list.append(added)
            hvi = hv.hypervolume(pf_list, r) - hvPrevious
            
            if hvi <= a:
                countHVIlessA = countHVIlessA + 1
                mc_rst[i] = hvi
            else:
                print('')
                
    plt.plot(evaluated_points[:,0] , evaluated_points[:,1] , 'o', color='black');
            
    return np.sum(mc_rst)/n_mc, countHVIlessA/n_mc
        



    


def func_v3(mean, variance, truncatedLB, truncatedUB, p):
    L1 = truncatedLB[0]
    L2 = truncatedLB[1]
    U1 = truncatedUB[0]
    U2 = truncatedUB[1]
    
    if L1 * U2 > U1 * L2: # swap y_1' and y_2'
        L1 = truncatedLB[1]
        L2 = truncatedLB[0]
        U1 = truncatedUB[1]
        U2 = truncatedUB[0]
        
    if L1 * L2 < p and p <= L1 * U2:
        alpha = L1
        belta = p / L1
    elif L1 * U2 < p and p <= U1 * L2:
        alpha = p / U2
        belta = p / L2
    elif U1 * L2 < p and p <= U1 * U2:
        alpha = p / U2 
        belta = U1
    else:
        print('error in lb and ub')
    
    
    res = quad(
            original,
            alpha,
            belta,
            args=(mean, variance, p),
            # limit=1000,
            # epsabs=1e-30,
            # epsrel=1e-10
        )[0]
    
    return res
    
    
    
    

# reference point
r = np.array([6, 6])

# Pareto-front approximation set
pf = np.array([[3, 4], [1, 5], [5, 1]])

mu = np.array([2, 3])  # mean of f1 and f2
ss = np.array([1, 1])  # variance, not std
m = 2
a = 3  # input to CDF of HVI
nTaylor = 50
lbInf = -50


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


# n+1 = num


# For version 3
# cell gamma: 
gamma = np.array([[(0) for j in range(0, num)] for i in range(0, num)], dtype=float)
truncatedLB = np.array([[("nan", "nan") for j in range(0, num)] for i in range(0, num)], dtype=float)
truncatedUB = np.array([[("nan", "nan") for j in range(0, num)] for i in range(0, num)], dtype=float)
# initialized D:
Dset = np.array([[("nan",) * m for j in range(0, num)] for i in range(0, num)], dtype=float)

muPrime = np.array([[("nan",) * m for j in range(0, num)] for i in range(0, num)], dtype=float)
# initialized HV for each cell
HVinCell = np.array([["nan" for j in range(0, num)] for i in range(0, num)], dtype=float)

pInCell = np.array([[(0) for j in range(0, num)] for i in range(0, num)], dtype=float)
aInCell = np.array([[(0) for j in range(0, num)] for i in range(0, num)], dtype=float)



for i in range(0, num):
    for j in range(0, num - i):
        truncatedLB[i,j] = [q[0, num-j] - cellUB[i,j][0], 
                            q[1, i] - cellUB[i,j][1]
                            ]
        
        
        truncatedUB[i,j] = [q[0, num-j] - cellLB[i,j][0], 
                            q[1, i] - cellLB[i,j][1]
                            ]
        
        muPrime[i, j][0] = q[0,num-j] - mu[0]
        muPrime[i, j][1] = q[1,i] - mu[1]
        
    
        
        Dset[i,j] = [
            norm.cdf(truncatedUB[i,j][k], loc=muPrime[i, j][k], scale=np.sqrt(ss[k])) - 
            norm.cdf(truncatedLB[i,j][k], loc=muPrime[i, j][k], scale=np.sqrt(ss[k]))
            for k in range(0, m)
        ]
        
        
        gamma[i,j] = hvi(pf, r, cellUB[i,j]) - truncatedLB[i,j][0] * truncatedLB[i,j][1]
        
        area = (cellUB[i,j][0] - cellLB[i,j][0]) * (cellUB[i,j][1] - cellLB[i,j][1])
        if a > area :
            aInCell[i,j] = area
        else:
            aInCell[i,j] = a
        
        # aInCell[i,j] = a
            
        pInCell[i,j] = aInCell[i,j] - gamma[i,j]
        
        # HVinCell[i,j] = func_v3(mu, ss, truncatedLB[i,j], truncatedUB[i,j], pInCell[i,j]) / (
        #     2 * np.pi * np.sqrt(ss[0] * ss[1]) * Dset[i,j][0] * Dset[i,j][1]
        #     )
        
        
      



# # plot Monte Carlo method approximation with explicit formula results 
i = 1
j = 1

n_mc = 10000
rst_mc, mc_pro = MC_approximate(mu, ss, pf, r, n_mc, a, cellLB[i,j], cellUB[i,j])
rst_new = func_v3(mu, ss, truncatedLB[i,j], truncatedUB[i,j], pInCell[i,j]) / (
            2 * np.pi * np.sqrt(ss[0] * ss[1]) * Dset[i,j][0] * Dset[i,j][1]
            )


print("Result by using Eq.(4) of V3 is %f" %rst_new)
print("MC result is (%f,%f)" %(rst_mc, mc_pro))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 23:04:28 2021

@author: kaifengyang
"""
import matplotlib.pyplot as plt

import gpflow
import gpflowopt
import numpy as np


# Objective
def vlmop2(x):
    transl = 1 / np.sqrt(2)
    part1 = (x[:, [0]] - transl) ** 2 + (x[:, [1]] - transl) ** 2
    part2 = (x[:, [0]] + transl) ** 2 + (x[:, [1]] + transl) ** 2
    y1 = 1 - np.exp(-1 * part1)
    y2 = 1 - np.exp(-1 * part2)
    return np.hstack((y1, y2))

# Setup input domain
domain = gpflowopt.domain.ContinuousParameter('x1', -2, 2) + \
         gpflowopt.domain.ContinuousParameter('x2', -2, 2)

# Plot
def plotfx():
    X = gpflowopt.design.FactorialDesign(101, domain).generate()
    Z = vlmop2(X)
    shape = (101, 101)

    axes = []
    plt.figure(figsize=(15, 5))
    for i in range(Z.shape[1]):
        axes = axes + [plt.subplot2grid((1, 2), (0, i))]

        axes[-1].contourf(X[:,0].reshape(shape), X[:,1].reshape(shape), Z[:,i].reshape(shape))
        axes[-1].set_title('Objective {}'.format(i+1))
        axes[-1].set_xlabel('x1')
        axes[-1].set_ylabel('x2')
        axes[-1].set_xlim([domain.lower[0], domain.upper[0]])
        axes[-1].set_ylim([domain.lower[1], domain.upper[1]])

    return axes

plotfx();


# Initial evaluations
design = gpflowopt.design.LatinHyperCube(11, domain)
X = design.generate()
Y = vlmop2(X)

# One model for each objective
objective_models = [gpflow.gpr.GPR(X.copy(), Y[:,[i]].copy(), gpflow.kernels.Matern52(2, ARD=True)) for i in range(Y.shape[1])]
for model in objective_models:
    model.likelihood.variance = 0.01

hvpoi = gpflowopt.acquisition.HVProbabilityOfImprovement(objective_models)


# First setup the optimization strategy for the acquisition function
# Combining MC step followed by L-BFGS-B
acquisition_opt = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(domain, 1000),
                                                   gpflowopt.optim.SciPyOptimizer(domain)])

# Then run the BayesianOptimizer for 20 iterations
optimizer = gpflowopt.BayesianOptimizer(domain, hvpoi, optimizer=acquisition_opt, verbose=True)
result = optimizer.optimize([vlmop2], n_iter=2)

print(result)
print(optimizer.acquisition.pareto.front.value)


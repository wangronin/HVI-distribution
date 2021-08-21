#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:42:44 2021

@author: kaifengyang
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

n = 400
n_var = 2
real_c = np.full((2, 2), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))
design = np.random.uniform(size=n * n_var).reshape(-1, 2)
test = np.random.uniform(size=n * n_var).reshape(-1, 2)
response = np.apply_along_axis(lambda x: np.sin(np.sum(x)), 1, design)
kernel = RBF(length_scale=(1, 1))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                               optimizer="fmin_l_bfgs_b").fit(design, response)
gpr.predict(test, return_std=True)
theta = gpr.kernel_.get_params()["length_scale"]
#theta = gpr.kernel_.theta
k_inv = gpr._K_inv
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:56:24 2021

@author: kaifengyang
"""
from pymoo.algorithms.so_cmaes import CMAES
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("sphere")

algorithm = GA()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

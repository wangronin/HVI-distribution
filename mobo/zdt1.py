#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 00:43:36 2021

@author: kaifengyang
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 23:57:53 2021

@author: kaifengyang
"""
from utils import expand, find_pareto_front, safe_divide

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
# from arguments import get_vis_args
# from utils import get_problem_dir, get_algo_names, defaultColors
from math import floor
import matplotlib.pyplot as plt



df = pd.read_csv('EvaluatedSamples6.csv')
pf_a = df[['f1','f2']].to_numpy()
pf = find_pareto_front(pf_a, return_index=False)


plt.scatter(pf[:,0], pf[:,1], color="r")
# plt.scatter(mu[0], mu[1], color="b")
plt.show()


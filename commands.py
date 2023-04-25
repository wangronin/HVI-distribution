#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 09:01:41 2021

@author: kaifengyang
"""
command: 
    python scripts/run.py --problem zdt1 zdt2 zdt3 zdt4 zdt6 --algo hvic-m4-es-e-dr --n-seed 15 --n-process 5 --n-var 5 --batch-size 1 --n-iter 170
    
    python visualization/visualize_hv_batch.py --problem zdt1 zdt2 zdt3 zdt4 zdt6 --algo ucb hvi-ucb --n-seed 15

## SWEVO
python scripts/run.py --problem zdt1 zdt2 zdt3 zdt4 zdt6 --algo dgemo tsemo usemo_ei moead_ego parego --n-seed 15 --n-process 5 --n-var 5 --batch-size 2 --n-iter 120
    
squeue -u wangh5 -h -t running -r | wc -l
sinfo

scp -r das51:/home/haowang/hvi/HVI-distribution_cprofiler_ci_das/result /Users/kaifengyang/Documents/GitHub/HVI-distribution/result_distribution_cprofiler_ci_das

scp -r alice:/home/wangh5/hvi /Users/kaifengyang/Documents/GitHub

scp -r /Users/kaifengyang/Documents/GitHub/hvi/HVI-distribution_alice alice:/home/wangh5/hvi/HVI-distribution_alice_neagtive

scp -r /Users/kaifengyang/Documents/GitHub/HVI-distribution alice:/home/wangh5/hvi/HVI-distribution_epoi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import exp, abs, log
from scipy.special import gamma, factorial
from utils import *

def ucb_strategy(lam, predY0, std_varY0, sample_Y0, ninvest = 10):
    rt_v, x_vec = [], []

    for i in range(len(predY0)):
        x_opt = np.zeros(len(predY0[0]))
        p = predY0[i]
#         sell_idx = np.argsort(p-lam*std_varY0[i])[:ninvest//2]
        buy_idx = np.argsort(-(p+lam*std_varY0[i]))[:ninvest]
        rr = 1/ninvest
        x_opt[buy_idx] = rr
#         tmp = rr*(sum(exp(sample_Y0[i, buy_idx]))+sum(exp(-sample_Y0[i, sell_idx])))
        tmp = rr*(sum(exp(sample_Y0[i, buy_idx])))
        rt_v.append(log(tmp))
        x_vec.append(x_opt)
        
    return rt_v, x_vec

def passive_strategy(predY0, std_var_Y0, sample_Y0, cov, eps, delta, cash=1.0):
    rt_v, x_vec = [], []
    x_ori = np.ones(len(predY0[0])) / len(predY0[0])
    for i in range(len(predY0)):
        r = exp(predY0[i])
        cov_dia = std_var_Y0[i] ** 2 * r * r
#          cov_current = np.zeros_like(cov)
        cov_current = cov.copy()
        for k in range(len(r)):
            cov_current[k, k] = cov_dia[k]
        x_opt = passive_solve(len(r), x_ori, r, cov_current, eps, delta, cash)
        if x_opt is None:
            print('No opt solution')
            x_opt = x_ori
        
#         print(x_opt)
#         print(sum(abs(x_opt-x_ori)))
        
        tmp = sum(x_opt * exp(sample_Y0[i])) / cash
        rt_v.append(log(tmp))
        x_vec.append(x_opt)
        
        x_ori = x_opt
    return rt_v, x_vec

def opt_strategy(predY0, std_var_Y0, sample_Y0, cov, gamma, cash=1.0):
    rt_v, x_vec = [], []
    for i in range(len(predY0)):
        r = exp(predY0[i])
        cov_dia = std_var_Y0[i]**2 * r * r
        cov_current = cov.copy()
        for k in range(len(r)):
            cov_current[k, k] = cov_dia[k]
        x_opt = solve2(len(r), r, cov_current, gamma)
        
#         print(x_opt)
        
        tmp = sum(x_opt * exp(sample_Y0[i])) / cash
        rt_v.append(log(tmp))
        x_vec.append(x_opt)
    return rt_v, x_vec
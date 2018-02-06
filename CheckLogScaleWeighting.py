# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 06:17:29 2018

@author: Enrique Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt
from Lib_FS import log_scale


#SIMULATION RESULTS#
k01 = np.loadtxt('FS_Simul_Parabolic.txt', skiprows=1)
shape = np.loadtxt('FS_Simul_Params_Parabolic.txt', skiprows=1)

alfa = shape[0]   #cell constant 16*sqrt(R)/3.0
tip_simul = k01[:,2]*1.0e-9   #converting to meters

F_simul = k01[:,3]*1.0e-9  #converting to Newtons
t_simul = k01[:,0]
#SIMULATION RESULTS#

def FD_log2(x, f_x, nn, liminf, suplim):
    """this function returns time and f_time arrays equally spaced in logarithmic scale"""
    """Input: time starting with average dt(comming from repulsive_FD function), and f_time related to that time array"""
    lim_inf = round(np.log10(liminf),2)
    sup_lim = round(np.log10(suplim),2)
    b = np.linspace(lim_inf, sup_lim, nn)
    x_log = 10.0**b
    fx_log = np.zeros(len(x_log))
    for j in range(1, len(x_log)-1):
        for i in range(len(x)-1):
            if (x_log[j] - x[i])*(x_log[j] - x[i+1]) < 0.0 :  #change of sign
                if (x_log[j] - x[i]) < (x_log[j] - x[i+1]):
                    x_log[j] = x[i]
                    fx_log[j] = f_x[i]
                else:
                    x_log[j] = x[i+1]
                    fx_log[j] = f_x[i+1]
    return x_log, fx_log


#Weighting time, tip and force arrays in logarithmic scale
tip_log, t_log = log_scale(tip_simul, t_simul, 1.0e-4, 1.0)
F_log, _ = log_scale(F_simul, t_simul, 1.0e-4, 1.0)

t_log2, tip_log2 = FD_log2(t_simul, tip_simul, 40, 1.0e-4, 1.0)

plt.plot(t_log, tip_log, color='green', linestyle='dashed', marker='o', markerfacecolor= 'none', markersize=12, label ='log original')
plt.plot(t_log2, tip_log2, '^', label ='log new')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc=4)
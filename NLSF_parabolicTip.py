# -*- coding: utf-8 -*-
"""
Created on Mon Jan 5th 2018

@author: Enrique Alejandro
"""

import numpy as np
from Lib_FS import log_scale, chi_fit, linear_fit_Nob, log_tw, percent_error
import matplotlib.pyplot as plt
from Lib_rheology import chi_th, compliance, theta_loss


t_res = 1.0e-4   #time resolution (inverse of sampling frequency)
t_exp = 1.0  #total experimental time


#ORIGINAL PARAMETERS#
v = np.loadtxt('Original_Voigt_Params.txt', skiprows=1)
J_v = v[:,1]
tau_v = v[:,2]
Jg_v = v[0,0]
phi_0 = v[0,3]
#ORIGINAL PARAMETERS#

#SIMULATION RESULTS#
k01 = np.loadtxt('FS_Simul_Parabolic.txt', skiprows=1)
shape = np.loadtxt('FS_Simul_Params_Parabolic.txt', skiprows=1)

alfa = shape[0]   #cell constant 16*sqrt(R)/3.0
tip_simul = k01[:,2]*1.0e-9   #converting to meters

F_simul = k01[:,3]*1.0e-9  #converting to Newtons
t_simul = k01[:,0]

tip_log, t_log_sim = log_scale(tip_simul, t_simul, 1.0e-4, 1.0) #Weighting time and tip arrays in logarithmic scale
F_log, _ = log_scale(F_simul, t_simul, 1.0e-4, 1.0) #Weighting force array in logarithmic scale
Fdot = linear_fit_Nob(t_log_sim, F_log)    #Getting linear slope of force in time trace
chi_simul = alfa*pow(tip_log,1.5)/Fdot     #according to eq 19, relation between chi and tip when force is assumed to be linear
#SIMULATION RESULTS#

#NON-LINEAR SQUARE FITTING
arms = 4
R = (3.0/16*alfa)**2
Jg_c, tau_c, J_c = chi_fit(t_simul, tip_simul, F_simul, R, t_res, t_exp, arms, 1, 2.0e-10, 5.0e-8, 1.0e-3, 1.0e-7, 1.0e-2, 1.0e-6, 1.0e-1, 1.0e-6, 1.0e0)
Jg_a = np.zeros(arms)
Jg_a[:] = Jg_c
#NON-LINEAR SQUARE FITTING



#FIGURE WITH SUBPLOTS
t_th = log_tw(1.0e-5, 10.0)
t_log = log_tw(1.0e-4, 1.0)
omega = log_tw(1.0e-1, 1.0e5, 20)

plt.figure(figsize=(15,10))
plt.subplot(221)
chi_theor = chi_th(t_th, Jg_v, J_v, tau_v)
chi_5 = chi_th(t_log, Jg_c, J_c, tau_c)
plt.plot(t_log_sim, chi_simul, 'r*', markersize=15, label=r'Simulation, see Eq.(14)')
plt.plot(t_th, chi_theor, 'y', lw =5.0, label =r'Theoretical, see Eq.(8)')
plt.plot(t_log, chi_5, 'b', lw = 3.0, label=r'5-Voigt Fit, Eq. (9)')
plt.legend(loc=4, fontsize=13)
plt.xlabel(r'$time, \,s$', fontsize='20',fontweight='bold')
plt.ylabel(r'$\chi(t), \,Pa^{-1}s$',fontsize='20',fontweight='bold')
plt.xscale('log')
plt.yscale('log')

plt.subplot(222)
comp_5 = compliance(t_log, Jg_c, J_c, tau_c)
comp_th = compliance(t_log, Jg_v, J_v, tau_v)
plt.plot(t_log, comp_th, 'y', lw = 5.0, label=r'Theoretical')
plt.plot(t_log, comp_5, 'b', lw = 3.0, label=r'5-Voigt Fit')
plt.legend(loc=4, fontsize=14)
plt.xlabel(r'$time, \,s$', fontsize='20',fontweight='bold')
plt.ylabel(r'$J(t), \,Pa^{-1}$',fontsize='20',fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1.0e-4, 1.0)
plt.ylim(0.5e-8, 1.0e-5)

plt.subplot(223)
theta_th = theta_loss(omega, Jg_v, J_v, tau_v)
theta_5 = theta_loss(omega, Jg_c, J_c, tau_c)
plt.plot(omega, theta_th, 'y', lw = 5.0, label=r'Theoretical')
plt.plot(omega, theta_5, 'b', lw = 3.0, label=r'5-Voigt Fit')
plt.legend(loc=4, fontsize=13)
plt.xlabel(r'$\omega, \,rad/s$', fontsize='20',fontweight='bold')
plt.ylabel(r'$\theta(\omega),\,deg$',fontsize='20',fontweight='bold')
plt.xscale('log')

plt.subplot(224)
plt.plot(omega, percent_error(theta_th, theta_5), 'b', lw = 3.0, label=r'5-Voigt Fit')
plt.legend(loc=1, fontsize=13)
plt.xlabel(r'$\omega, \,rad/s$', fontsize='20',fontweight='bold')
plt.ylabel(r'$ Error \,\theta(\omega),\,\%$',fontsize='20',fontweight='bold')
plt.xscale('log')

plt.savefig('NLSF_LinearForce.png', bbox_inches='tight')
#FIGURE WITH SUBPLOTS


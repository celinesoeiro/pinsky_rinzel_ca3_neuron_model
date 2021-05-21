# -*- coding: utf-8 -*-
"""
Created on Fri May 21 07:23:37 2021

@author: celin
"""

import numpy as np;
import matplotlib.pyplot as plt

# Constants
mili = 10**(-3)
micro = 10**(-6)
pico = 10**(-9)
nano = 10**(-12)
kilo = 10**(3)
mega = 10**(6)

p = 1/3                     # admensional (1/3 da area eh soma)
G_LS = p*5*nano             # Siemens
G_Na = p*3*micro            # Siemens
G_K = p*2*micro             # Siemens
G_LD = (1 - p)*5*nano       # Siemens
G_Ca = (1 - p)*2*micro      # Siemens
G_KCa = (1 - p)*2.5*micro   # Siemens
G_KAHP = (1 - p)*40*nano    # Siemens
G_link = 20*nano            # Siemens (condutancia que liga dendrito e soma)
E_Na = 60*mili              # V
E_K = -75*mili              # V
E_Ca = 80*mili              # V
E_L = -60*mili              # V
C_S = p*100*nano            # F
C_D = (1 - p)*100*nano      # F

tau_Ca = 50*mili            # segundos
k = 2.5*mega/(1 - p)        # M/C

# Equations
## exp
def e(x):
    return np.exp(x)

def dendriticGateAuxVar(Vd, Ca):
    alpha_mca = 1600/(1 + e(-72*(Vd - 0.005)))
    if (Vd == -0.0089):
        beta_mca = 20/0.2 
    else:
        beta_mca = 20*kilo*(Vd + 0.0089)/(e(200*(Vd + 0.0089)) - 1)
    
    if (Vd > -0.010):
        alpha_kca = 2*kilo*e(-(0.0535 + Vd)/0.027)
    else:
        alpha_kca = e((Vd + 0.050)/0.011 -(Vd + 0.0535)/0.027)/0.018975

    if (Vd <= -0.010):
        beta_kca = (2*kilo*e(-(0.0535 + Vd)/0.027) - alpha_kca) 
    else:
        beta_kca = 0
    
    alpha_kahp = np.minimum(20,20000*Ca)
    beta_kahp = 4
    
    return alpha_mca, beta_mca, alpha_kca, beta_kca, alpha_kahp, beta_kahp

def somaticGateAuxVar(Vs):
    if (Vs == -0.0469):
        alpha_m = 320/0.25
    else:
        alpha_m = (320*kilo*(Vs + 0.0469))/(1 - e(-250*(Vs + 0.0469))) 
        
    if (Vs == -0.0199):
        beta_m = 280/0.2
    else:
        beta_m = (280*kilo*(Vs + 0.0199))/(e(200*(Vs + 0.0199)) - 1)

    alpha_h = 128*e(-(Vs + 0.043)/0.018)
    beta_h = 4*kilo/(1 + e(-200*(Vs + 0.020)))
    
    if (Vs == -0.0249):
        alpha_n = 16/0.2
    else:
        alpha_n = (16*kilo*(Vs + 0.0249))/(1 - e(-200*(Vs + 0.0249)))

    beta_n = 250*e(-25*(Vs + 0.040))

    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


# Time grid
simulationTime = 2;     # s
deltaT = 1*micro;       # s
t = np.arange(0, simulationTime + deltaT, deltaT) 

# Vector declarations
alpha_m = np.zeros(len(t))
beta_m = np.zeros(len(t))
alpha_h = np.zeros(len(t))
beta_h = np.zeros(len(t))
alpha_n = np.zeros(len(t))
beta_n = np.zeros(len(t))
alpha_Ca = np.zeros(len(t))
beta_Ca = np.zeros(len(t))
alpha_KCa = np.zeros(len(t))
beta_KCa = np.zeros(len(t))
alpha_KAHP = np.zeros(len(t))
beta_KAHP = np.zeros(len(t))

VS = np.zeros(len(t))     # Somatic voltage vector
VD = np.zeros(len(t))     # Dendritic voltage vector

Ca = np.zeros(len(t))     # dendritic calcium level (extra Ca above base level)

I_S = np.zeros(len(t))
I_D = np.zeros(len(t))

m = np.zeros(len(t))      # m: sodium activation gating variable
n = np.zeros(len(t))      # n: potassium activation gating variable
h = np.zeros(len(t))      # h: sodium inactivation gating variable
mCa = np.zeros(len(t))    # Ca current activation gating variable
mKCa = np.zeros(len(t))   # K_Ca current activation gating variable
mKAHP = np.zeros(len(t))  # K_AHP current activation gating variable

J_S = np.zeros(len(t))    # current injection in soma
J_D = np.zeros(len(t))    # current injection in dendritic

# initial conditions
VS[0] = E_L
VD[0] = E_L
Ca[0] = micro
m[0] = 0
h[0] = 0.5
n[0] = 0.4
mCa[0] = 0
mKCa[0] = 0.2
mKAHP[0] = 0.2


# Method: Euler -> y[n+1] = y[n] + dt*f(y[n])
for i in range(len(t) - 1):   
    if (np.isnan(VS[i]) or np.isnan(VD[i])): 
        print('position = ',i)
        break
    
    alpha_m[i], beta_m[i], alpha_h[i], beta_h[i], alpha_n[i], beta_n[i] = somaticGateAuxVar(VS[i]) 
    alpha_Ca[i], beta_Ca[i], alpha_KCa[i], beta_KCa[i], alpha_KAHP[i], beta_KAHP[i] = dendriticGateAuxVar(VD[i], Ca[i])
    
    m[i + 1] = m[i] + deltaT*(alpha_m[i]*(1 - m[i]) - beta_m[i]*m[i])
    h[i + 1] = h[i] + deltaT*(alpha_h[i]*(1 - h[i]) - beta_h[i]*h[i])
    n[i + 1] = n[i] + deltaT*(alpha_n[i]*(1 - n[i]) - beta_n[i]*n[i])
    mCa[i + 1] = mCa[i] + deltaT*(alpha_Ca[i]*(1 - mCa[i]) - beta_Ca[i]*mCa[i])
    mKCa[i + 1] = mKCa[i] + deltaT*(alpha_KCa[i]*(1 - mKCa[i]) - beta_KCa[i]*mKCa[i])
    mKAHP[i + 1] = mKAHP[i] + deltaT*(alpha_KAHP[i]*(1 - mKAHP[i]) - beta_KAHP[i]*mKAHP[i])
    
    # somatic conductances
    G_Na_now = G_Na*m[i + 1]*m[i + 1]*h[i + 1]
    G_K_now = G_K*n[i + 1]*n[i + 1]
    
    G_S_now = G_LS + G_Na_now + G_K_now + G_link
    
    # dendritic conductances
    G_Ca_now = G_Ca*mCa[i + 1]*mCa[i + 1]
    G_KAHP_now = G_KAHP*mKAHP[i + 1]
    if (Ca[i] > 250*micro): 
        G_KCa_now = G_KCa*mKCa[i + 1]
    else:
        G_KCa_now = G_KCa*mKCa[i + 1]*Ca[i + 1]/250*micro
    
    G_D_now = G_LD + G_Ca_now + G_KAHP_now + G_KCa_now + G_link
   
    # link current
    I_link = G_link*(VD[i] - VS[i])
    
    # somatic currents
    I_LS_now = -G_LS*(VS[i] - E_L)
    I_Na_now = -G_Na_now*(VS[i] - E_Na)
    I_K_now = -G_K_now*(VS[i] - E_K)
    
    I_S[i] = I_LS_now + I_Na_now + I_K_now + I_link
    
    # dendritic currents
    I_LD_now = -G_LD*(VD[i] - E_L)
    I_Ca_now = -G_Ca_now*(VD[i] - E_Ca)
    I_KCa_now = -G_KCa_now*(E_K - VD[i])
    I_KAHP_now = -G_KAHP_now*(VD[i] - E_K)
    
    I_D[i] = I_LD_now + I_Ca_now + I_KCa_now + I_KAHP_now - I_link
    
    # Voltages
    VS_inf = (G_LS*E_L + G_Na_now*E_Na + G_K_now*E_K + VD[i]*G_link)/G_S_now;
    VD_inf = (G_LD*E_L + G_Ca_now*E_Ca + G_KCa_now*E_K + G_KAHP_now*E_K + VS[i]*G_link )/G_D_now;
    
    VS[i + 1] = VS_inf - (VS_inf - VS[i])*e(-deltaT*G_S_now/C_S)
    VD[i + 1] = VD_inf - (VD_inf - VD[i])*e(-deltaT*G_D_now/C_D)
    
    Ca_inf = tau_Ca*k*I_Ca_now
    Ca[i + 1] = Ca_inf - (Ca_inf - Ca[i])*e(-deltaT/tau_Ca)
   

# Plots 
fig, axs = plt.subplots(2, sharex=True, figsize=(14,13))
fig.suptitle('Questao 2')
axs[0].set_title('VS - Tensão no soma')
axs[0].set(ylabel='VS (mV)')
axs[0].grid()
axs[0].plot(t,VS*kilo)

axs[1].set_title('VD - Tensão no dendrito')
axs[1].set(ylabel='VD (mV)')
axs[1].plot(t,VD*kilo)
axs[1].grid()

fig2, axs2 = plt.subplots(2, sharex=True, figsize=(14,13))
fig2.suptitle('Questao 2 - variaveis de gating')
axs2[0].plot(t,n, 'b', label='n - Ativação K+')
axs2[0].plot(t,m, 'r', label='m - Ativação Na+')
axs2[0].plot(t,h, 'k', label='h - Inativação Na+')
axs2[0].set_title('n, m e h - Variáveis de gating')
axs2[0].set(ylabel='n x m x h')
axs2[0].legend(shadow=True, fancybox=True)
axs2[0].grid()

axs2[1].plot(t,mCa, 'b', label='mCa - Ativação K+')
axs2[1].plot(t,mKCa, 'r', label='mKCa - Ativação Na+')
axs2[1].plot(t,mKAHP, 'k', label='mKAHP - Inativação Na+')
axs2[1].set_title('mCa, mKCa e mKAHP - Variáveis de gating')
axs2[1].set(ylabel='mCa x mKCa x mKAHP')
axs2[1].legend(shadow=True, fancybox=True)
axs2[1].grid()
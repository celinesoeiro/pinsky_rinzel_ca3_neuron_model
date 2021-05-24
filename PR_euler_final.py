import numpy as np;
import matplotlib.pyplot as plt

dt = 1e-06
tmax = 2

E_L = - 0.06
E_Na = 0.06
E_K = - 0.075
E_Ca = 0.08
S_frac = 1/3
D_frac = 1 - S_frac

# Conductance values for somatic channels follow
G_LS = np.dot(5e-09,S_frac)
G_Na = np.dot(3e-06,S_frac)
G_K = np.dot(2e-06,S_frac)

# Conductance values for dendritic channels follow
G_LD = np.dot(5e-09,D_frac)
G_Ca = np.dot(2e-06,D_frac)
G_KAHP = np.dot(4e-08,D_frac)
G_KCa = np.dot(2.5e-06,D_frac)

G_Link = 50e-09

tau_Ca = 0.05
convert_Ca = 5e6 / D_frac

CmS = np.dot(1e-10,S_frac)
CmD = np.dot(1e-10,D_frac)

# time grid
t = np.arange(0,tmax,dt)

# Vector declarations
VS = np.zeros(np.size(t))
VD=np.zeros(np.size(t))

Ca=np.zeros(np.size(t))

I_LD = np.zeros(np.size(t))
I_LS = np.zeros(np.size(t))
I_Na = np.zeros(np.size(t))
I_K = np.zeros(np.size(t))
I_Ca = np.zeros(np.size(t))
I_KAHP = np.zeros(np.size(t))
I_KCa = np.zeros(np.size(t))

n = np.zeros(np.size(t))
m = np.zeros(np.size(t))
h = np.zeros(np.size(t))

mca = np.zeros(np.size(t))
mkca = np.zeros(np.size(t))
mkahp = np.zeros(np.size(t))

Itot = np.zeros(np.size(t))
I_Link = np.zeros(np.size(t))
IS = np.zeros(np.size(t))
ID = np.zeros(np.size(t))
IS_inj_v = 200e-12
IS_inj = np.zeros(np.size(t))
ID_inj_v = 200e-12
ID_inj = ID_inj_v*np.ones(np.size(t))

# functions
def PR_soma_gating(Vm=None,*args,**kwargs):    
    alpha_m=np.multiply((Vm == - 0.0469),320) / 0.25 + np.multiply((Vm != - 0.0469),(np.dot(np.dot(320,1000.0),(Vm + 0.0469)))) / (1 - np.exp(np.dot(- 250,(Vm + 0.0469))))
    beta_m=np.multiply((Vm == - 19.9),280) / 0.2 + np.multiply((Vm != - 19.9),(np.dot(280000.0,(Vm + 0.0199)))) / (np.exp(np.dot(200,(Vm + 0.0199))) - 1)

    alpha_h=np.dot(128,np.exp(- (Vm + 0.043) / 0.018))
    beta_h=4000.0 / (1 + np.exp(np.dot(- 200,(Vm + 0.02))))

    alpha_n=np.multiply((Vm == - 0.0249),16) / 0.2 + np.multiply((Vm != - 0.0249),(np.dot(16000.0,(Vm + 0.0249)))) / (1 - np.exp(np.dot(- 200,(Vm + 0.0249))))
    beta_n=np.dot(250,np.exp(np.dot(- 25,(Vm + 0.04))))

    return alpha_m,beta_m,alpha_h,beta_h,alpha_n,beta_n

def PR_dend_gating(VmD=None,Ca=None,*args,**kwargs):
    alpha_mca=1600.0 / (1 + np.exp(np.dot(- 72,(VmD - 0.005))))
    beta_mca=np.multiply((VmD == - 0.0089),20) / 0.2 + np.multiply(np.multiply((VmD != - 0.0089),20000.0),(VmD + 0.0089)) / (np.exp(np.dot(200,(VmD + 0.0089))) - 1)

    alpha_kca=np.multiply(np.dot(2000.0,np.exp(- (0.0535 + VmD) / 0.027)),(VmD > - 0.01)) + np.multiply(np.exp((VmD + 0.05) / 0.011 - (VmD + 0.0535) / 0.027) / 0.018975,(VmD <= - 0.01))
    beta_kca=np.multiply((np.dot(2000.0,np.exp(- (0.0535 + VmD) / 0.027)) - alpha_kca),(VmD <= - 0.01))

    alpha_kahp=min(20,np.dot(20000.0,Ca))
    beta_kahp=np.dot(4,np.ones(np.size(alpha_kahp)))

    return alpha_mca,beta_mca,alpha_kca,beta_kca,alpha_kahp,beta_kahp

# initial conditions
VS[0] = E_L
VD[0] = E_L
Ca[0] = 0
n[0] = 0.4
h[0] = 0.5
mkahp[0] = 0.2
mkca[0] = 0.2
Ca[0] = 1e-06

for i in range(len(t) - 1):
    I_LS[i + 1]=np.dot(G_LS,(E_L - VS[i]))
    I_LD[i + 1]=np.dot(G_LD,(E_L - VD[i]))

    Vm=VS[i]
    VmD=VD[i]

    Catmp=Ca[i]

    mtmp=m[i]
    htmp=h[i]
    ntmp=n[i]
    mcatmp=mca[i]
    mkcatmp=mkca[i]
    mkahptmp=mkahp[i]

    alpha_m,beta_m,alpha_h,beta_h,alpha_n,beta_n = PR_soma_gating(Vm,nargout=6)
    alpha_mca,beta_mca,alpha_mkca,beta_mkca,alpha_mkahp,beta_mkahp = PR_dend_gating(VmD,Catmp,nargout=6)

    m[i + 1]=mtmp + np.dot(dt,(np.dot(alpha_m,(1 - mtmp)) - np.dot(beta_m,mtmp)))
    h[i + 1]=htmp + np.dot(dt,(np.dot(alpha_h,(1 - htmp)) - np.dot(beta_h,htmp)))
    n[i + 1]=ntmp + np.dot(dt,(np.dot(alpha_n,(1 - ntmp)) - np.dot(beta_n,ntmp)))

    mca[i + 1]=mcatmp + np.dot(dt,(np.dot(alpha_mca,(1 - mcatmp)) - np.dot(beta_mca,mcatmp)))
    mkca[i + 1]=mkcatmp + np.dot(dt,(np.dot(alpha_mkca,(1 - mkcatmp)) - np.dot(beta_mkca,mkcatmp)))
    mkahp[i + 1]=mkahptmp + np.dot(dt,(np.dot(alpha_mkahp,(1 - mkahptmp)) - np.dot(beta_mkahp,mkahptmp)))

    G_Na_now=np.dot(np.dot(np.dot(G_Na,m[i + 1]),m[i + 1]),h[i + 1])
    I_Na[i + 1]=np.dot(G_Na_now,(E_Na - VS[i]))

    G_K_now=np.dot(np.dot(G_K,n[i + 1]),n[i + 1])
    I_K[i + 1]=np.dot(G_K_now,(E_K - VS[i]))

    G_Ca_now=np.dot(np.dot(G_Ca,mca[i + 1]),mca[i + 1])
    I_Ca[i + 1]=np.dot(G_Ca_now,(E_Ca - VD[i]))

    if (Ca[i] > 0.00025):
        G_KCa_now=np.dot(G_KCa,mkca[i + 1])
    else:
        G_KCa_now=np.dot(np.dot(G_KCa,mkca[i + 1]),Ca[i]) / 0.00025
    I_KCa[i + 1]=np.dot(G_KCa_now,(E_K - VD[i]))

    G_KAHP_now=np.dot(G_KAHP,mkahp[i + 1])
    I_KAHP[i + 1]=np.dot(G_KAHP_now,(E_K - VD[i]))

    
    I_Link[i + 1]=np.dot(G_Link,(VD[i] - VS[i]))    
    IS[i + 1]=I_LS[i + 1] + I_Na[i + 1] + I_K[i + 1] + I_Link[i + 1] + IS_inj[i + 1]
    ID[i + 1]=I_LD[i + 1] + I_Ca[i + 1] + I_KCa[i + 1] + I_KAHP[i + 1] - I_Link[i + 1] + ID_inj[i + 1]

    gS_Tot=G_LS + G_Na_now + G_K_now + G_Link
    VS_inf=(np.dot(G_LS,E_L) + np.dot(G_Na_now,E_Na) + np.dot(G_K_now,E_K) + np.dot(VD[i],G_Link)) / gS_Tot

    gD_Tot=G_LD + G_Ca_now + G_KCa_now + G_KAHP_now + G_Link
    VD_inf=(np.dot(G_LD,E_L) + np.dot(G_Ca_now,E_Ca) + np.dot(G_KCa_now,E_K) + np.dot(G_KAHP_now,E_K) + np.dot(VS[i],G_Link)) / gD_Tot

    VS[i + 1]=VS_inf - np.dot((VS_inf - VS[i]),np.exp(np.dot(- dt,gS_Tot) / CmS))
    VD[i + 1]=VD_inf - np.dot((VD_inf - VD[i]),np.exp(np.dot(- dt,gD_Tot) / CmD))

    Ca_inf=np.dot(np.dot(tau_Ca,convert_Ca),I_Ca[i + 1])
    Ca[i + 1]=Ca_inf - np.dot((Ca_inf - Ca[i]),np.exp(- dt / tau_Ca))


'''
# Action Potential Counter
AP_Counter_S = 0
AP_Counter_D = 0
position = 0
t_AP_S = []
t_AP_D = []
for j in range (len(t) - 1):
    if (VS[j]*1000 > -10):
        for k in range(j, len(t) -1):
            if (VS[k]*1000 < -30):
                AP_Counter_S += 1
                t_AP_S.append(j)
    
    if (VD[j]*1000 > -10):
        for k in range(j, len(t) -1):
            if (VD[k]*1000 < -30):
                AP_Counter_D += 1
                t_AP_D.append(j)


print('Potenciais de açao S = ',AP_Counter_S)

t_between_AP_S = []
for l in range(len(t_AP_S) - 1):
    t_between_AP_S.append(t_AP_S[l + 1] - t_AP_S[l])

m_t_AP_S = round(np.mean(t_between_AP_S)*dt, 2)
print('media do tempo entre AP = ', m_t_AP_S)

t_between_AP_D = []
for l in range(len(t_AP_D) - 1):
    t_between_AP_D.append(t_AP_D[l + 1] - t_AP_D[l])

m_t_AP_D = round(np.mean(t_between_AP_D)*dt, 2)
print('media do tempo entre AP = ', m_t_AP_D)
'''

# Plots 
fig, axs = plt.subplots(3, sharex=True, figsize=(14,13))
fig.suptitle(f'Questao 5 - Tensões de membrana - G_c = {G_Link} - IDinj = {ID_inj_v}', fontsize=25)
axs[0].set_title('VS - Tensão no soma')
axs[0].set(ylabel='VS (mV)')
axs[0].grid()
axs[0].plot(t,VS*1000.0,'k')

axs[1].set_title('VD - Tensão no dendrito')
axs[1].set(ylabel='VD (mV)')
axs[1].plot(t,VD*1000.0, ':k')
axs[1].grid()

axs[2].set_title('Correntes injetadas')
axs[2].set(ylabel='IS_inj x ID_inj (pA)')
axs[2].plot(t,IS_inj, 'r', label='Corrente injetada no soma')
axs[2].plot(t,ID_inj, 'b', label='Corrente injetada no dendrito')
axs[2].legend(shadow=True, fancybox=True)
axs[2].grid()

fig2, axs2 = plt.subplots(3, sharex=True, figsize=(14,13))
fig2.suptitle(f'Questao 5 - variaveis de gating - G_c = {G_Link} - IDinj = {ID_inj_v}', fontsize=25)
axs2[0].plot(t,n, 'b', label='n - Ativação K+')
axs2[0].plot(t,m, 'r', label='m - Ativação Na+')
axs2[0].plot(t,h, 'k', label='h - Inativação Na+')
axs2[0].set_title('n, m e h - Variáveis de gating')
axs2[0].set(ylabel='n x m x h')
axs2[0].legend(shadow=True, fancybox=True)
axs2[0].grid()

axs2[1].plot(t,mca, 'b', label='mCa - Ativação K+')
axs2[1].plot(t,mkca, 'r', label='mKCa - Ativação Na+')
axs2[1].plot(t,mkahp, 'k', label='mKAHP - Inativação Na+')
axs2[1].set_title('mCa, mKCa e mKAHP - Variáveis de gating')
axs2[1].set(ylabel='mCa x mKCa x mKAHP')
axs2[1].legend(shadow=True, fancybox=True)
axs2[1].grid()

axs2[2].set_title('Correntes injetadas')
axs2[2].set(ylabel='IS_inj x ID_inj (pA)')
axs2[2].plot(t,IS_inj, 'r', label='Corrente injetada no soma')
axs2[2].plot(t,ID_inj, 'b', label='Corrente injetada no dendrito')
axs2[2].legend(shadow=True, fancybox=True)
axs2[2].grid()
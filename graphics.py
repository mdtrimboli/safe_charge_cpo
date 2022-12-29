import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt


plt.style.use('seaborn-v0_8-darkgrid')

"""
### COMPARACION DE REWARD

rew_curves_SL = open('curves/Rew_DDPG_SL.csv', 'rb')
rew_curves_DDPG = open('curves/Rew_DDPG_01.csv', 'rb')
rew_curves_RS5 = open('curves/Rew_DDPG_RS5.csv', 'rb')
rew_curves_RS12 = open('curves/Rew_DDPG_RS12.csv', 'rb')

data_SL = loadtxt(rew_curves_SL, delimiter=",")
data_DDPG = loadtxt(rew_curves_DDPG, delimiter=",")
data_RS5 = loadtxt(rew_curves_RS5, delimiter=",")
data_RS12 = loadtxt(rew_curves_RS12, delimiter=",")



plt.plot(data_DDPG, label='Only DDPG')
plt.plot(data_RS5, label='DDPG+RS (M=0.05)')
plt.plot(data_RS12, label='DDPG+RS (M=0.12)')
plt.plot(data_SL, label='DDPG+SL')


plt.legend(loc="lower right")
plt.xlabel("Episodes of evaluation phase")
plt.ylabel("Episodic reward")
plt.savefig('curves/Reward_comp.png', dpi=1200)
plt.show()
"""

"""
### COMPARACION DE LONGITUD

len_curves_SL = open('curves/Len_DDPG_SL.csv', 'rb')
len_curves_DDPG = open('curves/Len_DDPG_01.csv', 'rb')
len_curves_RS5 = open('curves/Len_DDPG_RS5.csv', 'rb')
len_curves_RS12 = open('curves/Len_DDPG_RS12.csv', 'rb')
data_SL = loadtxt(len_curves_SL, delimiter=",")
data_DDPG = loadtxt(len_curves_DDPG, delimiter=",")
data_RS_05 = loadtxt(len_curves_RS5, delimiter=",")
data_RS_12 = loadtxt(len_curves_RS12, delimiter=",")

plt.plot(data_SL, label='DDPG+SL')
plt.plot(data_DDPG, label='Only DDPG')
plt.plot(data_RS_05, label='DDPG+RS (M=0.05)')
plt.plot(data_RS_12, label='DDPG+RS (M=0.12)')


plt.legend(loc="upper right")
plt.xlabel("Episodes of evaluation phase")
plt.ylabel("Episode length")
plt.savefig('curves/Length_comp.png', dpi=1200)
plt.show()
"""

"""
### COMPARACION DE VL

vr_curves_SL = open('curves/AVConst_DDPG_SL.csv', 'rb')
vr_curves_DDPG = open('curves/AVConst_DDPG_01.csv', 'rb')
vr_curves_RS5 = open('curves/ACVConst_DDPG_RS5.csv', 'rb')
vr_curves_RS12 = open('curves/ACVConst_DDPG_RS12.csv', 'rb')
data_SL = loadtxt(vr_curves_SL, delimiter=",")
data_DDPG = loadtxt(vr_curves_DDPG, delimiter=",")
data_RS5 = loadtxt(vr_curves_RS5, delimiter=",")
data_RS12 = loadtxt(vr_curves_RS12, delimiter=",")

plt.plot(data_SL, label='DDPG+SL')
plt.plot(data_DDPG, label='Only DDPG')
plt.plot(data_RS5, label='DDPG+RS (M=0.05)')
plt.plot(data_RS12, label='DDPG+RS (M=0.12)')

plt.legend(loc="lower right")
plt.xlabel("Episodes of evaluation phase")
plt.ylabel("Cumulative constraint violations")
plt.savefig('curves/AVC_comp.png', dpi=1200)
plt.show()
"""


### COMPARACIÓN DE ITO
"""
t_curves = open('curves/T_SL.csv', 'rb')
t_curves2 = open('curves/T2_SL.csv', 'rb')
t_curves4 = open('curves/T4_SL.csv', 'rb')
t_curves6 = open('curves/T6_SL.csv', 'rb')

data_tdet = loadtxt(t_curves, delimiter=",")
data_T2 = loadtxt(t_curves2, delimiter=",")
data_T4 = loadtxt(t_curves4, delimiter=",")
data_T6 = loadtxt(t_curves6, delimiter=",")

plt.plot(45*np.ones(175), label='Temperature Limit')
plt.plot(data_tdet[:168], label='Deterministic')
plt.plot(data_T2[:168], label='Ito=2')
plt.plot(data_T4[:168], label='Ito=4')
plt.plot(data_T6[:168], label='Ito=6')
plt.legend(loc="lower right")
plt.xlabel("Evaluation steps")
plt.ylabel("Temperature [°C]")
plt.savefig('curves/Ito_var.png', dpi=1200)
plt.show()
"""

"""
### PERFILES

t_curves_i = open('curves/T_DDPG_RS5.csv', 'rb')
v_curves = open('curves/V_DDPG_RS5.csv', 'rb')
i_curves = open('curves/I_DDPG_RS5.csv', 'rb')
soc_curves = open('curves/SOC_DDPG_RS5.csv', 'rb')


data_T2_i = loadtxt(t_curves_i, delimiter=",")
data_V2 = loadtxt(v_curves, delimiter=",")
data_I2 = loadtxt(i_curves, delimiter=",")
data_SOC2 = loadtxt(soc_curves, delimiter=",")

sample = 192
#RS5=192, RS12=196, DDPG=125, SL=173

fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(hspace=0.4)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.plot(data_T2_i[:sample])
ax1.plot(45*np.ones(sample),  '--', label='Temperature Limit')
ax2.plot(data_V2[:sample])
ax3.plot(data_SOC2[:sample])
ax4.plot(data_I2[:sample])

ax1.set_ylim(24, 55)
ax2.set_ylim(3.6, 4.6)
ax3.set_ylim(0.2, 1.1)
ax4.set_ylim(-50, 0)




#fig.suptitle("Curves for Ito = 0", fontsize=14)
ax1.set_ylabel('Temperature')
ax2.set_ylabel('Voltage')
ax3.set_ylabel('SOC')
ax4.set_ylabel('Current')

ax1.set_xlabel('Steps')
ax2.set_xlabel('Steps')
ax3.set_xlabel('Steps')
ax4.set_xlabel('Steps')

ax1.set_title('(a)')
ax2.set_title('(b)')
ax3.set_title('(c)')
ax4.set_title('(d)')
##############################################
#IMPORTANTE!!!
plt.savefig('curves/EvalCurves_08Nov_RS5.png', dpi=600)
plt.show()
##############################################
"""

fig2 = plt.figure(figsize=(8, 6))

soh_SL = open('curves/SOH_DDPG_SL.csv', 'rb')
soh_DDPG = open('curves/SOH_DDPG_01.csv', 'rb')
soh_RS5 = open('curves/SOH_DDPG_RS5.csv', 'rb')
soh_RS12 = open('curves/SOH_DDPG_RS12.csv', 'rb')

SOH_SL = loadtxt(soh_SL, delimiter=",")
SOH_DDPG = loadtxt(soh_DDPG, delimiter=",")
SOH_RS5 = loadtxt(soh_RS5, delimiter=",")
SOH_RS12 = loadtxt(soh_RS12, delimiter=",")

plt.plot(SOH_SL[:173], label='with SL')
plt.plot(SOH_DDPG[:125], label='only DDPG')
plt.plot(SOH_RS5[:192], label='with RS (M=0.05)')
plt.plot(SOH_RS12[:196], label='with RS (M=0.12)')

plt.ylabel('SOH')
plt.xlabel('Steps')

#plt.xlim([0, 200])
#plt.ylim([0.8, 0.9])



plt.legend(loc="lower left")
plt.savefig('curves/SOH_compare.png', dpi=600)
plt.show()


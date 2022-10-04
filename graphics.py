import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt


plt.style.use('ggplot')


### COMPARACION DE REWARD

rew_curves_SL = open('curves/Rew_DDPG_SL.csv', 'rb')
rew_curves_DDPG = open('curves/Rew_DDPG.csv', 'rb')
rew_curves_RS_01 = open('curves/Rew_DDPG_01.csv', 'rb')
rew_curves_RS_02 = open('curves/Rew_DDPG_02.csv', 'rb')
data_SL = loadtxt(rew_curves_SL, delimiter=",")
data_DDPG = loadtxt(rew_curves_DDPG, delimiter=",")
data_RS_01 = loadtxt(rew_curves_RS_01, delimiter=",")
data_RS_02 = loadtxt(rew_curves_RS_02, delimiter=",")

plt.plot(data_SL, label='DDPG+SL')
plt.plot(data_DDPG, label='Only DDPG')
plt.plot(data_RS_01, label='DDPG+RS (M=0.1)')
plt.plot(data_RS_02, label='DDPG+RS (M=0.2)')

plt.legend(loc="lower right")
plt.xlabel("Episodes of evaluation phase")
plt.ylabel("Episodic reward")
plt.savefig('curves/Reward_comp.png', dpi=1200)
plt.show()


### COMPARACION DE LONGITUD

len_curves_SL = open('curves/Len_DDPG_SL.csv', 'rb')
len_curves_DDPG = open('curves/Len_DDPG.csv', 'rb')
len_curves_RS_01 = open('curves/Len_DDPG_01.csv', 'rb')
len_curves_RS_02 = open('curves/Len_DDPG_02.csv', 'rb')
data_SL = loadtxt(len_curves_SL, delimiter=",")
data_DDPG = loadtxt(len_curves_DDPG, delimiter=",")
data_RS_01 = loadtxt(len_curves_RS_01, delimiter=",")
data_RS_02 = loadtxt(len_curves_RS_02, delimiter=",")

plt.plot(data_SL, label='DDPG+SL')
plt.plot(data_DDPG, label='Only DDPG')
plt.plot(data_RS_01, label='DDPG+RS (M=0.1)')
plt.plot(data_RS_02, label='DDPG+RS (M=0.2)')

plt.legend(loc="upper right")
plt.xlabel("Episodes of evaluation phase")
plt.ylabel("Episode length")
plt.savefig('curves/Length_comp.png', dpi=1200)
plt.show()

### COMPARACION DE VR

vr_curves_SL = open('curves/AVConst_DDPG_SL.csv', 'rb')
vr_curves_DDPG = open('curves/AVConst_DDPG.csv', 'rb')
vr_curves_RS_01 = open('curves/AVConst_DDPG_01.csv', 'rb')
vr_curves_RS_02 = open('curves/AVConst_DDPG_02.csv', 'rb')
data_SL = loadtxt(vr_curves_SL, delimiter=",")
data_DDPG = loadtxt(vr_curves_DDPG, delimiter=",")
data_RS_01 = loadtxt(vr_curves_RS_01, delimiter=",")
data_RS_02 = loadtxt(vr_curves_RS_02, delimiter=",")

plt.plot(data_SL, label='DDPG+SL')
plt.plot(data_DDPG, label='Only DDPG')
plt.plot(data_RS_01, label='DDPG+RS (M=0.1)')
plt.plot(data_RS_02, label='DDPG+RS (M=0.2)')

plt.legend(loc="lower right")
plt.xlabel("Episodes of evaluation phase")
plt.ylabel("Cumulative constraint violations")
plt.savefig('curves/AVC_comp.png', dpi=1200)
plt.show()

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

plt.plot(data_tdet[:145], label='Deterministic')
plt.plot(data_T2[:145], label='Ito=2')
plt.plot(data_T4[:145], label='Ito=4')
plt.plot(data_T6[:145], label='Ito=6')
plt.legend(loc="upper left")
plt.xlabel("Evaluation steps")
plt.ylabel("Temperature [°C]")
plt.savefig('curves/Ito_var.png', dpi=1200)
plt.show()
"""


### PERFILES
"""
t_curves_i = open('curves/T_SL.csv', 'rb')
v_curves = open('curves/V_SL.csv', 'rb')
i_curves = open('curves/I_SL.csv', 'rb')
soc_curves = open('curves/SOC_SL.csv', 'rb')
soh_curves = open('curves/SOH_SL.csv', 'rb')
#soh_wsl_curves = open('curves/SOH_wSL.csv', 'rb')

data_T2_i = loadtxt(t_curves_i, delimiter=",")
data_V2 = loadtxt(v_curves, delimiter=",")
data_I2 = loadtxt(i_curves, delimiter=",")
data_SOC2 = loadtxt(soc_curves, delimiter=",")
data_SOH = loadtxt(soh_curves, delimiter=",")
#data_SOH_wsl = loadtxt(soh_wsl_curves, delimiter=",")

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.plot(data_T2_i[:161])
ax2.plot(data_V2[:161])
ax3.plot(data_SOC2[:161])
ax4.plot(data_I2[:161])
#fig.suptitle("Curves for Ito = 0", fontsize=14)
ax1.set_ylabel('Temperature')
ax2.set_ylabel('Voltage')
ax3.set_ylabel('SOC')
ax4.set_ylabel('Current')
##############################################
#IMPORTANTE!!!
plt.savefig('curves/EvalCurves_27Jul.png', dpi=1200)
plt.show()
##############################################


fig2 = plt.figure(figsize=(10, 5))
plt.plot(data_SOH, label='with SL')
plt.plot(data_SOH_wsl[:126], label='without SL')
plt.legend(loc="lower left")
plt.savefig('curves/SOH_compare1.png', dpi=1200)
plt.show()
"""

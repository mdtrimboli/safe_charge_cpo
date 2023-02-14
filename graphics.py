import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

sns.set_theme()

"""
### COMPARACION DE REWARD

rew_curves_SL = open('curves/Rew_DDPG_SL.csv', 'rb')
rew_curves_DDPG = open('curves/Rew_DDPG_01.csv', 'rb')
rew_curves_RS5 = open('curves/Rew_DDPG_RS5.csv', 'rb')
rew_curves_RS12 = open('curves/Rew_DDPG_RS12.csv', 'rb')

data_SL = gaussian_filter1d(loadtxt(rew_curves_SL, delimiter=","), sigma=5)
data_DDPG = gaussian_filter1d(loadtxt(rew_curves_DDPG, delimiter=","), sigma=5)
data_RS5 = gaussian_filter1d(loadtxt(rew_curves_RS5, delimiter=","), sigma=5)
data_RS12 = gaussian_filter1d(loadtxt(rew_curves_RS12, delimiter=","), sigma=5)




plt.plot(data_DDPG, label='DDPG')
plt.plot(data_RS5, label='DDPG+RS (M=0.05)')
plt.plot(data_RS12, label='DDPG+RS (M=0.12)')
plt.plot(data_SL, label='DDPG+SL')


plt.legend(loc="lower left")
plt.xlabel("Episodes of evaluation phase")
plt.ylabel("Episodic reward")
plt.savefig('curves/Reward_comp.png', dpi=600)
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

"""
### COMPARACIÓN DE ITO

t_curves = open('curves/T_DDPG_118.csv', 'rb')
t_curves2 = open('curves/T2_DDPG_118.csv', 'rb')
t_curves4 = open('curves/T4_DDPG_118.csv', 'rb')
t_curves6 = open('curves/T6_DDPG_118.csv', 'rb')

i_curves = open('curves/I_DDPG_118.csv', 'rb')
i_curves2 = open('curves/I2_DDPG_118.csv', 'rb')
i_curves4 = open('curves/I4_DDPG_118.csv', 'rb')
i_curves6 = open('curves/I6_DDPG_118.csv', 'rb')

soc_curves = open('curves/SOC_DDPG_118.csv', 'rb')
soc_curves2 = open('curves/SOC2_DDPG_118.csv', 'rb')
soc_curves4 = open('curves/SOC4_DDPG_118.csv', 'rb')
soc_curves6 = open('curves/SOC6_DDPG_118.csv', 'rb')

v_curves = open('curves/V_DDPG_118.csv', 'rb')
v_curves2 = open('curves/V2_DDPG_118.csv', 'rb')
v_curves4 = open('curves/V4_DDPG_118.csv', 'rb')
v_curves6 = open('curves/V6_DDPG_118.csv', 'rb')

data_tdet = loadtxt(t_curves, delimiter=",")
data_T2 = loadtxt(t_curves2, delimiter=",")
data_T4 = loadtxt(t_curves4, delimiter=",")
data_T6 = loadtxt(t_curves6, delimiter=",")

data_Vdet = loadtxt(v_curves, delimiter=",")
data_V2 = loadtxt(v_curves2, delimiter=",")
data_V4 = loadtxt(v_curves4, delimiter=",")
data_V6 = loadtxt(v_curves6, delimiter=",")

data_Idet = loadtxt(i_curves, delimiter=",")
data_I2 = loadtxt(i_curves2, delimiter=",")
data_I4 = loadtxt(i_curves4, delimiter=",")
data_I6 = loadtxt(i_curves6, delimiter=",")

data_SOCdet = loadtxt(soc_curves, delimiter=",")
data_SOC2 = loadtxt(soc_curves2, delimiter=",")
data_SOC4 = loadtxt(soc_curves4, delimiter=",")
data_SOC6 = loadtxt(soc_curves6, delimiter=",")


fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(hspace=0.4)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

sample = 125
#RS5=192, RS12=196, DDPG=125, SL=173

ax1.plot(data_tdet[0:sample], label='Deterministic')
ax1.plot(data_T2[0:sample], label='Ito=2')
ax1.plot(data_T4[0:sample], label='Ito=4')
ax1.plot(data_T6[0:sample], label='Ito=6')
ax1.legend(loc="lower right", fontsize='small')


ax2.plot(data_Vdet[0:sample], label='Deterministic')
ax2.plot(data_V2[0:sample], label='Ito=2')
ax2.plot(data_V4[0:sample], label='Ito=4')
ax2.plot(data_V6[0:sample], label='Ito=6')
ax2.legend(loc="lower right", fontsize='small')


ax3.plot(data_SOCdet[0:sample], label='Deterministic')
ax3.plot(data_SOC2[0:sample], label='Ito=2')
ax3.plot(data_SOC4[0:sample], label='Ito=4')
ax3.plot(data_SOC6[0:sample], label='Ito=6')
ax3.legend(loc="lower right", fontsize='small')

ax4.plot(data_Idet[0:sample], label='Deterministic')
ax4.plot(data_I2[0:sample], label='Ito=2')
ax4.plot(data_I4[0:sample], label='Ito=4')
ax4.plot(data_I6[0:sample], label='Ito=6')
ax4.legend(loc="lower right", fontsize='small')



#ax1.set_ylim(24, 55)
#ax2.set_ylim(3.6, 4.6)
#ax3.set_ylim(0.2, 1.1)
#ax4.set_ylim(-50, 0)


#fig.suptitle("Curves for Ito = 0", fontsize=14)
ax1.set_ylabel('Temperature [°C]')
ax2.set_ylabel('Voltage [Volts]')
ax3.set_ylabel('SOC')
ax4.set_ylabel('Current [Amp.]')

ax1.set_xlabel('Steps')
ax2.set_xlabel('Steps')
ax3.set_xlabel('Steps')
ax4.set_xlabel('Steps')

ax1.set_title('Mean Temperature')
ax2.set_title('Terminal Voltage')
ax3.set_title('State of Charge')
ax4.set_title('Action Current')

plt.savefig('curves/Ito_var.png', dpi=600)
plt.show()
"""

"""
### PERFILES

sample = 300
limit_1 = 171
limit_2 = 172
limit_4 = 171

t_curves_i1 = loadtxt(open('curves/T_DDPG_01_f.csv', 'rb'), delimiter=",")[:limit_1]
t_curves_i2 = loadtxt(open('curves/T_DDPG_02_f.csv', 'rb'), delimiter=",")[:limit_2]
t_curves_i4 = loadtxt(open('curves/T_DDPG_04_f.csv', 'rb'), delimiter=",")[:limit_4]

print(type(t_curves_i4))

v_curves_1 = loadtxt(open('curves/V_DDPG_01_f.csv', 'rb'), delimiter=",")[:limit_1]
v_curves_2 = loadtxt(open('curves/V_DDPG_02_f.csv', 'rb'), delimiter=",")[:limit_2]
v_curves_4 = loadtxt(open('curves/V_DDPG_04_f.csv', 'rb'), delimiter=",")[:limit_4]

i_curves_1 = loadtxt(open('curves/I_DDPG_01_f.csv', 'rb'), delimiter=",")[:limit_1]
i_curves_2 = loadtxt(open('curves/I_DDPG_02_f.csv', 'rb'), delimiter=",")[:limit_2]
i_curves_4 = loadtxt(open('curves/I_DDPG_04_f.csv', 'rb'), delimiter=",")[:limit_4]

soc_curves_1 = loadtxt(open('curves/SOC_DDPG_01_f.csv', 'rb'), delimiter=",")[:limit_1]
soc_curves_2 = loadtxt(open('curves/SOC_DDPG_02_f.csv', 'rb'), delimiter=",")[:limit_2]
soc_curves_4 = loadtxt(open('curves/SOC_DDPG_04_f.csv', 'rb'), delimiter=",")[:limit_4]



#RS5=230, RS12=228, DDPG=179, SL=183
#Again
#RS5=230, RS12=205, DDPG=179, SL=171
fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(hspace=0.4)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)



ax1.plot(t_curves_i4, label='Ito=4', color='tab:blue')
ax1.plot(t_curves_i2, label='Ito=2', color='tab:green')
ax1.plot(t_curves_i1, label='Ito=1', color='tab:orange')
ax1.plot(45*np.ones(max([sample, sample, sample])),  '--', label='Temperature Limit', color='red')
ax1.legend(loc="lower right", fontsize='x-small')

ax2.plot(v_curves_4, label='Ito=4', color='tab:blue')
ax2.plot(v_curves_2, label='Ito=2', color='tab:green')
ax2.plot(v_curves_1, label='Ito=1', color='tab:orange')


ax3.plot(soc_curves_4, label='Ito=4', color='tab:blue')
ax3.plot(soc_curves_2, label='Ito=2', color='tab:green')
ax3.plot(soc_curves_1, label='Ito=1', color='tab:orange')

ax4.plot(i_curves_4, label='Ito=4', color='tab:blue')
ax4.plot(i_curves_2, label='Ito=2', color='tab:green')
ax4.plot(i_curves_1, label='Ito=1', color='tab:orange')



ax1.set_ylim(25, 50.01)
ax1.set_xlim(0, 250)
ax2.set_ylim(3.4, 4.6)
ax2.set_xlim(0, 250)
ax3.set_ylim(0.2, 1.03)
ax3.set_xlim(0, 250)
ax4.set_ylim(-50, 2)
ax4.set_xlim(0, 250)

ax1.set_ylabel('Temperature [ºC]')
ax2.set_ylabel('Voltage [Volts]')
ax3.set_ylabel('SOC')
ax4.set_ylabel('Current [Amp.]')

ax1.set_xlabel('Steps')
ax2.set_xlabel('Steps')
ax3.set_xlabel('Steps')
ax4.set_xlabel('Steps')

ax1.set_title('Mean Temperature')
ax2.set_title('Terminal Voltage')
ax3.set_title('State of Charge')
ax4.set_title('Action Current')
##############################################
#IMPORTANTE!!!
plt.savefig('curves/Eval_31Ene_DDPG.png', dpi=600)
plt.show()
##############################################
"""


# DEGRADACION DEL SOH
fig2 = plt.figure(figsize=(8, 6))

soh_SL = open('curves/SOH_DDPG_SL.csv', 'rb')
soh_DDPG = open('curves/SOH_DDPG_01.csv', 'rb')
soh_RS5 = open('curves/SOH_DDPG_RS5.csv', 'rb')
soh_RS12 = open('curves/SOH_DDPG_RS12.csv', 'rb')

SOH_SL = loadtxt(soh_SL, delimiter=",")
SOH_DDPG = loadtxt(soh_DDPG, delimiter=",")
SOH_RS5 = loadtxt(soh_RS5, delimiter=",")
SOH_RS12 = loadtxt(soh_RS12, delimiter=",")

plt.plot(SOH_SL[:170], label='with SL')
plt.plot(SOH_DDPG[:170], label='DDPG')
plt.plot(SOH_RS5[:193], label='with RS (M=0.05)')
plt.plot(SOH_RS12[:202], label='with RS (M=0.12)')

print(0.9-SOH_SL[169])
print(0.9-SOH_DDPG[169])
print(0.9-SOH_RS5[192])
print(0.9-SOH_RS12[201])

plt.ylabel('SOH')
plt.xlabel('Steps')

#plt.xlim([0, 200])
#plt.ylim([0.8, 0.9])

plt.legend(loc="lower left")
plt.savefig('curves/SOH_compare.png', dpi=600)
plt.show()


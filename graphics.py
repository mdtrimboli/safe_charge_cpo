import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt


t_curves = open('curves/Tdet.csv', 'rb')
t_curves2 = open('curves/T2.csv', 'rb')
t_curves4 = open('curves/T4.csv', 'rb')
t_curves6 = open('curves/T6.csv', 'rb')

data_tdet = loadtxt(t_curves, delimiter=",")
data_T2 = loadtxt(t_curves2, delimiter=",")
data_T4 = loadtxt(t_curves4, delimiter=",")
data_T6 = loadtxt(t_curves6, delimiter=",")




plt.style.use('ggplot')
plt.plot(data_tdet[:145], label='Deterministic')
plt.plot(data_T2[:145], label='Ito=2')
plt.plot(data_T4[:145], label='Ito=4')
plt.plot(data_T6[:145], label='Ito=6')
plt.legend(loc="upper left")
plt.xlabel("Evaluation steps")
plt.ylabel("Temperature [°C]")
plt.savefig('curves/Ito_var.png', dpi=1200)
plt.show()

v_curves = open('curves/V2.csv', 'rb')
i_curves = open('curves/I2.csv', 'rb')
soc_curves = open('curves/SOC2.csv', 'rb')
data_V2 = loadtxt(v_curves, delimiter=",")
data_I2 = loadtxt(i_curves, delimiter=",")
data_SOC2 = loadtxt(soc_curves, delimiter=",")

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.plot(data_T2[:150])
ax2.plot(data_V2[:150])
ax3.plot(data_SOC2[:150])
ax4.plot(data_I2[:150])
ax4.set_ylim([-50, -0])
fig.suptitle("Curves for Ito = 2", fontsize=14)
ax1.set_ylabel('Temperature')
ax2.set_ylabel('Voltage')
ax3.set_ylabel('SOC')
ax4.set_ylabel('Current')
plt.savefig('curves/EvalCurves.png', dpi=1200)
plt.show()


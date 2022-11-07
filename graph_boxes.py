import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_3 = np.array([324, 6863, 478, 158, 216, 99, 95, 524, 285])
data_7 = np.array([546, 355, 12, 354, 8, 29, 10, 146, 25, 1336])
data_11 = np.array([29, 38, 520, 6, 20, 27, 157, 37, 21, 11])
data_15 = np.array([5, 167, 3, 89, 23, 45, 43, 37, 7, 5])
#data_19 = np.array([518, 34, 104, 328, 40])
#data_23 = np.array([68, 32, 20, 62, 68])

data = np.transpose([data_3, data_7, data_11, data_15])
print(data)

df = pd.DataFrame([data], index=['0.05', '0.1', '0.12', '0.15'])

"""
fig, ax = plt.subplots()
ax.boxplot(data)
plt.xticks([1, 2, 3, 4, 5, 6], ["0.03", "0.07", "0.11", "0.15", "0.19", "0.23"], rotation=10)
plt.xlabel("Reward Shaping Margin (M)")
plt.ylabel("Accumulated Constraint Violations")
plt.savefig('curves/DDPG+RS.png', dpi=1200)
plt.show()
"""
x = np.array([0.05, 0.1, 0.12, 0.15])

sns.set_style("darkgrid")
fig = sns.boxplot(data=df, color='skyblue')
plt.xlabel("Margin (M)")
plt.ylabel("Accumulated Constraint Violations")
plt.savefig('curves/DDPG+RS.png', dpi=1200)

"""
# Select which box you want to change
mybox = ax.artists[2]

# Change the appearance of that box
mybox.set_facecolor('red')
mybox.set_edgecolor('black')
mybox.set_linewidth(2)
"""
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_3 = np.array([597, 345, 339, 329, 0, 526, 267, 777, 141, 350])
data_7 = np.array([267, 216, 0, 60, 66, 60, 0, 0, 0, 192])
data_11 = np.array([0, 0, 60, 165, 60, 0, 0, 189, 51, 0])
data_15 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#data_19 = np.array([518, 34, 104, 328, 40])
#data_23 = np.array([68, 32, 20, 62, 68])

data = np.transpose([data_3, data_7, data_11, data_15])

df = pd.DataFrame(data, columns=['0.05', '0.1', '0.15', '0.2'])

"""
fig, ax = plt.subplots()
ax.boxplot(data)
plt.xticks([1, 2, 3, 4, 5, 6], ["0.03", "0.07", "0.11", "0.15", "0.19", "0.23"], rotation=10)
plt.xlabel("Reward Shaping Margin (M)")
plt.ylabel("Accumulated Constraint Violations")
plt.savefig('curves/DDPG+RS.png', dpi=1200)
plt.show()
"""
x = np.array([0.05, 0.1, 0.15, 0.2])

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

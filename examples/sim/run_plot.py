import matplotlib.pyplot as plt
import numpy as np

# %%%%%%%%%%%%%%%% plot timing between QP and ours
# number of contacts
n_cube = np.array([5, 10, 15, 20])
# timing
qp_timing = np.array([0.00463, 0.01009, 0.01660, 0.02416]) * 1000
ours_timing = np.array([0.00126, 0.00256, 0.00384, 0.005190]) * 1000

# Plotting the bar chart
bar_width = 0.2
x_indices = np.arange(len(n_cube))  # Create an array of indices for bar positioning
plt.bar(x_indices - bar_width / 2, qp_timing, width=bar_width, color='#1f77b4', alpha=0.9, label='QP model')
plt.bar(x_indices + bar_width / 2, ours_timing, width=bar_width, color='#ff7f0e', alpha=0.9, label='proposed')

plt.xticks(x_indices, n_cube, fontsize=22)
plt.yticks(fontsize=22)

plt.xlabel('Number of Cubes', fontsize=22)
plt.ylabel('Time (ms)', fontsize=22)
plt.legend(fontsize=18)
plt.grid()
plt.tight_layout()
plt.show()

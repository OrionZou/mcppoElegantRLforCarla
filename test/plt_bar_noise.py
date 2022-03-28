import matplotlib.pyplot as plt
import numpy as np
font={'weight':'bold','size':20}
# Longitudinal Velocity
a = [9.0911, 9.0978, 8.9198, 8.7912]
a_SD = [0.0092, 0.0443, 0.0654, 0.1286]
b = [9.2301, 9.7837, 9.8011, 9.6816]
b_SD = [0.0076, 0.0670, 0.1398, 0.1554]
# # Delta Steer
# a = [0.0030, 0.0040, 0.0054, 0.0061]
# a_SD = [0, 0.0001, 0.0004, 0.0015]
# b = [0.0156, 0.0149, 0.0226, 0.0257]
# b_SD = [0.0003, 0.0010, 0.0034, 0.0066]
# # Comfort
# a = [0.7620, 0.7579, 0.7725, 0.8062]
# a_SD = [0.0002, 0.0023, 0.0361, 0.0664]
# b = [0.7135, 0.6962, 0.7111, 0.7470]
# b_SD = [0.0008, 0.0043, 0.0579, 0.0676]

x = np.array(['', '0.00', '0.01', '0.05', '0.10'])
# fig, ax = plt.subplots(figsize=(3.5, 3))
fig, ax = plt.subplots(figsize=(5, 5))
plt.errorbar(np.arange(4), a, yerr=a_SD, label='MCPPO', fmt='o:',linewidth=4, elinewidth=3, ms=13, capsize=5)
plt.errorbar(np.arange(4), b, yerr=b_SD, label='PPO', fmt='o:',linewidth=4, elinewidth=3, ms=13, capsize=5)
ax.set_xticklabels(x, rotation=0)
plt.legend(fontsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.ylim((8, 11))
plt.ylabel("$V_{lon}$", fontdict=font)
# plt.ylim((0.000, 0.035))
# plt.ylabel("$\Delta Steer$", fontdict=font)
# plt.ylim((0.65, 0.90))
# plt.ylabel("$Comfort$", fontdict=font)

plt.xlabel("$Noise \delta$", fontdict=font)

ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
plt.show()
# fig.savefig(f'/home/zgy/fig/comfort.svg', format='svg', dpi=5000)
# fig.savefig(f'/home/zgy/fig/steer.svg', format='svg', dpi=5000)
fig.savefig(f'/home/zgy/fig/lon_v.svg', format='svg', dpi=5000)
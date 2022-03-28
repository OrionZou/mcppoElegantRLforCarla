import numpy as np
import matplotlib.pyplot as plt

size = 4
x = np.arange(size)
font={'weight':'bold','size':20}

# a = [581.8678, 632.1061, 645.9068, 612.5583]
# a_SD = [0.2285, 27.6458, 44.3300, 63.9349]
# b = [492.7769, 211.7627, 77.8927, 53.7920]
# b_SD = [0.0008, 0.9489, 28.0084, 12.9126]

# a = [50.0000, 49.5372, 48.7310, 46.8809]
# a_SD = [0, 1.4867, 2.6328, 3.9099]
# b = [50.0000, 6.4471, 2.7018, 2.1127]
# b_SD = [0, 0.1231, 0.7891, 0.3763]

# Longitudinal Velocity
# a = [9.2197, 9.1299, 8.9017, 8.6111]
# a_SD = [0] * 4
# b = [10.1102, 10.0770, 9.8439, 9.8173]
# b_SD = [0] * 4

# # Longitudinal Velocity
a = [9.2197, 9.1299, 8.9017, 8.6111]
a_SD = [0] * 4
b = [10.1102, 10.0770, 9.8439, 9.8173]
b_SD = [0] * 4
c = [9.8039, 9.7272, 9.6500, 9.8147]
c_SD = [0] * 4

# Delta Steer
# a = [0.0029, 0.0028, 0.0050, 0.0097]
# a_SD = [0] * 4
# b = [0.0092, 0.0121, 0.0200, 0.0264]
# b_SD = [0] * 4
# c = [0.0185, 0.0256, 0.0350, 0.0504]
# c_SD = [0] * 4

# Comfort
# a = [0.7512, 0.7556, 0.7588, 0.7456]
# a_SD = [0] * 4
# b = [0.6950, 0.6989, 0.6995, 0.6993]
# b_SD = [0] * 4
# c = [0.7076, 0.7075, 0.7031, 0.6843]
# c_SD = [0] * 4

total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2
labels = ['0.00', '0.01', '0.05', '0.1']

# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 3
# labels = ['0.02', '0.05', '0.10', '0.20']

fz=(3.5, 3)
# fig, ax = plt.subplots(figsize=fz)
fig, ax = plt.subplots()
plt.bar(x, a, width=width, yerr=a_SD, tick_label=labels, label='MCPPO')
plt.bar(x + width, b, width=width, yerr=b_SD, tick_label=labels, label='PPO')
# plt.bar(x + width * 2, c, width=width, yerr=c_SD, tick_label=labels, label='PPO-D')
plt.legend(fontsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.ylim((0, 1000))
# plt.title("Distance")
# plt.ylim((0, 70))
# plt.title("$Time$")
plt.ylim((8, 11))
plt.title("$V_{lon}$",fontdict=font)
# plt.ylim((0.000, 0.060))
# plt.title("$\Delta Steer$",fontdict=font)
# plt.ylim((0.68, 0.80))
# plt.title("$Comfort$",fontdict=font)

# plt.xlabel("$Noise \delta$")
plt.xlabel("$\Delta t$", fontdict=font)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
plt.show()

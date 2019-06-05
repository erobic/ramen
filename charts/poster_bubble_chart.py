# libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

fig, ax = plt.subplots(figsize=(48, 18))
comparisons_table = np.array([['Dataset/Algorithm', 'UpDn', 'QCG', 'BAN', 'MAC', 'RN', 'RAMEN'],
                              ['VQAv1', 60.62, 59.9, 62.98, 54.08, 51.84, 61.98],
                              ['VQAv2', 64.55, 57.08, 67.39, 54.35, 60.96, 65.96],
                              ['TDIUC', 68.82, 65.57, 71.1, 66.43, 65.06, 72.52],
                              ['CVQA', 57.01, 56.45, 57.36, 50.99, 48.11, 58.92],
                              ['VQACPV2', 38.01, 38.32, 39.31, 31.96, 26.7, 39.21],
                              ['CLEVR', 80.04, 46.73, 90.79, 98, 95.97, 96.92],
                              ['CLEVR-CoGenT-A', 82.47, 59.63, 92.5, 98.04, 96.5, 96.74],
                              ['CLEVR-CoGenT-B', 72.22, 53.45, 79.48, 90.41, 84.68, 89.07]])

natural_vs_synthetic = np.array([['Dataset/Algorithm', 'UpDn', 'QCG', 'BAN', 'MAC', 'RN', 'RAMEN'],
                                 ['Natural Datasets', 57.802, 55.464, 59.628, 51.562, 50.534, 59.718],
                                 ['Synthetic Datasets', 78.24333333, 53.27, 87.59, 95.48333333, 92.38333333,
                                  94.24333333]])
# # create data
# x = np.random.rand(5)
# y = np.random.rand(5)
font_scale = 3
scale = 10
z = np.random.rand(5)

x_natural = np.array([57.802, 55.464, 59.628, 51.562, 50.534, 59.718])
y_synthetic = np.array([78.24333333, 53.27, 87.59, 95.48333333, 92.38333333,
                        94.24333333])
methods = ['UpDn', 'QCG', 'BAN', 'MAC', 'RN', 'RAMEN']
colors = ['brown', 'black', 'blue', 'orange', 'magenta', 'blue']
# Change color with c and alpha
ax.scatter(x_natural, y_synthetic, s=1500 * scale, c='#5b9bd5ff', alpha=0.75)
plt.xlabel('Natural Datasets', fontsize=35 * font_scale)
plt.ylabel('Synthetic Datasets', fontsize=35 * font_scale)

for x, y, m in zip(x_natural, y_synthetic, methods):
    x_shift = .35 * len(m) / 2
    if m == 'BAN':
        y_shift = -8.0
    else:
        y_shift = +5.0
    ax.annotate(m, (x - x_shift, y + y_shift), fontsize=40 * font_scale)

ax.set_xbound(49, 61)
ax.set_xticks([52, 56, 60])
ax.tick_params(axis='both', which='both', labelsize=30 * font_scale)
ax.grid(b=True, which='both', axis='both', linewidth=2, linestyle='-')
ax.set_yscale('log')
# ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.set_ybound(50, 125)
ax.set_yticks([])
ax.set_yticks([50, 75, 100], minor=True)

ax.get_yaxis().set_minor_formatter(ScalarFormatter())
#ax.set_facecolor('xkcd:salmon')


plt.show()
# plt.savefig('a.png')

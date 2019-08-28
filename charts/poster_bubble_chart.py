# libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

fig, ax = plt.subplots(figsize=(40, 18))

# natural_vs_synthetic = np.array([['Dataset/Algorithm', 'UpDn', 'QCG', 'BAN', 'MAC', 'RN', 'RAMEN'],
#                                  ['Natural Datasets', 3.8, 2.8, 5.6, 2, 1.4, 5.4],
#                                  ['Synthetic Datasets', 2, 1, 3, 6, 4, 5]])

natural_vs_synthetic = np.array([['Dataset/Algorithm', 'UpDn', 'QCG', 'BAN', 'MAC', 'RN'],
                                 ['Natural Datasets', 3.8, 2.8, 5.6, 2, 1.4],
                                 ['Synthetic Datasets', 2, 1, 3, 6, 4]])

# # create data
# x = np.random.rand(5)
# y = np.random.rand(5)
font_scale = 3
scale = 10
z = np.random.rand(5)

x_natural = np.array(natural_vs_synthetic[1, 1:]).astype(np.float32)
y_synthetic = np.array(natural_vs_synthetic[2, 1:]).astype(np.float32)
methods = natural_vs_synthetic[0][1:]
# colors = ['brown', 'black', 'blue', 'orange', 'magenta', 'blue']
GREEN = '#6aa84fff'
YELLOW = '#f1c232ff'
colors = [GREEN, GREEN, GREEN, YELLOW, YELLOW, 'gray']
# Change color with c and alpha
BLUE = '#5b9bd5ff'
ax.scatter(x_natural, y_synthetic, s=3000 * scale, c=colors, alpha=1.0)
# plt.xlabel(natural_vs_synthetic[1,0], fontsize=35 * font_scale)
# plt.ylabel(natural_vs_synthetic[2,0], fontsize=35 * font_scale)
plt.xlabel('')
plt.ylabel('')
for x, y, m in zip(x_natural, y_synthetic, methods):
    # x_shift = .35 * len(m) / 2
    # if m == 'BAN':
    #     y_shift = -8.0
    # else:
    #     y_shift = +5.0
    x_shift = .25 * len(m) / 2
    y_shift = .65
    if 'ramen' in m.lower():
        ax.annotate('RAMEN', (x - x_shift, y + 2 * y_shift), fontsize=40 * font_scale)
        ax.annotate('(Best Overall)', (x - 2* x_shift, y + y_shift), fontsize=40 * font_scale)
    else:
        ax.annotate(m, (x - x_shift, y + y_shift), fontsize=40 * font_scale)

# ax.set_xbound(49, 61)
# ax.set_xticks([52, 56, 60])
ax.tick_params(axis='both', which='both', labelsize=30 * font_scale)
ax.grid(b=True, which='both', axis='both', linewidth=2, linestyle='-')
# ax.set_yscale('log')
# ax.get_yaxis().set_major_formatter(ScalarFormatter())
# ax.set_ybound(50, 125)
ax.set_ybound(0, 8)
ax.set_xbound(0, 6)
ax.set_xticks(range(2,7, 2))
ax.set_yticks(range(2,7, 2))
ax.set_xticklabels([])
ax.set_yticklabels([])
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_yticks([])
# ax.set_yticks([50, 75, 100], minor=True)

# ax.get_yaxis().set_minor_formatter(ScalarFormatter())
# ax.set_facecolor('xkcd:salmon')


plt.show()
# plt.savefig('a.png')

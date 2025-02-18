import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
'''
梯度更新过程性能
'''
acc_train=[0.375, 0.75,0.95833333,1,1,1]
acc_test=[0.336,0.9185,0.969,0.969,0.969,0.969]

font_size = 12
# error_witdh = np.arange(1, 10)
fig, axes = plt.subplots(figsize=(3, 2), tight_layout=True)
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
x=list(range(0, 6))
axes.plot(x, acc_train,
              lw=3,
              ls='-',
              color='seagreen',
              marker='o',
              markersize=6,
              alpha=0.8,
              markerfacecolor='seagreen',
              label="Meta-training phase")
axes.plot(x, acc_test,
              lw=3,
              ls='--',
              color='indianred',
              marker='s',
              markersize=6,
              alpha=0.8,
              markerfacecolor='indianred',
              label="Fine-tuning phase")
axes.set_xlabel('Number of gradient steps', fontProperties=font)
axes.set_ylabel('Accuracy', fontProperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
axes.legend(prop=font, frameon=True, framealpha=0.5,
                    loc='best', ncol=1)

axes.tick_params(labelsize=14, direction='in', top=True, right=True)
axes.grid(axis="y", color='black',
           alpha=.3, linewidth=0.5, linestyle="-")
axes.grid(axis="x", color='black',
           alpha=.3, linewidth=0.5, linestyle="-")
axes.yaxis.set_minor_locator(AutoMinorLocator())
axes.xaxis.set_minor_locator(AutoMinorLocator())
axes.tick_params(axis='y', which='minor', direction='in', right=True)
axes.tick_params(axis='x', which='minor', direction='in', top=True)
plt.show()
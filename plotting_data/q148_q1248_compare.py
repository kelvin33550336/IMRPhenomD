import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('q148_testing.dat', delimiter=' ', header=None)
data2 = pd.read_csv('q1248_testing.dat', delimiter=' ', header=None)
bins = np.arange(-2, 2, 0.125)
logdiff = np.log10(data[2]/data[1])
logdiff2 = np.log10(data2[2]/data2[1])

plt.hist(logdiff, bins=bins, alpha=0.6, label='q148')
plt.hist(logdiff2, bins=bins, alpha=0.6, label='q1248')

plt.legend()
plt.ylabel('Count')
plt.xlabel('$\log(\mathrm{mismatch})$')
plt.savefig('../mismatch/q148_q1248_compare.pdf', dpi=5000, bbox_inches='tight')
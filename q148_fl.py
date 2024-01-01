import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import paths

data = np.loadtxt(paths.data / "q148_fl.dat")

bins = np.arange(-5, -1, 0.25)
plt.hist(np.log10(data[4]).values, bins=bins, alpha=0.6, label='original')
plt.hist(np.log10(data[5].values), bins=bins, alpha=0.6, label='optimized')
plt.legend()
plt.ylabel('Count')
plt.xlabel('$\log(\mathrm{mismatch})$')
plt.savefig(paths.figures / "q148_fl.pdf", bbox_inches="tight")
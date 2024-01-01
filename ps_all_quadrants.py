import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import paths

data = np.loadtxt(paths.data / "all_quadrants.dat")

logdiff = np.log10(data[:,5] / data[:,4])
cm = plt.cm.get_cmap('RdYlBu')
colors = logdiff
sc = plt.scatter(data[:,2], data[:,3], c=colors, vmin=-1, vmax=1, s=30, cmap=cm, alpha=0.8)
plt.colorbar(sc).ax.set_ylabel('$\log(MM_{\mathrm{opt}}/MM_{\mathrm{ori}})$', rotation=270, labelpad=15)
plt.xlabel('$\chi_1$')
plt.ylabel('$\chi_2$')

plt.savefig(paths.figures / "ps_all_quadrants.pdf", bbox_inches="tight")
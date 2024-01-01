import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import paths

data = np.loadtxt(paths.data / "q148.dat")

logdiff = np.log10(data[:,5] / data[:,4])
cm = plt.cm.get_cmap('RdYlBu')
colors = logdiff
sc = plt.scatter(data[:,0], data[:,1], c=colors, vmin=-1, vmax=1, s=30, cmap=cm, alpha=0.8)
plt.colorbar(sc).ax.set_ylabel('$\log(MM_{\mathrm{opt}}/MM_{\mathrm{ori}})$', rotation=270, labelpad=15)
plt.xlabel('$q$')
plt.ylabel('$\chi_{\mathrm{PN}}$')

plt.savefig(paths.figures / "ps_q148_qchi.pdf", bbox_inches="tight")
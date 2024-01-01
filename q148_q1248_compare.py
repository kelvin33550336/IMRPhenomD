import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import paths

q148_data = np.loadtxt(paths.data / "q148.dat")
q1248_data = np.loadtxt(paths.data / "q1248.dat")

bins = np.arange(-2, 2, 0.125)
q148_logdiff = np.log10(q148_data[5]/q148_data[4])
q1248_logdiff = np.log10(q1248_data[5]/q1248_data[4])

plt.hist(q148_logdiff, bins=bins, alpha=0.6, label='q148')
plt.hist(q1248_logdiff, bins=bins, alpha=0.6, label='q1248')

plt.legend()
plt.ylabel('Count')
plt.xlabel('$\log(\mathrm{mismatch})$')
plt.savefig(paths.figures / "q148_q1248_compare.pdf", bbox_inches="tight")
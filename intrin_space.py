import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import paths

train_data = np.loadtxt(paths.data / "ps_train.dat")
test_data = np.loadtxt(paths.data / "q148.dat")

# train_data = np.loadtxt('./plotting_data/ps_train.dat', delimiter=' ') # [q, chi_PN, chi1, chi2]
# test_data = np.loadtxt('./plotting_data/q148.dat', delimiter=' ') # [q, chi_PN, chi1, chi2, ori_loss, opt_loss]
plt.scatter(test_data[:,0], test_data[:,1], s=30, label='testing')
plt.scatter(train_data[:,0], train_data[:,1], s=100, label='training')
plt.legend()
plt.xlabel('$q$')
plt.ylabel('$\chi_{\mathrm{PN}}$')
plt.savefig(paths.figures / "intrin_space.pdf", bbox_inches="tight")
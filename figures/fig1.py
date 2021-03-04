"""
Simple two-neuron schematic of single trial responses and decoding axis, with gaussians projected onto w_opt

Shuffled distribution of pairwise covariance for the two neurons
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 6


# generate pseudo data, X, for decoding
k = 50
u1 = [3, 0]
u2 = [0, 1]
cov = np.array([[1, 0.3], [0.3, 1]])
x1 = np.random.multivariate_normal(u1, cov, k)
x2 = np.random.multivariate_normal(u2, cov, k)
X = np.stack([x1, x2])

# generate list of shuffled covariance values for k=k
nshuff = 500
scov = []
for i in range(0, nshuff):
    x = np.random.choice(x1[:, 0], x1.shape[0], replace=False)
    y = np.random.choice(x1[:, 1], x1.shape[0], replace=False)
    scov.append(np.cov(x, y)[0, 1])

# for a range of k, generate shuffled distros
kscov = []
krange = np.arange(25, 200, 2)
_x1 = np.random.multivariate_normal(u1, cov, 250)
for _k in krange:
    _s = []
    for i in range(0, nshuff):
        x = np.random.choice(_x1[:, 0], _k, replace=False)
        y = np.random.choice(_x1[:, 1], _k, replace=False)
        _s.append(np.cov(x, y)[0, 1])
    kscov.append(_s)
kscov = np.stack(kscov)

# plot the results
f, ax = plt.subplots(1, 3, figsize=(6, 2))

# plot decoding figure
ax[0].scatter(X[0, :, 0], X[0, :, 1], s=5, label='Stim. a')
ax[0].scatter(X[1, :, 0], X[1, :, 1], s=5, label='Stim. b')
ax[0].set_xlabel(r"$n_1$ spike counts")
ax[0].set_ylabel(r"$n_2$ spike counts")
ax[0].axis('equal')

# plot shuffled distro for k=k in decoding figure
ax[1].hist(scov, bins=np.arange(-0.5, 0.5, 0.05), histtype='stepfilled', edgecolor='k', alpha=0.5, label='Shuff.')
ax[1].set_xlim((-0.5, 0.5))
ax[1].axvline(cov[0, 1], linestyle='-', color='r', lw=2, label='True value')
ax[1].set_title(r"$k=%s, \sigma^2=%s$" % (str(k), str(round(np.var(scov, ddof=1), 3))))
ax[1].set_xlabel(r"$cov(n_1, n_2)$")

# plot range of shuffle var for different values of k
ax[2].plot(krange, kscov.var(axis=-1, ddof=1))
ax[2].axhline(0, linestyle='--', color='k')
ax[2].set_ylabel(r"Shuff. $\sigma^2$")
ax[2].set_xlabel(r"No. of obs. ($k$)")

f.tight_layout()

plt.show()

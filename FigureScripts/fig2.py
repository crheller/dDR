"""
Simualtion of e1 estimation vs. individual covariance element estimation
for a large (N=100) neuron population with low-D noise structure.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

np.random.seed(123)

savefig = True
fig_name = os.path.join(os.getcwd(), 'figures/fig2.svg')

# Generate high-D data with low dimensional structure
nsamps = 200
nUnits = 100
sf = 2 # scale lv magnitude
u = np.zeros(nUnits)
lv = np.random.normal(0, 1, (nUnits, 1)) # low-dimensional LV
lv /= np.linalg.norm(lv)
lv = sf * lv
cov = lv.dot(lv.T)
cov += np.random.normal(0, 0.1, cov.shape) # add small amount of random noise
cov = cov.dot(cov.T) # force positive, semi-definite

X = np.random.multivariate_normal(u, cov, nsamps)

# Get eigenvalues / eigenvectors of the covariance matrix and sort
evals, evecs = np.linalg.eig(cov)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

# get goodness of fit for eigenvector / single covariance value
# across sample sizes. For each sample size, draw nSamples to get 
# sense of the variance in the estimate for each k
e1_sim = []
cov_val = []
nSamples = 100
krange = np.arange(10, 150, 2)
for ii, k in enumerate(krange):
    print(f"Iteration {ii}/{len(krange)}")
    _e1 = []
    _cov = []
    for i in range(nSamples):
        x = np.random.multivariate_normal(u, cov, k)
        _cov.append(np.cov(x)[0, 1])
        _evals, _evecs = np.linalg.eig(np.cov(x.T))
        _e1.append(abs(_evecs[:, np.argmax(_evals)].dot(evecs[:, 0])))
    e1_sim.append(_e1)
    cov_val.append(_cov)

e1_sim = np.stack(e1_sim)
cov_val = np.stack(cov_val)

# Make figure
f, ax = plt.subplots(1, 3, figsize=(7.2, 2.1))


# Covariance matrix - TODO gridspec colorbar
im = ax[0].imshow(cov, aspect='auto', cmap='Reds')
[s.set_visible(False) for s in ax[0].spines.values()]
[t.set_visible(False) for t in ax[0].get_xticklines()]
[t.set_visible(False) for t in ax[0].get_yticklines()]
ax[0].set_title(r"$\Sigma$")
ax[0].set_xlabel("Neuron")
ax[0].set_ylabel("Neuron")
f.colorbar(im, ax=ax[0])

# scree plot
ax[1].plot(evals / sum(evals), '-o', markersize=3, lw=0.5, color='grey')
ax[1].set_ylabel(r"Fraction var. exp.")
ax[1].set_xlabel(r"Principal Component")

# variance of evec alignment, variance of cov[0, 1], evec similarity (on twinx)
ax[2].plot(krange, cov_val.mean(axis=-1))
ax[2].axhline(cov[0, 1], linestyle='--', color='tab:blue')
ax[2].set_ylabel(r"Sample $\Sigma_{0, 1}$", color='tab:blue')
ax[2].set_xlabel(r"Sample size ($k$)")
ax[2].tick_params(axis='y', labelcolor='tab:blue')
ax[2].set_ylim((-0.3, 0.3))

ax2 = ax[2].twinx()
ax2.plot(krange, e1_sim.mean(axis=-1), color='orchid')
ax2.spines['right'].set_visible(True)
ax2.set_ylabel("Cosine similarity\n(True $e_1$ vs. sampled)", color='orchid')
ax2.axhline(1, linestyle='--', color='orchid')
ax2.tick_params(axis='y', labelcolor='orchid')
ax2.set_ylim((0, 1.05))
f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()
"""
Simualtion of e1 estimation vs. individual covariance element estimation
for a N=100 neuron population with low-D noise structure.

Compare independent noise to 1-D noise to 1/n noise -- extreme, extreme, real-ish?
"""
import dDR.utils.surrogate_helpers as sh

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

np.random.seed(123)

savefig = False
fig_name = os.path.join(os.getcwd(), 'figures/fig3.svg')

# Generate high-D data with variable noise structure
nUnits = 100
u = np.zeros(nUnits)

# generate full rank set of eigenvectors
evecsa = np.concatenate([sh.generate_lv_loading(nUnits, mean_loading=0.4, variance=1, mag=1) for i in range(nUnits)], axis=1)
evecsa = sh.orthonormal(evecsa)
evecsa *= 10

# DATASET 1: independent noise (diag cov matrix)
svs = np.append(1, 0.3 / np.arange(2, nUnits+1)**(1/2))
svs = (svs / sum(svs)) * 10
cov = np.zeros(evecsa.shape)
np.fill_diagonal(cov, np.diag(sh.generate_full_rank_cov(evecsa * svs)))
# Get eigenvalues / eigenvectors of the covariance matrix and sort
evals, evecs = np.linalg.eig(cov)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

# DATASET 2: 1-D noise, not aligned (one big dimension, all others small)
svs = np.append(1, 0.3 / np.arange(2, nUnits+1)**(1/2))
svs = (svs / sum(svs)) * 10
cov1 = sh.generate_full_rank_cov(evecsa * svs)
# Get eigenvalues / eigenvectors of the covariance matrix and sort
evals1, evecs1 = np.linalg.eig(cov1)
idx = np.argsort(evals1)[::-1]
evals1 = evals1[idx]
evecs1 = evecs1[:, idx]

# DATASET 3: 1/n noise
svs = 1/np.arange(1, nUnits+1)**(1/2)
svs = (svs / sum(svs)) * 10
cov2 = sh.generate_full_rank_cov(evecsa * svs)
# Get eigenvalues / eigenvectors of the covariance matrix and sort
evals2, evecs2 = np.linalg.eig(cov2)
idx = np.argsort(evals2)[::-1]
evals2 = evals2[idx]
evecs2 = evecs2[:, idx]

# decay in covariance with samples depends on the covariance value (and indep. var)
# so, for sake of comparison, find an index that matches across all three datasets
ccexample = 0.04
cidx = tuple(np.argwhere(abs(cov-ccexample) == np.min(abs(cov-ccexample)))[0])
c1idx = tuple(np.argwhere(abs(cov1-ccexample) == np.min(abs(cov1-ccexample)))[0])
cidx = c1idx
c2idx = tuple(np.argwhere(abs(cov2-ccexample) == np.min(abs(cov2-ccexample)))[0])
# get goodness of fit for eigenvector / a single covariance value
# across sample sizes. For each sample size, draw nSamples to get 
# sense of the variance in the estimate for each k
e1_sim = []
e1_sim1 = []
e1_sim2 = []
cov_val = []
cov_val1 = []
cov_val2 = []
nSamples = 100
krange = np.arange(10, 150, 2)
for ii, k in enumerate(krange):
    print(f"Iteration {ii}/{len(krange)}")
    _e1 = []
    _e11 = []
    _e12 = []
    _cov = []
    _cov1  = []
    _cov2 = []
    for i in range(nSamples):
        x = np.random.multivariate_normal(u, cov, k)
        _cov.append(np.cov(x.T)[cidx])
        _evals, _evecs = np.linalg.eig(np.cov(x.T))
        _e1.append(abs(_evecs[:, np.argmax(_evals)].dot(evecs[:, 0])))

        x1 = np.random.multivariate_normal(u, cov1, k)
        _cov1.append(np.cov(x1.T)[c1idx])
        _evals, _evecs = np.linalg.eig(np.cov(x1.T))
        _e11.append(abs(_evecs[:, np.argmax(_evals)].dot(evecs1[:, 0])))

        x2 = np.random.multivariate_normal(u, cov2, k)
        _cov2.append(np.cov(x2.T)[c2idx])
        _evals, _evecs = np.linalg.eig(np.cov(x2.T))
        _e12.append(abs(_evecs[:, np.argmax(_evals)].dot(evecs2[:, 0])))

    e1_sim.append(_e1)
    e1_sim1.append(_e11)
    e1_sim2.append(_e12)
    cov_val.append(_cov)
    cov_val1.append(_cov1)
    cov_val2.append(_cov2)

e1_sim = np.stack(e1_sim)
e1_sim1 = np.stack(e1_sim1)
e1_sim2 = np.stack(e1_sim2)
cov_val = np.stack(cov_val)
cov_val1 = np.stack(cov_val1)
cov_val2 = np.stack(cov_val2)

# Make figure
f, ax = plt.subplots(2, 2, figsize=(4.5, 4))
ax = ax.flatten()

im = ax[0].imshow(cov2, aspect='auto', cmap='bwr', vmin=-1, vmax=1)
[s.set_visible(False) for s in ax[0].spines.values()]
[t.set_visible(False) for t in ax[0].get_xticklines()]
[t.set_visible(False) for t in ax[0].get_yticklines()]
ax[0].set_title(r"$\Sigma$")
ax[0].set_xlabel("Neuron")
ax[0].set_ylabel("Neuron")
f.colorbar(im, ax=ax[0])

# scree plot for each dataset
cmap = cm.get_cmap('Greens_r', 100)
c1 = 'tab:blue' #cmap(10)
c2 = 'tab:orange' #cmap(30)
c3 = 'tab:green' #cmap(60)
ax[1].plot(np.arange(1, nUnits+1), evals1 / evals1.sum(), '.-', markersize=10, markerfacecolor='white', lw=1, color=c2, label='1-D')
ax[1].plot(np.arange(1, nUnits+1), evals / evals.sum(), '.-', markersize=10, markerfacecolor='white', lw=1, color=c3, label='Indep.')
ax[1].plot(np.arange(1, nUnits+1), evals2 / evals2.sum(), '.-', markersize=10, markerfacecolor='white', lw=1, color=c1, label='1/n')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_ylabel(r"Fraction noise var. exp.")
ax[1].set_xlabel(r"Principal Component ($\mathbf{e}_1$ - $\mathbf{e}_N$)")
ax[1].legend(frameon=False)
ax[1].set_ylim((None, 1.1))

ax[2].plot(krange, e1_sim2.mean(axis=-1), color=c1)
ax[2].fill_between(krange, e1_sim2.mean(axis=-1)-e1_sim2.std(axis=-1) / np.sqrt(nSamples),
                            e1_sim2.mean(axis=-1)+e1_sim2.std(axis=-1) / np.sqrt(nSamples), color=c1, alpha=0.5, lw=0)
ax[2].plot(krange, e1_sim1.mean(axis=-1), color=c2)
ax[2].fill_between(krange, e1_sim1.mean(axis=-1)-e1_sim1.std(axis=-1) / np.sqrt(nSamples),
                            e1_sim1.mean(axis=-1)+e1_sim1.std(axis=-1) / np.sqrt(nSamples), color=c2, alpha=0.5, lw=0)
ax[2].plot(krange, e1_sim.mean(axis=-1), color=c3)
ax[2].fill_between(krange, e1_sim.mean(axis=-1)-e1_sim.std(axis=-1) / np.sqrt(nSamples),
                            e1_sim.mean(axis=-1)+e1_sim.std(axis=-1) / np.sqrt(nSamples), color=c3, alpha=0.5, lw=0)
ax[2].set_ylabel("Cosine similarity\n"+r"(True $\mathbf{e}_1$ vs. estimated)")
ax[2].set_xlabel(r"Sample size ($k$)")
ax[2].axhline(1, linestyle='--', color='k', zorder=-1)
ax[2].set_ylim((0, 1.05))

# variance of cov[i, j]
ax[3].plot(krange, cov_val2.var(axis=-1), color=c1)
ax[3].plot(krange, cov_val1.var(axis=-1), color=c2)
ax[3].plot(krange, cov_val.var(axis=-1), color=c3)
ax[3].text(int(len(krange)/2), ax[3].get_ylim()[-1]/2, r"$\Sigma_{i,j}=%s$" % str(ccexample))
ax[3].set_ylabel(r"$Var(\hat{\Sigma}_{i, j})$")
ax[3].set_xlabel(r"Sample size ($k$)")

f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()
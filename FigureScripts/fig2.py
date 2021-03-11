"""
Simualtion of e1 estimation vs. individual covariance element estimation
for a large (N=100) neuron population with low-D noise structure.
"""
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
fig_name = os.path.join(os.getcwd(), 'figures/fig2.svg')

# Generate high-D data with random noise structure
nsamps = 200
nUnits = 100
u = np.zeros(nUnits)
lv = np.random.normal(0, 1, (nUnits, 1))
sf = 0.5 # scale lv magnitude
lv /= np.linalg.norm(lv) # low-dimensional LV
lv = sf * lv
cov = lv.dot(lv.T)
cov += np.random.normal(0, 0.1, cov.shape) # add small amount of random noise
cov = cov.dot(cov.T) # force positive, semi-definite
# Get eigenvalues / eigenvectors of the covariance matrix and sort
evals, evecs = np.linalg.eig(cov)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

# Generate high-D data with low dimensional structure (mag 1)
sf = 1.5 # scale lv magnitude
lv /= np.linalg.norm(lv) # low-dimensional LV
lv = sf * lv
cov1 = lv.dot(lv.T)
cov1 += np.random.normal(0, 0.1, cov1.shape) # add small amount of random noise
cov1 = cov1.dot(cov1.T) # force positive, semi-definite
# Get eigenvalues / eigenvectors of the covariance matrix and sort
evals1, evecs1 = np.linalg.eig(cov1)
idx = np.argsort(evals1)[::-1]
evals1 = evals1[idx]
evecs1 = evecs1[:, idx]

# Generate high-D data with low dimensional structure (mag2)
sf = 2 # scale lv magnitude
lv /= np.linalg.norm(lv)
lv = sf * lv
cov2 = lv.dot(lv.T)
cov2 += np.random.normal(0, 0.1, cov2.shape) # add small amount of random noise
cov2 = cov2.dot(cov2.T) # force positive, semi-definite
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
f, ax = plt.subplots(1, 4, figsize=(7.2, 1.75))

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
ax[1].plot(evals2 / sum(evals2), '.-', markersize=3, lw=1, color=cmap(10))
ax[1].plot(evals1 / sum(evals2), '.-', markersize=3, lw=1, color=cmap(30))
ax[1].plot(evals / sum(evals), '.-', markersize=3, lw=1, color=cmap(60))
ax[1].set_ylabel(r"Fraction var. exp.")
ax[1].set_xlabel(r"Principal Component ($\mathbf{e}_1$ - $\mathbf{e}_N$)")

ax[2].plot(krange, e1_sim2.mean(axis=-1), color=cmap(10))
ax[2].fill_between(krange, e1_sim2.mean(axis=-1)-e1_sim2.std(axis=-1) / np.sqrt(nSamples),
                            e1_sim2.mean(axis=-1)+e1_sim2.std(axis=-1) / np.sqrt(nSamples), color=cmap(10), alpha=0.5, lw=0)
ax[2].plot(krange, e1_sim1.mean(axis=-1), color=cmap(30))
ax[2].fill_between(krange, e1_sim1.mean(axis=-1)-e1_sim1.std(axis=-1) / np.sqrt(nSamples),
                            e1_sim1.mean(axis=-1)+e1_sim1.std(axis=-1) / np.sqrt(nSamples), color=cmap(30), alpha=0.5, lw=0)
ax[2].plot(krange, e1_sim.mean(axis=-1), color=cmap(60))
ax[2].fill_between(krange, e1_sim.mean(axis=-1)-e1_sim.std(axis=-1) / np.sqrt(nSamples),
                            e1_sim.mean(axis=-1)+e1_sim.std(axis=-1) / np.sqrt(nSamples), color=cmap(60), alpha=0.5, lw=0)
ax[2].set_ylabel("Cosine similarity\n"+r"(True $\mathbf{e}_1$ vs. sampled)")
ax[2].set_xlabel(r"Sample size ($k$)")
ax[2].axhline(1, linestyle='--', color='k')
ax[2].set_ylim((0, 1.05))

# variance of cov[0, 1], evec similarity (on twinx)
ax[3].plot(krange, cov_val2.var(axis=-1), color=cmap(10))
ax[3].plot(krange, cov_val1.var(axis=-1), color=cmap(30))
ax[3].plot(krange, cov_val.var(axis=-1), color=cmap(60))
ax[3].text(int(len(krange)/2), ax[3].get_ylim()[-1]-0.05, r"$\Sigma_{0,1}=%s$" % str(ccexample))
ax[3].set_ylabel(r"$Var(\Sigma_{0, 1})$")
ax[3].set_xlabel(r"Sample size ($k$)")

f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()
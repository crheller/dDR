"""
Information limiting correlations. Information saturates due to small,
information limiting covariance.
"""

"""
Illustrate efficacy of dimensionality reduction techinque in preserving information 
about small, differential correlations. Also illustrate ability of technique to be 
robust to overfitting, and work in low trial N regime.

Simulate a population with two noise PCs - one that projects along 
dU (differential) and another that is orthogonal (randomly).
Simulate for many (1000s) neurons, then run decoding for different subsets and 
show that information saturates as neurons are added.
"""
from dDR.utils.decoding import compute_dprime
from dDR.dDR import dDR
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

np.random.seed(123)

savefig = True
fig_name = os.path.join(os.getcwd(), 'figures/fig5.svg')

# data/sampling params
Ndim = 1000
maxDim = 1000
ksmall = 500
klarge = 10000
step = 50
RandSubsets = 50
var_ratio = 1.2 # pc1 has X times the variance as pc2

u1 = 4
u2 = 4
u = np.stack((np.random.poisson(u1, Ndim), np.random.poisson(u2, Ndim)))

# make two dimensional noise:
# one large dim ~orthogonal to dU and one smaller dim ~ parallel to dU
dU = u[[1], :] - u[[0], :]
dU = dU / np.linalg.norm(dU)

diff_cor = dU #+ np.random.normal(0, 0.0001, dU.shape)
diff_cor = diff_cor / np.linalg.norm(diff_cor) * 1.5
pc1 = np.random.normal(0, 1, dU.shape)
pc1 = (pc1 / np.linalg.norm(pc1)) * 2  * var_ratio

evecs = np.concatenate((diff_cor, pc1), axis=0)
cov = evecs.T.dot(evecs)
cov += np.random.normal(0, 0.005, cov.shape)
cov = cov.dot(cov.T)

# ========================================== low trial number example ============================================
Ntrials = ksmall

# simulate full data matrix
_X = np.random.multivariate_normal(np.zeros(Ndim), cov, Ntrials)
X1 = _X + u[0, :]
X2 = _X + u[1, :]
X_raw = np.stack((X1, X2)).transpose([-1, 1, 0])

# add random noise to data matrix
X_raw += np.random.normal(0, 0.5, X_raw.shape)

# get evals for raw data
evals_ksmall, evecs_ksmall = np.linalg.eig(np.cov(X_raw[:, :, 0]))
idx = np.argsort(evals_ksmall)[::-1]
evals_ksmall = evals_ksmall[idx]
evecs_ksmall = evecs_ksmall[:, idx]

n_subsets = np.append([2], np.arange(step, maxDim, step))

# get est/val indexes (can be the same for each subset of neurons)
eidx = np.random.choice(range(X_raw.shape[1]), int(X_raw.shape[1]/2), replace=False)
tidx = np.array(list(set(np.arange(X_raw.shape[1])).difference(set(eidx))))
dp_ddr_ksmall = []
dp_full_ksmall = []
for nset in n_subsets:
    print('nset: {}'.format(nset))
    _dp_full = []
    _dp_ddr = []
    for ii in range(RandSubsets):
        # choose random subset of neurons
        neurons = np.random.choice(np.arange(0, Ndim), nset, replace=False)
        X = X_raw[neurons, :, :]
        Xest = X[:, eidx]
        Xval = X[:, tidx]

        # full rank data
        try:
            r = compute_dprime(Xest[:, :, 0], Xest[:, :, 1])
            r = compute_dprime(Xval[:, :, 0], Xval[:, :, 1], wopt=r.wopt)
            _dp_full.append(r.dprimeSquared)
        except ValueError:
            # not enough reps
            _dp_full.append(np.nan)

        # dDR
        ddr = dDR()
        ddr.fit(Xest[:, :, 0].T, Xest[:, :, 1].T)
        Xest_ddr1 = ddr.transform(Xest[:, :, 0].T)
        Xest_ddr2 = ddr.transform(Xest[:, :, 1].T)
        Xval_ddr1 = ddr.transform(Xval[:, :, 0].T)
        Xval_ddr2 = ddr.transform(Xval[:, :, 1].T)

        r = compute_dprime(Xest_ddr1.T, Xest_ddr2.T)
        r = compute_dprime(Xval_ddr1.T, Xval_ddr2.T, wopt=r.wopt)

        _dp_ddr.append(r.dprimeSquared)
    
    dp_ddr_ksmall.append(_dp_ddr)
    dp_full_ksmall.append(_dp_full)

dp_ddr_ksmall = np.stack(dp_ddr_ksmall)
dp_full_ksmall = np.stack(dp_full_ksmall)

# ========================================== high trial number example ============================================
Ntrials = klarge

# simulate full data matrix
_X = np.random.multivariate_normal(np.zeros(Ndim), cov, Ntrials)
X1 = _X + u[0, :]
X2 = _X + u[1, :]
X_raw = np.stack((X1, X2)).transpose([-1, 1, 0])

# add random noise to data matrix
X_raw += np.random.normal(0, 0.5, X_raw.shape)

# get evals for raw data
evals_klarge, evecs_klarge = np.linalg.eig(np.cov(X_raw[:, :, 0]))
idx = np.argsort(evals_klarge)[::-1]
evals_klarge = evals_klarge[idx]
evecs_klarge = evecs_klarge[:, idx]

n_subsets = np.append([2], np.arange(step, maxDim, step))

# get est/val indexes (can be the same for each subset of neurons)
eidx = np.random.choice(range(X_raw.shape[1]), int(X_raw.shape[1]/2), replace=False)
tidx = np.array(list(set(np.arange(X_raw.shape[1])).difference(set(eidx))))
dp_ddr_klarge = []
dp_full_klarge = []
for nset in n_subsets:
    print('nset: {}'.format(nset))
    _dp_full = []
    _dp_ddr = []
    for ii in range(RandSubsets):
        # choose random subset of neurons
        neurons = np.random.choice(np.arange(0, Ndim), nset, replace=False)
        X = X_raw[neurons, :, :]
        Xest = X[:, eidx]
        Xval = X[:, tidx]

        # full rank data
        try:
            r = compute_dprime(Xest[:, :, 0], Xest[:, :, 1])
            r = compute_dprime(Xval[:, :, 0], Xval[:, :, 1], wopt=r.wopt)
            _dp_full.append(r.dprimeSquared)
        except ValueError:
            # not enough reps
            _dp_full.append(np.nan)

        # dDR
        ddr = dDR()
        ddr.fit(Xest[:, :, 0].T, Xest[:, :, 1].T)
        Xest_ddr1 = ddr.transform(Xest[:, :, 0].T)
        Xest_ddr2 = ddr.transform(Xest[:, :, 1].T)
        Xval_ddr1 = ddr.transform(Xval[:, :, 0].T)
        Xval_ddr2 = ddr.transform(Xval[:, :, 1].T)

        r = compute_dprime(Xest_ddr1.T, Xest_ddr2.T)
        r = compute_dprime(Xval_ddr1.T, Xval_ddr2.T, wopt=r.wopt)

        _dp_ddr.append(r.dprimeSquared)
    
    dp_ddr_klarge.append(_dp_ddr)
    dp_full_klarge.append(_dp_full)

dp_ddr_klarge = np.stack(dp_ddr_klarge)
dp_full_klarge = np.stack(dp_full_klarge)


# =========================================================== plot results ===========================================================

# use full rank data matrix, highest trial number, to determine the approximate "peak" information
norm1 = np.nanmax(np.concatenate((dp_full_klarge, dp_full_klarge)))
norm2 = np.nanmax(np.concatenate((dp_ddr_ksmall, dp_ddr_ksmall)))

dp_ddr_klarge_plot = dp_ddr_klarge / norm1
dp_ddr_ksmall_plot = dp_ddr_ksmall / norm2
dp_full_klarge_plot = dp_full_klarge / norm1
dp_full_ksmall_plot = dp_full_ksmall / norm2

f, ax = plt.subplots(2, 3, figsize=(6.8, 4))

# high sample size results
ax[0, 0].plot(n_subsets, dp_ddr_klarge_plot.mean(axis=-1), label=r"$dDR$", color='tab:blue')
ax[0, 0].fill_between(n_subsets, dp_ddr_klarge_plot.mean(axis=-1)-dp_ddr_klarge_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         dp_ddr_klarge_plot.mean(axis=-1)+dp_ddr_klarge_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         color='tab:blue', alpha=0.5, lw=0)
ax[0, 0].plot(n_subsets, dp_full_klarge_plot.mean(axis=-1), label="Full rank data", color='tab:orange')
ax[0, 0].fill_between(n_subsets, dp_full_klarge_plot.mean(axis=-1)-dp_full_klarge_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         dp_full_klarge_plot.mean(axis=-1)+dp_full_klarge_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         color='tab:orange', alpha=0.5, lw=0)
ax[0, 0].set_ylim((-0.1, 1.1))
ax[0, 0].set_xlabel(r'Number of neurons ($N$)')
ax[0, 0].set_ylabel(r"cross-validated $d'^2$"+"\n(norm. to peak)")
ax[0, 0].set_title(r"$N_{tot}=%s$, $k=%s$"%(str(Ndim), str(klarge)))
ax[0, 0].legend(frameon=False)

idx = np.argmax(abs(evecs_klarge.T.dot(dU.T)))
ax[0, 1].plot(evals_klarge / sum(evals_klarge), color='grey')
ax[0, 1].plot(idx, (evals_klarge / sum(evals_klarge))[idx], 'o', color='k', markersize=3)
ax[0, 1].set_xlabel(r"Prinicpal components ($e_1 - e_N$)")
ax[0, 1].set_ylabel("Fraction var. explained")
ax[0, 1].set_title("Scree plot")

ax[0, 2].plot(abs(evecs_klarge.T.dot(dU.T)), color='grey')
ax[0, 2].plot(idx, (abs(evecs_klarge.T.dot(dU.T)))[idx], 'o', color='k', markersize=3)
ax[0, 2].set_xlabel(r"Prinicpal components ($e_1 - e_N$)")
ax[0, 2].set_ylabel("Cosine similarity"+"\n"+r"($cos(\theta_{\Delta \mu, e_{n}})$)")
ax[0, 2].set_title("Signal vs. noise similarity")
ax[0, 2].set_ylim((-0.1, 1.1))

# small sample size results
ax[1, 0].plot(n_subsets, dp_ddr_ksmall_plot.mean(axis=-1), label=r"$dDR$", color='tab:blue')
ax[1, 0].fill_between(n_subsets, dp_ddr_ksmall_plot.mean(axis=-1)-dp_ddr_ksmall_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         dp_ddr_ksmall_plot.mean(axis=-1)+dp_ddr_ksmall_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         color='tab:blue', alpha=0.5, lw=0)
ax[1, 0].plot(n_subsets, dp_full_ksmall_plot.mean(axis=-1), label="Full rank data", color='tab:orange')
ax[1, 0].fill_between(n_subsets, dp_full_ksmall_plot.mean(axis=-1)-dp_full_ksmall_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         dp_full_ksmall_plot.mean(axis=-1)+dp_full_ksmall_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         color='tab:orange', alpha=0.5, lw=0)
ax[1, 0].set_ylim((-0.1, 1.1))
ax[1, 0].set_xlabel(r'Number of neurons ($N$)')
ax[1, 0].set_ylabel(r"cross-validated $d'^2$"+"\n(norm. to peak)")
ax[1, 0].set_title(r"$N_{tot}=%s$, $k=%s$"%(str(Ndim), str(ksmall)))

idx = np.argmax(abs(evecs_ksmall.T.dot(dU.T)))
ax[1, 1].plot(evals_ksmall / sum(evals_ksmall), color='grey')
ax[1, 1].plot(idx, (evals_ksmall / sum(evals_ksmall))[idx], 'o', color='k', markersize=3)
ax[1, 1].set_xlabel(r"Prinicpal components ($e_1 - e_N$)")
ax[1, 1].set_ylabel("Fraction var. explained")
ax[1, 1].set_title("Scree plot")

ax[1, 2].plot(abs(evecs_ksmall.T.dot(dU.T)), color='grey')
ax[1, 2].plot(idx, (abs(evecs_ksmall.T.dot(dU.T)))[idx], 'o', color='k', markersize=3)
ax[1, 2].set_xlabel(r"Prinicpal components ($e_1 - e_N$)")
ax[1, 2].set_ylabel("Cosine similarity"+"\n"+r"($cos(\theta_{\Delta \mu, e_{n}})$)")
ax[1, 2].set_title("Signal vs. noise similarity")
ax[1, 2].set_ylim((-0.1, 1.1))

f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()
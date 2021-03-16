"""
Remaking figure 5 in a slightly new way, also using new noise generation tools, and adding PCA.

Don't *think* we need to do small vs. large trial numbers here. Can just do small(ish) and show
that information still saturates when noise aligned (can cite lots of stuff) but increases linearly 
when not aligned. Same eigenspectrum in each case. (use 1/f cov matrix as in fig 4)

Interesting comparison point... when information limiting corr. are BIG, PCA can find them quickly, even if dU
is small. But, when information limiting are small and/or dU is small relative to noise, 
PCA really struggles, while dDR does not.

For figure:
    information limiting vs. non information limiting (linear vs. saturate) -- pretty big dU, so PCA does okay.
    Then, make small information limiting dim and small dU -- PCA should fail, while dDR is fine.
    plot scree plots as scatter with color indicating the alignment with the coding dimension, delta mu
    
# 1000s of trials in averbeck study x ~500 neurons, so this would show don't need that. 
# they had weak correlations -- mean 0.005, sd=0.069
"""
from dDR.utils.decoding import compute_dprime
import dDR.utils.surrogate_helpers as sh
from dDR.PCA import PCA
from dDR.dDR import dDR
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

np.random.seed(123)

savefig = False
fig_name = os.path.join(os.getcwd(), 'figures/fig5a.svg')

# data/sampling params
nUnits = 1000
maxDim = 1000
k = 100
step = 200  #50
RandSubsets = 10 #50

n_subsets = np.append([2], np.arange(step, maxDim, step))

# define mean response to each stimulus
duvar = 0.5
u1 = np.random.normal(4, duvar, Ndim)
u2 = np.random.normal(4, duvar, Ndim)
u = np.stack((u1, u2))

# make the covariance matrices
dU = u[[1], :] - u[[0], :]
dU = dU / np.linalg.norm(dU)

# with information limiting noise
lv = dU.T
evecsA = np.concatenate([sh.generate_lv_loading(Ndim, mean_loading=0, variance=1, mag=1) for i in range(nUnits-1)], axis=1)
evecsA = np.concatenate((lv, evecsA), axis=1)
evecsA = sh.orthonormal(evecsA)
evecsA *= 10
svs = 1 / np.arange(1, Ndim+1)**(1/2)
svs[0] = svs[1] # bump information limiting noise to a smaller dimension (e.g. Rumyantsev, Kafashan)
svs[1] = 1
covlim = sh.generate_full_rank_cov(evecsA * svs)

# without information limiting noise
evecs = np.concatenate([sh.generate_lv_loading(nUnits, mean_loading=0, variance=1, mag=1) for i in range(nUnits)], axis=1)
evecs = sh.orthonormal(evecs)
evecs *= 10
svs = 1 / np.arange(1, Ndim+1)**(1/2)
cov = sh.generate_full_rank_cov(evecs * svs)

# ========================================== subsample neurons ============================================
# simulate full data matrix w/o information limiting corr.
X_raw = np.random.multivariate_normal(np.zeros(nUnits), cov, k)
X1 = X_raw + u[0, :]
X2 = X_raw + u[1, :]
X_raw = np.stack((X1, X2)).transpose([-1, 1, 0])
# with information limiting corr.
Xlim = np.random.multivariate_normal(np.zeros(nUnits), covlim, k)
X1 = Xlim + u[0, :]
X2 = Xlim + u[1, :]
Xlim = np.stack((X1, X2)).transpose([-1, 1, 0])

# get est/val indexes (can be the same for each subset of neurons)
eidx = np.random.choice(range(X_raw.shape[1]), int(X_raw.shape[1]/2), replace=False)
tidx = np.array(list(set(np.arange(X_raw.shape[1])).difference(set(eidx))))
dp_ddr_lim = []
dp_pca_lim = []
dp_ddr = []
dp_pca = []
for nset in n_subsets:
    print('nset: {}'.format(nset))
    _dp_ddr_lim = []
    _dp_pca_lim = []
    _dp_ddr = []
    _dp_pca = []
    for ii in range(RandSubsets):
        # choose random subset of neurons
        neurons = np.random.choice(np.arange(0, Ndim), nset, replace=False)

        # w/o information limiting correlations
        X = X_raw[neurons, :, :]
        Xest = X[:, eidx]
        Xval = X[:, tidx]

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

        # PCA
        pca = PCA(n_components=2)
        pca.fit(np.concatenate((Xest[:, :, 0].T, Xest[:, :, 1].T), axis=0))
        Xest_pca1 = pca.transform(Xest[:, :, 0].T)
        Xest_pca2 = pca.transform(Xest[:, :, 1].T)
        Xval_pca1 = pca.transform(Xval[:, :, 0].T)
        Xval_pca2 = pca.transform(Xval[:, :, 1].T)

        r = compute_dprime(Xest_pca1.T, Xest_pca2.T)
        r = compute_dprime(Xval_pca1.T, Xval_pca2.T, wopt=r.wopt)

        _dp_pca.append(r.dprimeSquared)

        # with information limiting correlations
        X = Xlim[neurons, :, :]
        Xest = X[:, eidx]
        Xval = X[:, tidx]

        # dDR
        ddr = dDR()
        ddr.fit(Xest[:, :, 0].T, Xest[:, :, 1].T)
        Xest_ddr1 = ddr.transform(Xest[:, :, 0].T)
        Xest_ddr2 = ddr.transform(Xest[:, :, 1].T)
        Xval_ddr1 = ddr.transform(Xval[:, :, 0].T)
        Xval_ddr2 = ddr.transform(Xval[:, :, 1].T)

        r = compute_dprime(Xest_ddr1.T, Xest_ddr2.T)
        r = compute_dprime(Xval_ddr1.T, Xval_ddr2.T, wopt=r.wopt)

        _dp_ddr_lim.append(r.dprimeSquared)

        # PCA
        pca = PCA(n_components=2)
        pca.fit(np.concatenate((Xest[:, :, 0].T, Xest[:, :, 1].T), axis=0))
        Xest_pca1 = pca.transform(Xest[:, :, 0].T)
        Xest_pca2 = pca.transform(Xest[:, :, 1].T)
        Xval_pca1 = pca.transform(Xval[:, :, 0].T)
        Xval_pca2 = pca.transform(Xval[:, :, 1].T)

        r = compute_dprime(Xest_pca1.T, Xest_pca2.T)
        r = compute_dprime(Xval_pca1.T, Xval_pca2.T, wopt=r.wopt)

        _dp_pca_lim.append(r.dprimeSquared)

    
    dp_ddr_lim.append(_dp_ddr_lim)
    dp_pca_lim.append(_dp_pca_lim)
    dp_ddr.append(_dp_ddr)
    dp_pca.append(_dp_pca)

dp_ddr_lim = np.stack(dp_ddr_lim)
dp_pca_lim = np.stack(dp_pca_lim)
dp_ddr = np.stack(dp_ddr)
dp_pca = np.stack(dp_pca)

# =========================================================== plot results ===========================================================

# use full rank data matrix, highest trial number, to determine the approximate "peak" information
norm1 = np.nanmax(np.concatenate((dp_full_klarge, dp_full_klarge)))
norm2 = np.nanmax(np.concatenate((dp_ddr_ksmall, dp_ddr_ksmall)))

dp_ddr_klarge_plot = dp_ddr_klarge / norm1
dp_ddr_ksmall_plot = dp_ddr_ksmall / norm2
dp_full_klarge_plot = dp_full_klarge / norm1
dp_full_ksmall_plot = dp_full_ksmall / norm2
dp_pca_klarge_plot = dp_pca_klarge / norm1
dp_pca_ksmall_plot = dp_pca_ksmall / norm2

f, ax = plt.subplots(2, 3, figsize=(6.8, 4))

# high sample size results
ax[0, 0].plot(n_subsets, dp_ddr_klarge_plot.mean(axis=-1), label=r"$dDR$", color='tab:blue')
ax[0, 0].fill_between(n_subsets, dp_ddr_klarge_plot.mean(axis=-1)-dp_ddr_klarge_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         dp_ddr_klarge_plot.mean(axis=-1)+dp_ddr_klarge_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         color='tab:blue', alpha=0.5, lw=0)
ax[0, 0].plot(n_subsets, dp_pca_klarge_plot.mean(axis=-1), label=r"$PCA$", color='tab:orange')
ax[0, 0].fill_between(n_subsets, dp_pca_klarge_plot.mean(axis=-1)-dp_pca_klarge_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         dp_pca_klarge_plot.mean(axis=-1)+dp_pca_klarge_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         color='tab:orange', alpha=0.5, lw=0)
ax[0, 0].plot(n_subsets, dp_full_klarge_plot.mean(axis=-1), label="Full rank data", color='tab:green')
ax[0, 0].fill_between(n_subsets, dp_full_klarge_plot.mean(axis=-1)-dp_full_klarge_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         dp_full_klarge_plot.mean(axis=-1)+dp_full_klarge_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         color='tab:green', alpha=0.5, lw=0)
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
ax[0, 1].set_xscale('log')
ax[0, 1].set_yscale('log')

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
ax[1, 0].plot(n_subsets, dp_pca_ksmall_plot.mean(axis=-1), label=r"$PCA$", color='tab:orange')
ax[1, 0].fill_between(n_subsets, dp_pca_ksmall_plot.mean(axis=-1)-dp_pca_ksmall_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         dp_pca_ksmall_plot.mean(axis=-1)+dp_pca_ksmall_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         color='tab:orange', alpha=0.5, lw=0)
ax[1, 0].plot(n_subsets, dp_full_ksmall_plot.mean(axis=-1), label="Full rank data", color='tab:green')
ax[1, 0].fill_between(n_subsets, dp_full_ksmall_plot.mean(axis=-1)-dp_full_ksmall_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         dp_full_ksmall_plot.mean(axis=-1)+dp_full_ksmall_plot.std(axis=-1) / np.sqrt(RandSubsets),
                         color='tab:green', alpha=0.5, lw=0)
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
ax[1, 1].set_xscale('log')
ax[1, 1].set_yscale('log')

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
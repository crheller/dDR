"""
Remaking figure 5 in a slightly new way, also using new noise generation tools, and adding PCA.

Don't *think* we need to do small vs. large trial numbers here. Can just do small(ish) and show
that information still saturates when noise aligned (can cite lots of stuff) but increases linearly 
when not aligned. Same eigenspectrum in each case. (use 1/f cov matrix as in fig 4)

Interesting comparison point... when information limiting corr. are BIG, PCA can find them quickly, even if dU
is small. But, when information limiting are small and/or dU is small relative to noise, 
PCA really struggles, while dDR does not.

I think we can argue 
    1 - largest noise dims not usually aligned with stim (Rumyantsev)
    2 - Therefore, interesting comparisons are where dU is large vs small / where noise is aligned / not aligned


For figure:
    information limiting vs. non information limiting (linear vs. saturate) -- pretty big dU, so PCA does okay.
    Then, make small information limiting dim and small dU -- PCA should fail, while dDR is fine.
    plot scree plots as scatter with color indicating the alignment with the coding dimension, delta mu

    twinx to show that delta u scales linearly?

    Top, scree plots with color indicating alignment? Or, overlay cos similarity on twinx?
    Left, ratio of delta mu ** 2 to each eigenvector
    Grid - dprime over neurons for PCA / dDR. Also dU**2 across neurons with twinx
    Upper left corner, cartoon with the different features?
    
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

# define mean response to each stimulus (big difference between classes)
duvar = 0.5
u1 = np.random.normal(4, duvar, nUnits)
u2 = np.random.normal(4, duvar, nUnits)
ubig = np.stack((u1, u2))

# small difference between classes
duvar = 0.3
u1 = np.random.normal(4, duvar, nUnits)
u2 = np.random.normal(4, duvar, nUnits)
usmall = np.stack((u1, u2))

# ====================== make the covariance matrices / 4 different datasets ============================
datasets = ['big_orth', 'big_aligned', 'small_orth', 'small_aligned']
X = {k: None for k in datasets}
cov = dict.fromkeys(datasets)
evscale = 10
inflimdim = 2
limvar = 0.5 # smaller number = more aligned with dU

# big / orthogonal
evecs = np.concatenate([sh.generate_lv_loading(nUnits, mean_loading=0, variance=1, mag=1) for i in range(nUnits)], axis=1)
evecs = sh.orthonormal(evecs)
evecs *= evscale
svs = 1 / np.arange(1, nUnits+1)**(1/2)
cov['big_orth'] = sh.generate_full_rank_cov(evecs * svs)
X_raw = np.random.multivariate_normal(np.zeros(nUnits), cov['big_orth'], k)
X1 = X_raw + ubig[0, :]
X2 = X_raw + ubig[1, :]
X['big_orth']= np.stack((X1, X2)).transpose([-1, 1, 0])

# big / information limiting noise
lv = sh.generate_lv_loading(nUnits, mean_loading=ubig[0]-ubig[1], variance=limvar, mag=1)
evecs = np.concatenate([sh.generate_lv_loading(nUnits, mean_loading=0, variance=1, mag=1) for i in range(nUnits-1)], axis=1)
evecs = np.concatenate((lv, evecs), axis=1)
evecs = sh.orthonormal(evecs)
evecs *= evscale
svs = 1 / np.arange(1, nUnits+1)**(1/2)
svs[0] = svs[inflimdim] # bump information limiting noise to a slightly smaller dimension (e.g. Rumyantsev, Kafashan)
svs[inflimdim] = 1
cov['big_aligned'] = sh.generate_full_rank_cov(evecs * svs)
X_raw = np.random.multivariate_normal(np.zeros(nUnits), cov['big_aligned'], k)
X1 = X_raw + ubig[0, :]
X2 = X_raw + ubig[1, :]
X['big_aligned']= np.stack((X1, X2)).transpose([-1, 1, 0])

# small / orthogonal
cov['small_orth'] = cov['big_orth']
X_raw = np.random.multivariate_normal(np.zeros(nUnits), cov['small_orth'], k)
X1 = X_raw + usmall[0, :]
X2 = X_raw + usmall[1, :]
X['small_orth']= np.stack((X1, X2)).transpose([-1, 1, 0])

# small / information limiting noise
lv = sh.generate_lv_loading(nUnits, mean_loading=usmall[0]-usmall[1], variance=limvar, mag=1)
evecs = np.concatenate([sh.generate_lv_loading(nUnits, mean_loading=0, variance=1, mag=1) for i in range(nUnits-1)], axis=1)
evecs = np.concatenate((lv, evecs), axis=1)
evecs = sh.orthonormal(evecs)
evecs *= evscale
svs = 1 / np.arange(1, nUnits+1)**(1/2)
svs[0] = svs[inflimdim] # bump information limiting noise to a slightly smaller dimension (e.g. Rumyantsev, Kafashan)
svs[inflimdim] = 1
cov['small_aligned'] = sh.generate_full_rank_cov(evecs * svs)
X_raw = np.random.multivariate_normal(np.zeros(nUnits), cov['small_aligned'], k)
X1 = X_raw + usmall[0, :]
X2 = X_raw + usmall[1, :]
X['small_aligned']= np.stack((X1, X2)).transpose([-1, 1, 0])

# =========================================================================================================

# ========================================== subsample neurons ============================================
# get est/val trial indexes (can be the same for each subset of neurons)
eidx = np.random.choice(range(k), int(k/2), replace=False)
tidx = np.array(list(set(np.arange(k)).difference(set(eidx))))
rddr = {k: [] for k in datasets}
rpca = {k: [] for k in datasets}
for nset in n_subsets:
    print('nset: {}'.format(nset))
    _results = {k: {'pca': [], 'ddr': []} for k in datasets}
    for ii in range(RandSubsets):
        # choose random subset of neurons
        neurons = np.random.choice(np.arange(0, nUnits), nset, replace=False)

        for dtype in datasets:
            _X = X[dtype][neurons, :, :]
            Xest = _X[:, eidx]
            Xval = _X[:, tidx]

            # dDR
            ddr = dDR(n_additional_axes=None)
            ddr.fit(Xest[:, :, 0].T, Xest[:, :, 1].T)
            Xest_ddr1 = ddr.transform(Xest[:, :, 0].T)
            Xest_ddr2 = ddr.transform(Xest[:, :, 1].T)
            Xval_ddr1 = ddr.transform(Xval[:, :, 0].T)
            Xval_ddr2 = ddr.transform(Xval[:, :, 1].T)

            r = compute_dprime(Xest_ddr1.T, Xest_ddr2.T)
            r = compute_dprime(Xval_ddr1.T, Xval_ddr2.T, wopt=r.wopt)

            _results[dtype]['ddr'].append(r.dprimeSquared)

            # PCA
            pca = PCA(n_components=2)
            pca.fit(np.concatenate((Xest[:, :, 0].T, Xest[:, :, 1].T), axis=0))
            Xest_pca1 = pca.transform(Xest[:, :, 0].T)
            Xest_pca2 = pca.transform(Xest[:, :, 1].T)
            Xval_pca1 = pca.transform(Xval[:, :, 0].T)
            Xval_pca2 = pca.transform(Xval[:, :, 1].T)

            r = compute_dprime(Xest_pca1.T, Xest_pca2.T)
            r = compute_dprime(Xval_pca1.T, Xval_pca2.T, wopt=r.wopt)

            _results[dtype]['pca'].append(r.dprimeSquared)

    for dtype in datasets:
        rpca[dtype].append(_results[dtype]['pca'])
        rddr[dtype].append(_results[dtype]['ddr'])

for dtype in datasets:
    rpca[dtype] = np.stack(rpca[dtype])
    rddr[dtype] = np.stack(rddr[dtype])

# =========================================================== plot results ===========================================================
# normalize data for plotting?
norm = 1

f, ax = plt.subplots(3, 3, figsize=(7, 6))

ncomp = np.arange(1, nUnits+1)

# plot scree (variance explained) plots
# ortho noise
du = (usmall[0] - usmall[1])
du /= np.linalg.norm(du)
oevals, oevecs = np.linalg.eig(cov['small_orth'])
idx = np.argsort(oevals)[::-1]
oevals = oevals[idx]
oevecs = oevecs[:, idx]
ax[0, 1].plot(ncomp, oevals / oevals.sum(), '.', color='seagreen')
ax[0, 1].set_xscale('log'); ax[0, 1].set_yscale('log')
ax[0, 1].set_xlabel(r"Principal component")
ax[0, 1].set_ylabel("Fract. noise var. exp.", color='seagreen')
ax[0, 1].tick_params(axis='y', labelcolor='seagreen')
ax[0, 1].set_ylim(10**-4, 1)
ax2 = ax[0, 1].twinx()
ax2.spines['right'].set_visible(True)
ax2.plot(ncomp, abs(oevecs.T.dot(du)), '-', color='orchid', zorder=-nUnits-1)
ax2.set_ylabel(r"Noise-signal alignment", color='orchid')
ax2.tick_params(axis='y', labelcolor='orchid')
ax2.set_ylim(0, 1)

# aligned noise
aevals, aevecs = np.linalg.eig(cov['small_aligned'])
idx = np.argsort(aevals)[::-1]
aevals = aevals[idx]
aevecs = aevecs[:, idx]
ax[0, 2].plot(ncomp, aevals / aevals.sum(), '.', color='seagreen')
ax[0, 2].set_xscale('log'); ax[0, 2].set_yscale('log')
ax[0, 2].set_xlabel(r"Principal component")
ax[0, 2].set_ylabel("Fract. noise var. exp.", color='seagreen')
ax[0, 2].tick_params(axis='y', labelcolor='seagreen')# aligned noise
ax[0, 2].set_ylim(10**-4, 1)
ax2 = ax[0, 2].twinx()
ax2.plot(ncomp, abs(aevecs.T.dot(du)), '-', color='orchid', zorder=-nUnits-1)
ax2.set_ylabel(r"Noise-signal alignment", color='orchid')
ax2.tick_params(axis='y', labelcolor='orchid')
ax2.set_ylim(0, 1)

# plot var(delta mu) vs. var(pc)
# big delta mu
du = (ubig[0] - ubig[1])
du /= np.linalg.norm(du)
bevals, bevecs = np.linalg.eig(cov['big_orth'])
idx = np.argsort(bevals)[::-1]
bevals = bevals[idx]
bevecs = bevecs[:, idx]
totvar = X['big_orth'].var(axis=(1,2)).sum()
duvar = X['big_orth'].reshape(nUnits, -1).T.dot(du).var()
ax[1, 0].plot(ncomp, bevals / totvar, '.', color='grey', label=r'$PC$s')
ax[1, 0].axhline(duvar / totvar, color='purple', label=r'$\Delta \mu$')
ax[1, 0].set_xscale('log'); ax[1, 0].set_yscale('log')
ax[1, 0].set_xlabel(r"Principal component")
ax[1, 0].set_ylabel("Fract. total var. exp.")
ax[1, 0].legend(frameon=False)
ax[1, 0].set_ylim(10**-4, 1)

# small delta mu
du = (usmall[0] - usmall[1])
du /= np.linalg.norm(du)
sevals, sevecs = np.linalg.eig(cov['small_orth'])
idx = np.argsort(sevals)[::-1]
sevals = sevals[idx]
sevecs = sevecs[:, idx]
totvar = X['small_orth'].var(axis=(1,2)).sum()
duvar = X['small_orth'].reshape(nUnits, -1).T.dot(du).var()
ax[2, 0].plot(ncomp, sevals / totvar, '.', color='grey', label=r'$PC$s')
ax[2, 0].axhline(duvar / totvar, color='purple', label=r'$\Delta \mu$')
ax[2, 0].set_xscale('log'); ax[2, 0].set_yscale('log')
ax[2, 0].set_xlabel(r"Principal component")
ax[2, 0].set_ylabel("Fract. total var. exp.")
ax[2, 0].set_ylim(10**-4, 1)


# plot dprime results
for i, (a, dtype) in enumerate(zip([ax[1, 1], ax[1, 2], ax[2, 1], ax[2, 2]], datasets)):
    if i == 0:
        lab1 = r"$PCA$"
        lab2 = r"$dDR$"
    
    a.plot(n_subsets, rddr[dtype].mean(axis=1), lw=2, color='tab:blue', label=lab2)
    a.plot(n_subsets, rpca[dtype].mean(axis=1), lw=2, color='tab:orange', label=lab1)
    a.set_xlabel(r"Neurons ($N$)")
    a.set_ylabel(r"$d'^2$")
    a.legend(frameon=False)

f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()
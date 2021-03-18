"""
Remaking figure 5 in a slightly new way, also using new noise generation tools, and adding PCA.

Figure: 3 x 3
    Top row:    aligned noise - spectrum, large mu dp, small mu dp
    Middle row: orth. noise - spectrum large mu dp, small mu dp
    Bottom row: Projections for example bin - dDR, taPCA, stPCA 
"""
from dDR.utils.decoding import compute_dprime
from dDR.utils.plotting import compute_ellipse
import dDR.utils.surrogate_helpers as sh
from dDR.PCA import PCA
from dDR.dDR import dDR
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

np.random.seed(123)

savefig = True
fig_name = os.path.join(os.getcwd(), 'figures/fig5.svg')

# data/sampling params
nUnits = 1000
maxDim = 500
k = 50
step = 50  #50
RandSubsets = 50 #50
ndim = [2] # 2 / 3 / 4 etc. dimensional PCA/dDR
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

# independent noise
ind_noise = 0

# ====================== make the covariance matrices / 4 different datasets ============================
datasets = ['big_orth', 'big_aligned', 'small_orth', 'small_aligned']
X = {k: None for k in datasets}
cov = dict.fromkeys(datasets)
evscale = 7
inflimdim = 2
limvar = 0.4 # smaller number = more aligned with dU

# big / orthogonal
evecs = np.concatenate([sh.generate_lv_loading(nUnits, mean_loading=0, variance=1, mag=1) for i in range(nUnits)], axis=1)
evecs = sh.orthonormal(evecs)
evecs *= evscale
svs = 1 / np.arange(1, nUnits+1)**(1/2)
cov['big_orth'] = sh.generate_full_rank_cov(evecs * svs)
# do indep. noise
#c = np.zeros((nUnits, nUnits))
#temp = cov['big_orth']
#cov['big_orth'] = c
#np.fill_diagonal(cov['big_orth'], np.diag(temp)) 
X1 = np.random.multivariate_normal(ubig[0, :], cov['big_orth'], k)
X2 = np.random.multivariate_normal(ubig[1, :], cov['big_orth'], k)
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
X1 = np.random.multivariate_normal(ubig[0, :], cov['big_aligned'], k)
X2 = np.random.multivariate_normal(ubig[1, :], cov['big_aligned'], k)
X['big_aligned']= np.stack((X1, X2)).transpose([-1, 1, 0])

# small / orthogonal
cov['small_orth'] = cov['big_orth']
X1 = np.random.multivariate_normal(usmall[0, :], cov['small_orth'], k)
X2 = np.random.multivariate_normal(usmall[1, :], cov['small_orth'], k)
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
X1 = np.random.multivariate_normal(usmall[0, :], cov['small_aligned'], k)
X2 = np.random.multivariate_normal(usmall[1, :], cov['small_aligned'], k)
X['small_aligned']= np.stack((X1, X2)).transpose([-1, 1, 0])

# add indep. noise
for d in X.keys():
    if ind_noise != 0:
        X[d] += np.random.normal(0, ind_noise, X[d].shape)

# =========================================================================================================

# ========================================== subsample neurons ============================================
# get est/val trial indexes (can be the same for each subset of neurons)
eidx = np.random.choice(range(k), int(k/2), replace=False)
tidx = np.array(list(set(np.arange(k)).difference(set(eidx))))
results = {k: {k: {k: np.zeros((len(n_subsets), RandSubsets)) for k in ['ddr', 'stpca', 'tapca']} for k in datasets} for k in ndim}
for jj, nset in enumerate(n_subsets):
    print('nset: {}'.format(nset))
    for ii in range(RandSubsets):
        # choose random subset of neurons
        neurons = np.random.choice(np.arange(0, nUnits), nset, replace=False)
        for dtype in datasets:
            _X = X[dtype][neurons, :, :]
            Xest = _X[:, eidx]
            Xval = _X[:, tidx]

            for nd in ndim:
                # dDR
                if nd==2: d = None 
                else: d = nd-2
                ddr = dDR(n_additional_axes=d)
                ddr.fit(Xest[:, :, 0].T, Xest[:, :, 1].T)
                Xest_ddr1 = ddr.transform(Xest[:, :, 0].T)
                Xest_ddr2 = ddr.transform(Xest[:, :, 1].T)
                Xval_ddr1 = ddr.transform(Xval[:, :, 0].T)
                Xval_ddr2 = ddr.transform(Xval[:, :, 1].T)

                r = compute_dprime(Xest_ddr1.T, Xest_ddr2.T)
                r = compute_dprime(Xval_ddr1.T, Xval_ddr2.T, wopt=r.wopt)

                results[nd][dtype]['ddr'][jj, ii] = r.dprimeSquared

                # trial-averaged PCA (same as delta mu decoding, so always one-D)
                pca = PCA(n_components=1)
                pca.fit(np.concatenate((Xest[:, :, 0].T.mean(axis=0, keepdims=True), Xest[:, :, 1].T.mean(axis=0, keepdims=True)), axis=0))
                Xest_pca1 = pca.transform(Xest[:, :, 0].T)
                Xest_pca2 = pca.transform(Xest[:, :, 1].T)
                Xval_pca1 = pca.transform(Xval[:, :, 0].T)
                Xval_pca2 = pca.transform(Xval[:, :, 1].T)

                r = compute_dprime(Xval_pca1.T, Xval_pca2.T)

                results[nd][dtype]['tapca'][jj, ii] = abs(r)

                # single trial PCA
                pca = PCA(n_components=nd)
                pca.fit(np.concatenate((Xest[:, :, 0].T, Xest[:, :, 1].T), axis=0))
                Xest_pca1 = pca.transform(Xest[:, :, 0].T)
                Xest_pca2 = pca.transform(Xest[:, :, 1].T)
                Xval_pca1 = pca.transform(Xval[:, :, 0].T)
                Xval_pca2 = pca.transform(Xval[:, :, 1].T)

                r = compute_dprime(Xest_pca1.T, Xest_pca2.T)
                r = compute_dprime(Xval_pca1.T, Xval_pca2.T, wopt=r.wopt)

                results[nd][dtype]['stpca'][jj, ii] = r.dprimeSquared

# =========================================================== plot results ===========================================================
evcolor = 'lightgreen'
alcolor = 'orchid'
f, ax = plt.subplots(3, 3, figsize=(7, 6))

ncomp = np.arange(1, nUnits+1)

# plot scree (variance explained) plots
ylim = (10**-4, 1)
du = usmall[0] - usmall[1]
du /= np.linalg.norm(du)
for data, a in zip(['small_aligned', 'small_orth'], [ax[0, 0], ax[1, 0]]):
    evals, evecs = np.linalg.eig(cov[data])
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    a.plot(ncomp, evals / evals.sum(), '.-', markersize=10, markerfacecolor='white', color=evcolor)
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel(r"Noise component ($\mathbf{e}_1-\mathbf{e}_N$)")
    a.set_ylabel("Fract. noise var. exp.", color=evcolor)
    a.tick_params(axis='y', labelcolor=evcolor)
    a.set_ylim(ylim)
    ax2 = a.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.plot(ncomp, abs(evecs.T.dot(du)), color=alcolor, zorder=-nUnits-1)
    ax2.set_ylabel(r"Noise-signal alignment", color=alcolor)
    ax2.tick_params(axis='y', labelcolor=alcolor)
    ax2.set_ylim(0, 1)

# plot dprime results
nd = 2
c1 = 'royalblue'
c2 = 'orange'
c3 = 'k'
ms = 3
_datasets = ['small_aligned', 'big_aligned', 'small_orth', 'big_orth']
for i, (a, dtype) in enumerate(zip([ax[0, 1], ax[0, 2], ax[1, 1], ax[1, 2]], _datasets)):
    if i == 0:
        lab1 = r"$dDR$"
        lab2 = r"$stPCA$"
        lab3 = r"$taPCA$"
    a.plot(n_subsets, np.nanmean(results[nd][dtype]['ddr'], axis=1), lw=1, marker='.', color=c1, markersize=ms, label=lab1)
    a.fill_between(n_subsets, np.nanmean(results[nd][dtype]['ddr'], axis=1)-np.nanstd(results[nd][dtype]['ddr'], axis=1) / np.sqrt(RandSubsets),
                               np.nanmean(results[nd][dtype]['ddr'], axis=1)+np.nanstd(results[nd][dtype]['ddr'], axis=1) / np.sqrt(RandSubsets),
                                lw=0, color=c1, alpha=0.5)
    a.plot(n_subsets, np.nanmean(results[nd][dtype]['stpca'], axis=1), lw=1, marker='.', color=c2, markersize=ms, label=lab2)
    a.fill_between(n_subsets, np.nanmean(results[nd][dtype]['stpca'], axis=1)-np.nanstd(results[nd][dtype]['stpca'], axis=1) / np.sqrt(RandSubsets),
                               np.nanmean(results[nd][dtype]['stpca'], axis=1)+np.nanstd(results[nd][dtype]['stpca'], axis=1) / np.sqrt(RandSubsets),
                                lw=0, color=c2, alpha=0.5)
    a.plot(n_subsets, np.nanmean(results[nd][dtype]['tapca'], axis=1), lw=1, marker='.', color=c3, markersize=ms, label=lab3)
    a.fill_between(n_subsets, np.nanmean(results[nd][dtype]['tapca'], axis=1)-np.nanstd(results[nd][dtype]['tapca'], axis=1) / np.sqrt(RandSubsets),
                               np.nanmean(results[nd][dtype]['tapca'], axis=1)+np.nanstd(results[nd][dtype]['tapca'], axis=1) / np.sqrt(RandSubsets),
                                lw=0, color=c3, alpha=0.5)

    if (i == 0) | (i == 2):
        a.set_ylabel(r"$d'^2$")
    if (i == 2) | (i == 3):
        a.set_xlabel(r"Neurons ($N$)")
    if i == 0:
        a.legend(frameon=False)
            
# plot example projections
data = 'small_aligned'
nidx = np.argmin(abs(500-n_subsets))
nNeurons = n_subsets[nidx]

def gauss(x, a, b, c):
    return a * np.exp(-((x-b)**2) / (2*c**2))

ms = 2
s1 = 'tab:blue'
s2 = 'tab:orange'
for a, p, tit, col in zip([ax[2, 0], ax[2, 1], ax[2, 2]], ['ddr', 'stpca', 'tapca'], [r"$dDR$", r"$stPCA$", r"$taPCA$"], [c1, c2, c3]):
    # plot the projection onto first two PCs (if relevant) or histogram, if tapca
    n = np.random.choice(range(nUnits), nNeurons, replace=False)
    d = X[data][n, :, :]
    d = d - d.mean(axis=(1, 2), keepdims=True)
    if (p=='ddr') | (p=='stpca'):
        if p=='ddr':
            ddr = dDR()
            x1, x2 = ddr.fit_transform(d[:, :, 0].T, d[:, :, 1].T)
        else:
            pca = PCA(n_components=2)
            pca.fit(np.concatenate((d[:, :, 0].T, d[:, :, 1].T), axis=0))
            x1 = pca.transform(d[:, :, 0].T)
            x2 = pca.transform(d[:, :, 1].T)
        a.scatter(x1[:, 0], x1[:, 1], color=s1, s=ms)
        a.scatter(x2[:, 0], x2[:, 1], color=s2, s=ms)
        x, y = compute_ellipse(x1[:, 0], x1[:, 1])
        a.plot(x, y, color=s1)
        x, y = compute_ellipse(x2[:, 0], x2[:, 1])
        a.plot(x, y, color=s2)        
        a.set_title(tit, color=col)
        a.set_xlabel('Dim. 1')
        a.set_ylabel('Dim. 2')
    
    elif p=='tapca':
        pca = PCA(n_components=1)
        pca.fit(np.concatenate((d[:, :, 0].T.mean(axis=0, keepdims=True), d[:, :, 1].T.mean(axis=0, keepdims=True)), axis=0))
        x1 = pca.transform(d[:, :, 0].T).squeeze()
        x2 = pca.transform(d[:, :, 1].T).squeeze()
        bmax = round(np.max(np.concatenate((x1, x2))), 1) + 2
        bmin = round(np.min(np.concatenate((x1, x2))), 1) - 2
        bins = np.arange(bmin, bmax, 1)
        h1 = a.hist(x1, bins=bins, histtype='stepfilled', color=s1, alpha=0.3, edgecolor='none')
        h2 = a.hist(x2, bins=bins, histtype='stepfilled', color=s2, alpha=0.3, edgecolor='none')
        x = np.arange(bmin, bmax, 0.001)
        popt, _ = curve_fit(gauss, h1[1][:-1], h1[0])
        a.plot(x, gauss(x, *popt), color=s1)
        popt, _ = curve_fit(gauss, h2[1][:-1], h2[0])
        a.plot(x, gauss(x, *popt), color=s2)
        a.set_xlabel(r"$\Delta \mu$")
        a.set_title(tit, color=col)

# force certain axes to have same limits
y = (np.min(ax[2, 0].get_ylim() + ax[2, 1].get_ylim()), np.max(ax[2, 0].get_ylim() + ax[2, 1].get_ylim()))
ax[2, 0].set_ylim(y); ax[2, 1].set_ylim(y)
x = (np.min(ax[2, 0].get_xlim() + ax[2, 1].get_xlim()), np.max(ax[2, 0].get_xlim() + ax[2, 1].get_xlim()))
ax[2, 0].set_xlim(x); ax[2, 1].set_xlim(x)

f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()
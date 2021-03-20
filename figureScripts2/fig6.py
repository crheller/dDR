"""
dDR vs. PCA at low trial counts -- choosing when to use dDR.

Use small delta mu / aligned dataset from figure 5 to zoom in on very low trial counts for N=??
neurons. Where does each method perform best?

Argument for this particular dataset is that it's closest to the situation of interest:
    small dU, so near threshold
    noise aligned, which has been demonstrated recently
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

savefig = False
fig_name = os.path.join(os.getcwd(), 'figures/fig6.svg')

# data/sampling params
nUnits = 200
step = 10
krange = np.arange(10, 100, step)
krange = np.append(krange, 10000)
ndim = [2] # 2 / 3 / 4 etc. dimensional PCA/dDR
nSamples = 500

# small difference between classes
duvar = 0.5
u1 = np.random.normal(4, duvar, nUnits)
u2 = np.random.normal(4, duvar, nUnits)
u = np.stack((u1, u2))

# independent noise
ind_noise = 0.5

# small / information limiting noise
evscale = 10
inflimdim = 2
limvar = 0.7 # smaller number = more aligned with dU

lv = sh.generate_lv_loading(nUnits, mean_loading=u[0]-u[1], variance=limvar, mag=1)
evecs = np.concatenate([sh.generate_lv_loading(nUnits, mean_loading=0, variance=1, mag=1) for i in range(nUnits-1)], axis=1)
evecs = np.concatenate((lv, evecs), axis=1)
evecs = sh.orthonormal(evecs)
evecs *= evscale
svs = 1 / np.arange(1, nUnits+1)**(1/2)
svs[0] = svs[inflimdim] # bump information limiting noise to a slightly smaller dimension (e.g. Rumyantsev, Kafashan)
svs[inflimdim] = 1
cov = sh.generate_full_rank_cov(evecs * svs)

results = {n: {d: np.zeros((len(krange), nSamples)) for d in ['ddr', 'stpca', 'tapca', 'fullRank']} for n in ndim}
for ii, k in enumerate(krange):
    print(f"k = {k}")
    for jj in range(nSamples):
        for nd in ndim:
            x1 = np.random.multivariate_normal(u[0], cov, k)
            x2 = np.random.multivariate_normal(u[1], cov, k)
            X = np.stack([x1, x2])
            if ind_noise != 0:
                X += np.random.normal(0, ind_noise, X.shape)

            # get fit/test trial indices for cross-validation
            eidx = np.random.choice(np.arange(k), int(k/2), replace=False)
            tidx = np.array(list(set(np.arange(k)).difference(set(eidx))))

            # FULL RANK DECODING
            try:
                r = compute_dprime(X[0, eidx].T, X[1, eidx].T, suppress_log=True)
                r = compute_dprime(X[0, tidx].T, X[1, tidx].T, wopt=r.wopt)
                results[nd]['fullRank'][ii, jj] = r.dprimeSquared
            except ValueError:
                # too few samples for full rank approx.
                results[nd]['fullRank'][ii, jj] = np.nan

            # DDR
            if nd==2: d = None 
            else: d = nd-2
            ddr = dDR(n_additional_axes=d)
            ddr.fit(X[0, eidx], X[1, eidx])
            fit_x1ddr = ddr.transform(X[0, eidx]) 
            fit_x2ddr = ddr.transform(X[1, eidx])
            test_x1ddr = ddr.transform(X[0, tidx])
            test_x2ddr = ddr.transform(X[1, tidx])
            # compute d-prime^2 and save from val set
            r = compute_dprime(fit_x1ddr.T, fit_x2ddr.T)
            r = compute_dprime(test_x1ddr.T, test_x2ddr.T, wopt=r.wopt) # use fit decoding axis
            results[nd]['ddr'][ii, jj] = r.dprimeSquared

            # stPCA
            pca = PCA(n_components=nd)
            pca.fit(np.concatenate((X[0, eidx], X[1, eidx]), axis=0))
            Xest_pca1 = pca.transform(X[0, eidx])
            Xest_pca2 = pca.transform(X[1, eidx])
            Xval_pca1 = pca.transform(X[0, tidx])
            Xval_pca2 = pca.transform(X[1, tidx])

            r = compute_dprime(Xest_pca1.T, Xest_pca2.T)
            r = compute_dprime(Xval_pca1.T, Xval_pca2.T, wopt=r.wopt)

            results[nd]['stpca'][ii, jj] = r.dprimeSquared

            # taPCA
            pca = PCA(n_components=1)
            pca.fit(np.concatenate((X[0, eidx].mean(axis=0, keepdims=True), X[1, eidx].mean(axis=0, keepdims=True)), axis=0))
            Xest_pca1 = pca.transform(X[0, eidx])
            Xest_pca2 = pca.transform(X[1, eidx])
            Xval_pca1 = pca.transform(X[0, tidx])
            Xval_pca2 = pca.transform(X[1, tidx])

            r = compute_dprime(Xval_pca1.T, Xval_pca2.T)

            results[nd]['tapca'][ii, jj] = r

nd = 2
f, ax = plt.subplots(1, 1, figsize=(4, 3))

ax.plot(krange[:-1], results[nd]['ddr'].mean(axis=1)[:-1], '.-', color='royalblue', label=r"$dDR$")
ax.fill_between(krange[:-1], results[nd]['ddr'].mean(axis=1)[:-1] - results[nd]['ddr'].std(axis=1)[:-1] / np.sqrt(nSamples),
                             results[nd]['ddr'].mean(axis=1)[:-1] + results[nd]['ddr'].std(axis=1)[:-1] / np.sqrt(nSamples),
                             lw=0, alpha=0.5, color='royalblue')
ax.plot(krange[:-1], results[nd]['stpca'].mean(axis=1)[:-1], '.-', color='orange', label=r"$stPCA$")
ax.fill_between(krange[:-1], results[nd]['stpca'].mean(axis=1)[:-1] - results[nd]['stpca'].std(axis=1)[:-1] / np.sqrt(nSamples),
                             results[nd]['stpca'].mean(axis=1)[:-1] + results[nd]['stpca'].std(axis=1)[:-1] / np.sqrt(nSamples),
                             lw=0, alpha=0.5, color='orange')
ax.plot(krange[:-1], results[nd]['tapca'].mean(axis=1)[:-1], '.-', color='k', label=r"$taPCA$")
ax.fill_between(krange[:-1], results[nd]['tapca'].mean(axis=1)[:-1] - results[nd]['tapca'].std(axis=1)[:-1] / np.sqrt(nSamples),
                             results[nd]['tapca'].mean(axis=1)[:-1] + results[nd]['tapca'].std(axis=1)[:-1] / np.sqrt(nSamples),
                             lw=0, alpha=0.5, color='k')
ax.legend(frameon=False)
ax.set_ylabel(r"$d'^2$")
ax.set_xlabel(r"Trials ($k$)")
ax.set_ylim((0, None))

f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()

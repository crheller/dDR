"""
Compare dDR decoding with decoding from full rank data.
Idea is that dDR approaches the true value much more quickly with increasing k
Compare with other dim reduction methods here? e.g. separate curves for trial-averaged PCA, single trial PCA, FA, PLS?
"""
from dDR.utils.decoding import compute_dprime
from dDR.dDR import dDR
from dDR.PCA import PCA
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

np.random.seed(123)

savefig = False
fig_name = os.path.join(os.getcwd(), 'figures/fig4.svg')

# simulate dataset with single sig. cov. dimesion, for two different stimulus categories
nUnits = 100
noisescale = 0.2 # random noise to add to covariance
u1 = np.random.normal(4, 1, nUnits)
u2 = np.random.normal(4, 1, nUnits)
lv = np.random.normal(u1-u2, 2, (nUnits, 1)) # make LV1 somewhat aligned with the signal

# 1-d
sf = 2.5 # scale lv magnitude
lv /= np.linalg.norm(lv) # low-dimensional LV
lv = sf * lv
cov = lv.dot(lv.T)
cov += np.random.normal(0, noisescale, cov.shape)
cov = cov.dot(cov.T) # force positive, semi-definite

# 2-d
sf = 2 # scale lv magnitude
lv2 = np.random.normal(0, 1, (nUnits, 1))
lv2 /= np.linalg.norm(lv2) # low-dimensional LV
# for ortho
lv2 = lv2 - lv2.T.dot(lv/np.linalg.norm(lv)) * (lv / np.linalg.norm(lv))
lv2 /= np.linalg.norm(lv2)
lv2 = sf * lv2
evecs = np.concatenate((lv, lv2), axis=-1)
cov1 = evecs.dot(evecs.T)
cov1 += np.random.normal(0, noisescale, cov1.shape)
cov1 = cov1.dot(cov1.T) # force positive, semi-definite

 # 3-d
sf = 1.8
lv3 = np.random.normal(0, 1, (nUnits, 1))
lv3 /= np.linalg.norm(lv3) # low-dimensional LV
lv3 = lv3 - lv3.T.dot(lv2/np.linalg.norm(lv2)) * (lv2 / np.linalg.norm(lv2)) - lv3.T.dot(lv/np.linalg.norm(lv)) * (lv / np.linalg.norm(lv)) 
lv3 /= np.linalg.norm(lv3)
lv3 = sf * lv3
evecs = np.concatenate((lv, lv2, lv3), axis=-1)
cov2 = evecs.dot(evecs.T)
cov2 += np.random.normal(0, noisescale, cov2.shape)
cov2 = cov2.dot(cov2.T) # force positive, semi-definite

# for different k, generate nSamples of the data, split into 50/50 est/val and compute the mean val set d-prime
nSamples = 100
krange = np.arange(10, 510, 100)
krange = np.append(krange, 10000)
d_full = {
    'l': [],
    'm': [],
    'h': []
}
d_ddr = {
    'l': [],
    'm': [],
    'h': []
}
d_pca = {
    'l': [],
    'm': [],
    'h': []  
}
evals = {} # save evals of covariance matrix (with noise)
# add others for PCA etc?
for ii, k in enumerate(krange):
    print(f"Sample {ii}/{len(krange)}")
    for dim, c in zip(['l', 'm', 'h'], [cov, cov1, cov2]):
        _dfull = []
        _ddr = []
        _pca = []
        for i in range(nSamples):
            x1 = np.random.multivariate_normal(u1, c, k)
            x2 = np.random.multivariate_normal(u2, c, k)
            X = np.stack([x1, x2])
            # add a bit of random noise to prevent (over) fitting the non sig. eigenvalues in sigma
            X += np.random.normal(0, 2, X.shape)

            # save the covariance matrix (after noise added) in the high rep case
            if (k==krange[-1]) & (i==0):
                ev, evecs = np.linalg.eig(np.cov(X[0].T))
                evals[dim] = ev[np.argsort(ev)[::-1]]
            
            # split data
            eidx = np.random.choice(np.arange(k), int(k/2), replace=False)
            tidx = np.array(list(set(np.arange(k)).difference(set(eidx))))

            # perform dim-reduction 
            # DDR
            ddr = dDR(n_additional_axes=2)
            ddr.fit(X[0, eidx], X[1, eidx])
            fit_x1ddr = ddr.transform(X[0, eidx]) 
            fit_x2ddr = ddr.transform(X[1, eidx])
            test_x1ddr = ddr.transform(X[0, tidx])
            test_x2ddr = ddr.transform(X[1, tidx])

            # compute d-prime^2 and save dprime from val set
            # ddr
            r = compute_dprime(fit_x1ddr.T, fit_x2ddr.T)
            r = compute_dprime(test_x1ddr.T, test_x2ddr.T, wopt=r.wopt) # use fit decoding axis
            _ddr.append(r.dprimeSquared)

            # PCA (on trial-averaged data -- "dumbest" approach)
            # Could also do single trial PCA, but that becomes quite similar to dDR, though less interpretable
            pca = PCA(n_components=4)
            pca.fit(np.concatenate((X[0, eidx], X[1, eidx]), axis=0))
            fit_x1pca = pca.transform(X[0, eidx]) 
            fit_x2pca = pca.transform(X[1, eidx])
            test_x1pca = pca.transform(X[0, tidx])
            test_x2pca = pca.transform(X[1, tidx])      

            # compute d-prime^2 and save dprime from val set
            # pca
            r = compute_dprime(fit_x1pca.T, fit_x2pca.T)
            r = compute_dprime(test_x1pca.T, test_x2pca.T, wopt=r.wopt) # use fit decoding axis
            _pca.append(r.dprimeSquared)

            # full data
            try:
                r = compute_dprime(X[0, eidx].T, X[1, eidx].T)
                r = compute_dprime(X[0, tidx].T, X[1, tidx].T, wopt=r.wopt)
                _dfull.append(r.dprimeSquared)
            except ValueError:
                # too few samples for full rank approx.
                _dfull.append(np.nan)
        
        d_full[dim].append(_dfull)
        d_ddr[dim].append(_ddr)
        d_pca[dim].append(_pca)

d_full = {k: np.stack(d_full[k]) for k in d_full.keys()}
d_ddr = {k: np.stack(d_ddr[k]) for k in d_ddr.keys()}
d_pca = {k: np.stack(d_pca[k]) for k in d_pca.keys()}

# plot results 
f, ax = plt.subplots(2, 3, figsize=(6.5, 4))
xpos = [0, 1, 2]
nevals = 20
eval_lim = 0.1
for tit, dim, x in zip(['1-D noise', '2-D noise', '3-D noise'], d_full.keys(), xpos):
    norm = np.nanmax(np.concatenate([d_ddr[dim].mean(axis=1), d_full[dim].mean(axis=1)]))
    d_ddr_plot = d_ddr[dim] / norm
    d_full_plot = d_full[dim] / norm
    d_pca_plot = d_pca[dim] / norm

    # plot scree plots
    ax[0, x].bar(range(nevals), evals[dim][:nevals] / evals[dim].sum(), width=1, edgecolor='white', color='forestgreen')
    ax[0, x].set_ylabel('Fraction var. exp.')
    ax[0, x].set_xlabel(r"Principal Component ($\mathbf{e}_1$ - $\mathbf{e}_{%s}$)"%str(nevals))
    ax[0, x].set_title(tit)
    ax[0, x].set_ylim((0, eval_lim))

    # plot 2-D PCA / DDR
    ax[1, x].plot(krange[:-1], d_ddr_plot.mean(axis=1)[:-1], label=r'$dDR$', color='tab:blue')
    ax[1, x].fill_between(krange[:-1], d_ddr_plot.mean(axis=1)[:-1] - d_ddr_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                d_ddr_plot.mean(axis=1)[:-1] + d_ddr_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                color='tab:blue', alpha=0.5, lw=0)
    ax[1, x].errorbar(krange[-2]+(2*np.diff(krange)[0]), d_ddr_plot.mean(axis=1)[-1], 
                        yerr=d_ddr_plot.std(axis=1)[-1], capsize=3, marker='.', color='tab:blue')

    ax[1, x].plot(krange[:-1], d_pca_plot.mean(axis=1)[:-1], label=r'$PCA$', color='tab:orange')
    ax[1, x].fill_between(krange[:-1], d_pca_plot.mean(axis=1)[:-1] - d_pca_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                d_pca_plot.mean(axis=1)[:-1] + d_pca_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                color='tab:orange', alpha=0.5, lw=0)
    ax[1, x].errorbar(krange[-2]+(2*np.diff(krange)[0]), d_pca_plot.mean(axis=1)[-1], 
                        yerr=d_pca_plot.std(axis=1)[-1], capsize=3, marker='.', color='tab:orange')

    ax[1, x].plot(krange[:-1], d_full_plot.mean(axis=1)[:-1], label=r'Full rank data', color='tab:green')
    ax[1, x].fill_between(krange[:-1], d_full_plot.mean(axis=1)[:-1] - d_full_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                d_full_plot.mean(axis=1)[:-1] + d_full_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                color='tab:green', alpha=0.5, lw=0)
    ax[1, x].errorbar(krange[-2]+(2*np.diff(krange)[0]), d_full_plot.mean(axis=1)[-1], 
                        yerr=d_full_plot.std(axis=1)[-1], capsize=3, marker='.', color='tab:green')

    ax[1, x].axvline(nUnits, label=r"$k=N$", linestyle='--', color='grey')

    ax[1, x].set_xticks(np.append(krange[::4], krange[-2]+(2*np.diff(krange)[0])))
    ax[1, x].set_xticklabels(np.append(krange[::4], krange[-1]))
    ax[1, x].set_title(f"Stim. discriminability")
    ax[1, x].set_ylabel(r"cross-validated $d'^2$"+"\n(norm. to peak)")
    ax[1, x].set_xlabel(r'Sample size ($k$)')
    ax[1, x].legend(frameon=False)

    # ADD 3D PCA/dDR


    # ADD 4D PCA/dDR


    f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()
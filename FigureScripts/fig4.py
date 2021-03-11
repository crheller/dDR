"""
Compare dDR decoding with decoding from full rank data.
Idea is that dDR approaches the true value much more quickly with increasing k
Compare with other dim reduction methods here? e.g. separate curves for trial-averaged PCA, single trial PCA, FA, PLS?
"""
from dDR.utils.decoding import compute_dprime
from dDR.dDR import dDR
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
u1 = np.random.poisson(4, nUnits)
u2 = np.random.poisson(4, nUnits)
lv = np.random.normal(0, 1, (nUnits, 1))

# low-d
sf = 2 # scale lv magnitude
lv /= np.linalg.norm(lv) # low-dimensional LV
lv = sf * lv
cov = lv.dot(lv.T)
cov += np.random.normal(0, 0.1, cov.shape)
cov = cov.dot(cov.T) # force positive, semi-definite

# mid-d
sf = 1.5 # scale lv magnitude
lv /= np.linalg.norm(lv) # low-dimensional LV
lv = sf * lv
cov1 = lv.dot(lv.T)
cov1 += np.random.normal(0, 0.1, cov1.shape)
cov1 = cov1.dot(cov1.T) # force positive, semi-definite

sf = 0.5 # high d
lv /= np.linalg.norm(lv) # low-dimensional LV
lv = sf * lv
cov2 = lv.dot(lv.T)
cov2 += np.random.normal(0, 0.1, cov2.shape)
cov2 = cov2.dot(cov2.T) # force positive, semi-definite

# for different k, generate nSamples of the data, split into 50/50 est/val and compute the mean val set d-prime
nSamples = 100
krange = np.arange(10, 510, 10)
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
# add others for PCA etc?
for ii, k in enumerate(krange):
    print(f"Sample {ii}/{len(krange)}")
    for dim, c in zip(['l', 'm', 'h'], [cov, cov1, cov2]):
        _dfull = []
        _ddr = []
        for i in range(nSamples):
            x1 = np.random.multivariate_normal(u1, c, k)
            x2 = np.random.multivariate_normal(u2, c, k)
            X = np.stack([x1, x2])
            # add a bit of random noise to prevent (over) fitting the non sig. eigenvalues in sigma
            X += np.random.normal(0, 1, X.shape)
            
            # split data
            eidx = np.random.choice(np.arange(k), int(k/2), replace=False)
            tidx = np.array(list(set(np.arange(k)).difference(set(eidx))))

            # perform dim-reduction (or dont, for full data)
            ddr = dDR()
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

d_full = {k: np.stack(d_full[k]) for k in d_full.keys()}
d_ddr = {k: np.stack(d_ddr[k]) for k in d_ddr.keys()}

# plot results 
f, axe = plt.subplots(1, 3, figsize=(6, 2))
for tit, dim, ax in zip(['low-dim', 'mid-dim', 'high-dim'], d_full.keys(), axe):
    norm = np.nanmax(np.concatenate([d_ddr[dim].mean(axis=1), d_full[dim].mean(axis=1)]))
    d_ddr_plot = d_ddr[dim] / norm
    d_full_plot = d_full[dim] / norm

    ax.plot(krange[:-1], d_ddr_plot.mean(axis=1)[:-1], label=r'$dDR$', color='tab:blue')
    ax.fill_between(krange[:-1], d_ddr_plot.mean(axis=1)[:-1] - d_ddr_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                d_ddr_plot.mean(axis=1)[:-1] + d_ddr_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                color='tab:blue', alpha=0.5, lw=0)
    ax.errorbar(krange[-2]+(2*np.diff(krange)[0]), d_ddr_plot.mean(axis=1)[-1], 
                        yerr=d_ddr_plot.std(axis=1)[-1], capsize=3, marker='o', color='tab:blue')

    ax.plot(krange[:-1], d_full_plot.mean(axis=1)[:-1], label=r'Full rank data', color='tab:orange')
    ax.fill_between(krange[:-1], d_full_plot.mean(axis=1)[:-1] - d_full_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                d_full_plot.mean(axis=1)[:-1] + d_full_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                color='tab:orange', alpha=0.5, lw=0)
    ax.errorbar(krange[-2]+(2*np.diff(krange)[0]), d_full_plot.mean(axis=1)[-1], 
                        yerr=d_full_plot.std(axis=1)[-1], capsize=3, marker='o', color='tab:orange')

    ax.axvline(nUnits*2, label=r"$k=2N$", linestyle='--', color='grey')

    ax.set_xticks(np.append(krange[::4], krange[-2]+(2*np.diff(krange)[0])))
    ax.set_xticklabels(np.append(krange[::4], krange[-1]))
    ax.set_title(f"Stimulus discriminability, {tit}")
    ax.set_ylabel(r"cross-validated $d'^2$"+"\n(norm. to peak)")
    ax.set_xlabel(r'Sample size ($k$)')
    ax.legend(frameon=False)

    f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()
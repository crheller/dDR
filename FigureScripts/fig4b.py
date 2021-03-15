"""
Goal here is to illustrate extension of dDR for cases where there are multiple latent variables
"""
from dDR.utils.decoding import compute_dprime
import dDR.utils.surrogate_helpers as sh
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
fig_name = os.path.join(os.getcwd(), 'figures/fig4b.svg')

nUnits = 100
# define mean response to each stimulus
u1 = np.random.normal(4, 0.25, nUnits)
u2 = np.random.normal(4, 0.25, nUnits)

# spike noise variance
# on each sample, add small amount of random, independent noise with variance "spike_noise" to each neuron.
# idea is to make small eigenvalues non-significant
spike_noise = 0.5

# generate full rank set of eigenvectors, so that we can control total variance
evecs = np.concatenate([sh.generate_lv_loading(nUnits, mean_loading=u1-u2, variance=1, mag=1) for i in range(nUnits)], axis=1)
evecs = sh.orthonormal(evecs)
evecs *= 7

# generate another set, with the first PC~perfectly aligned to signal
lv = sh.generate_lv_loading(nUnits, mean_loading=u1-u2, variance=0.1, mag=1) 
evecsA = np.concatenate([sh.generate_lv_loading(nUnits, mean_loading=u1-u2, variance=1, mag=1) for i in range(nUnits-1)], axis=1)
evecsA = np.concatenate((lv, evecsA), axis=1)
evecsA = sh.orthonormal(evecsA)
evecsA *= 7

# DATASET 1: independent noise (diag cov matrix)
covind = np.zeros(evecs.shape)
np.fill_diagonal(covind, np.diag(sh.generate_full_rank_cov(evecs)))

# DATASET 2: 1-D noise, not aligned (one big dimension, all others small)
svs = np.append(1, 0.3 / np.arange(2, nUnits+1)**(1/2))
cov1 = sh.generate_full_rank_cov(evecs * svs)

# DATASET 3: 1-D noise, aligned, other dimensions small
svs = np.append(1, 0.3 / np.arange(2, nUnits+1)**(1/2))
cov1a = sh.generate_full_rank_cov(evecsA * svs)

# DATASET 4:  2-D noise
svs = np.append(1, np.append(0.5, 0.2 / np.arange(3, nUnits+1)**(1/2)))
cov2 = sh.generate_full_rank_cov(evecs * svs)

# DATASET 5: 1/f noise
svs = 1/np.arange(1, nUnits+1)**(1/2)
cov1f = sh.generate_full_rank_cov(evecs * svs)

# for different k, generate nSamples of the data, split into 50/50 est/val and compute the mean val set d-prime
nSamples = 100
krange = np.arange(10, 310, 10) #np.arange(10, 510, 10)
krange = np.append(krange, 10000)
d_full = {
    'ind': [],
    '1d': [],
    '1da': [],
    '2d': [],
    '1f': []
}
d_ddr = {
    'ind': [],
    '1d': [],
    '1da': [],
    '2d': [],
    '1f': []
}
d_pca = {
    'ind': [],
    '1d': [],
    '1da': [],
    '2d': [],
    '1f': []
}
d_ddr2 = {
    'ind': [],
    '1d': [],
    '1da': [],
    '2d': [],
    '1f': []
}
d_pca2 = {
    'ind': [],
    '1d': [],
    '1da': [],
    '2d': [],
    '1f': []
}
d_sig = {
    'ind': [],
    '1d': [],
    '1da': [],
    '2d': [],
    '1f': []
}
evals = {} # save evals of covariance matrix (with noise)
for ii, k in enumerate(krange):
    print(f"Sample {ii}/{len(krange)}")
    for dim, c in zip(['ind', '1d', '1da', '2d', '1f'], [covind, cov1, cov1a, cov2, cov1f]):
        _dfull = []
        _ddr = []
        _pca = []
        _ddr2 = []
        _pca2 = []
        _sig = []
        for i in range(nSamples):
            x1 = np.random.multivariate_normal(u1, c, k)
            x2 = np.random.multivariate_normal(u2, c, k)
            X = np.stack([x1, x2])
            # add a bit of random, ind. noise
            if spike_noise != 0:
                X += np.random.normal(0, spike_noise, X.shape)

            # save the covariance matrix (after noise added) in the high rep case
            if (k==krange[-1]) & (i==0):
                ev, evecs = np.linalg.eig(np.cov(X[0].T))
                evals[dim] = ev[np.argsort(ev)[::-1]]
            
            # split data
            eidx = np.random.choice(np.arange(k), int(k/2), replace=False)
            tidx = np.array(list(set(np.arange(k)).difference(set(eidx))))

            # perform dim-reduction 
            # DDR
            ddr = dDR(n_additional_axes=None)
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

            # PCA (on single-trial data -- "dumbest" approach)
            pca = PCA(n_components=2)
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

            # ddr / pca again, but with an additional dimension
            # DDR
            ddr = dDR(n_additional_axes=1)
            ddr.fit(X[0, eidx], X[1, eidx])
            fit_x1ddr = ddr.transform(X[0, eidx]) 
            fit_x2ddr = ddr.transform(X[1, eidx])
            test_x1ddr = ddr.transform(X[0, tidx])
            test_x2ddr = ddr.transform(X[1, tidx])

            # compute d-prime^2 and save dprime from val set
            # ddr
            r = compute_dprime(fit_x1ddr.T, fit_x2ddr.T)
            r = compute_dprime(test_x1ddr.T, test_x2ddr.T, wopt=r.wopt) # use fit decoding axis
            _ddr2.append(r.dprimeSquared)

            # PCA (on single-trial data -- "dumbest" approach)
            pca = PCA(n_components=3)
            pca.fit(np.concatenate((X[0, eidx], X[1, eidx]), axis=0))
            fit_x1pca = pca.transform(X[0, eidx]) 
            fit_x2pca = pca.transform(X[1, eidx])
            test_x1pca = pca.transform(X[0, tidx])
            test_x2pca = pca.transform(X[1, tidx])      

            # compute d-prime^2 and save dprime from val set
            # pca
            r = compute_dprime(fit_x1pca.T, fit_x2pca.T)
            r = compute_dprime(test_x1pca.T, test_x2pca.T, wopt=r.wopt) # use fit decoding axis
            _pca2.append(r.dprimeSquared)

            # dU (signal) decoder - sort of the "null" reference point
            du = X[0, eidx].mean(axis=0) - X[1, eidx].mean(axis=0)
            du = du / np.linalg.norm(du)
            du = du[:, np.newaxis]
            test_x1sig = X[0, tidx].dot(du) 
            test_x2sig = X[1, tidx].dot(du) 

            # compute d-prime^2 and save dprime from val set
            # pca
            r = compute_dprime(test_x1sig.T, test_x2sig.T) # for single dim data, just returns dprime^2
            _sig.append(r)

            # full data
            try:
                r = compute_dprime(X[0, eidx].T, X[1, eidx].T, suppress_log=True)
                r = compute_dprime(X[0, tidx].T, X[1, tidx].T, wopt=r.wopt)
                _dfull.append(r.dprimeSquared)
            except ValueError:
                # too few samples for full rank approx.
                _dfull.append(np.nan)
        
        d_full[dim].append(_dfull)
        d_ddr[dim].append(_ddr)
        d_pca[dim].append(_pca)
        d_ddr2[dim].append(_ddr2)
        d_pca2[dim].append(_pca2)
        d_sig[dim].append(_sig)

d_full = {k: np.stack(d_full[k]) for k in d_full.keys()}
d_ddr = {k: np.stack(d_ddr[k]) for k in d_ddr.keys()}
d_pca = {k: np.stack(d_pca[k]) for k in d_pca.keys()}
d_ddr2 = {k: np.stack(d_ddr2[k]) for k in d_ddr2.keys()}
d_pca2 = {k: np.stack(d_pca2[k]) for k in d_pca2.keys()}
d_sig = {k: np.stack(d_sig[k]) for k in d_sig.keys()}

# plot results 
f, ax = plt.subplots(3, 5, figsize=(10, 6))
xpos = [0, 1, 2, 3, 4]
eval_lim = (10**-2.5, 0.3)
for tit, dim, x in zip(['Ind. Noise', '1-D', '1-D aligned', '2-D', '1/f'], d_full.keys(), xpos):
    norm = np.nanmax(np.concatenate([d_ddr[dim].mean(axis=1), d_full[dim].mean(axis=1)]))
    #norm = 1
    d_ddr_plot = d_ddr[dim] / norm
    d_ddr2_plot = d_ddr2[dim] / norm
    d_full_plot = d_full[dim] / norm
    d_pca_plot = d_pca[dim] / norm
    d_pca2_plot = d_pca2[dim] / norm
    d_sig_plot = d_sig[dim] / norm
    if norm == 1:
        lim = max([np.nanmax(d_full[dim].mean(axis=1)) for dim in d_full.keys()])
    else: 
        lim = 1.2

    # plot scree plots
    ax[0, x].plot(np.arange(1, nUnits+1), evals[dim]/evals[dim].sum(), '.')
    ax[0, x].set_ylim(eval_lim)
    ax[0, x].set_xscale('log')
    ax[0, x].set_yscale('log')
    ax[0, x].set_ylabel('Fraction var. exp.')
    ax[0, x].set_xlabel(r"Principal Component ($\mathbf{e}_1$ - $\mathbf{e}_{N}$)")
    ax[0, x].set_title(tit)

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
    
    ax[1, x].plot(krange[:-1], d_sig_plot.mean(axis=1)[:-1], label=r'$\Delta \mu$', color='k')
    ax[1, x].fill_between(krange[:-1], d_sig_plot.mean(axis=1)[:-1] - d_sig_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                d_sig_plot.mean(axis=1)[:-1] + d_sig_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                color='k', alpha=0.5, lw=0)
    ax[1, x].errorbar(krange[-2]+(2*np.diff(krange)[0]), d_sig_plot.mean(axis=1)[-1], 
                        yerr=d_sig_plot.std(axis=1)[-1], capsize=3, marker='.', color='k')

    ax[1, x].plot(krange[:-1], d_full_plot.mean(axis=1)[:-1], label=r'Full rank data', color='grey')
    ax[1, x].fill_between(krange[:-1], d_full_plot.mean(axis=1)[:-1] - d_full_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                d_full_plot.mean(axis=1)[:-1] + d_full_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                color='grey', alpha=0.5, lw=0)
    ax[1, x].errorbar(krange[-2]+(2*np.diff(krange)[0]), d_full_plot.mean(axis=1)[-1], 
                        yerr=d_full_plot.std(axis=1)[-1], capsize=3, marker='.', color='grey')

    ax[1, x].axvline(nUnits, label=r"$k=N$", linestyle='--', color='grey')

    ax[1, x].set_xticks(np.append(krange[::4], krange[-2]+(2*np.diff(krange)[0])))
    ax[1, x].set_xticklabels(np.append(krange[::4], krange[-1]))
    ax[1, x].set_title(f"Stim. discriminability")
    ax[1, x].set_ylabel(r"cross-validated $d'^2$"+"\n(norm. to peak)")
    if x==0:
        ax[1, x].legend(frameon=False)
    ax[1, x].set_ylim((0, lim))

    # plot PCA / dDR same as above, but using an additional dimension
    ax[2, x].plot(krange[:-1], d_ddr2_plot.mean(axis=1)[:-1], label=r'$dDR$', color='tab:blue')
    ax[2, x].fill_between(krange[:-1], d_ddr2_plot.mean(axis=1)[:-1] - d_ddr2_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                d_ddr2_plot.mean(axis=1)[:-1] + d_ddr2_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                color='tab:blue', alpha=0.5, lw=0)
    ax[2, x].errorbar(krange[-2]+(2*np.diff(krange)[0]), d_ddr2_plot.mean(axis=1)[-1], 
                        yerr=d_ddr2_plot.std(axis=1)[-1], capsize=3, marker='.', color='tab:blue')

    ax[2, x].plot(krange[:-1], d_pca2_plot.mean(axis=1)[:-1], label=r'$PCA$', color='tab:orange')
    ax[2, x].fill_between(krange[:-1], d_pca2_plot.mean(axis=1)[:-1] - d_pca2_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                d_pca2_plot.mean(axis=1)[:-1] + d_pca2_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                color='tab:orange', alpha=0.5, lw=0)
    ax[2, x].errorbar(krange[-2]+(2*np.diff(krange)[0]), d_pca2_plot.mean(axis=1)[-1], 
                        yerr=d_pca2_plot.std(axis=1)[-1], capsize=3, marker='.', color='tab:orange')

    ax[2, x].axvline(nUnits, label=r"$k=N$", linestyle='--', color='grey')

    ax[2, x].set_xticks(np.append(krange[::4], krange[-2]+(2*np.diff(krange)[0])))
    ax[2, x].set_xticklabels(np.append(krange[::4], krange[-1]))
    ax[2, x].set_ylabel(r"cross-validated $d'^2$"+"\n(norm. to peak)")
    ax[2, x].set_xlabel(r'Sample size ($k$)')
    ax[2, x].set_ylim((0, 1.2))

    f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()
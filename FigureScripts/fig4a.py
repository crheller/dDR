"""
Compare dDR decoding with decoding from full rank data and using PCA and using signal only decoding (dU)
For this figure (4a) show how performance changes as a function of PC1 size.
That is, as we "increase dimensionality" by decreasing variance of the latent variable,
how does decoding performance change?
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
fig_name = os.path.join(os.getcwd(), 'figures/fig4a.svg')

nUnits = 100
# define mean response to each stimulus
u1 = np.random.normal(4, 0.5, nUnits)
u2 = np.random.normal(4, 0.5, nUnits)

# spike noise variance
# on each sample, add random noise with variance "spike_noise" to each neuron.
# idea is to make small eigenvalues non-significant
spike_noise = 1.5

# generate "baseline" cov noise
cov_noise = np.random.normal(0, 0.1, (nUnits, nUnits))

# define the latent variable
lv = sh.generate_lv_loading(nUnits, mean_loading=u1-u2, variance=1, mag=1) # make LV1 somewhat aligned with the signal

# Dataset 1
sf = 2.5
cov1 = sh.generate_full_rank_cov(lv * sf, cov_noise)
# Dataset 2
sf = 1.8
cov2 = sh.generate_full_rank_cov(lv * sf, cov_noise)
# Dataset 3 (non-sig dimension)
sf = 1
cov3 = cov_noise.dot(cov_noise.T) #sh.generate_full_rank_cov(lv * sf, cov_noise)

# for different k, generate nSamples of the data, split into 50/50 est/val and compute the mean val set d-prime
nSamples = 100
krange = np.arange(10, 510, 100) #np.arange(10, 510, 10)
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
d_sig = {
    'l': [],
    'm': [],
    'h': [],
}
evals = {} # save evals of covariance matrix (with noise)
for ii, k in enumerate(krange):
    print(f"Sample {ii}/{len(krange)}")
    for dim, c in zip(['l', 'm', 'h'], [cov1, cov2, cov3]):
        _dfull = []
        _ddr = []
        _pca = []
        _sig = []
        for i in range(nSamples):
            x1 = np.random.multivariate_normal(u1, c, k)
            x2 = np.random.multivariate_normal(u2, c, k)
            X = np.stack([x1, x2])
            # add a bit of random noise to prevent (over) fitting the non sig. eigenvalues in sigma
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
                r = compute_dprime(X[0, eidx].T, X[1, eidx].T)
                r = compute_dprime(X[0, tidx].T, X[1, tidx].T, wopt=r.wopt)
                _dfull.append(r.dprimeSquared)
            except ValueError:
                # too few samples for full rank approx.
                _dfull.append(np.nan)
        
        d_full[dim].append(_dfull)
        d_ddr[dim].append(_ddr)
        d_pca[dim].append(_pca)
        d_sig[dim].append(_sig)

d_full = {k: np.stack(d_full[k]) for k in d_full.keys()}
d_ddr = {k: np.stack(d_ddr[k]) for k in d_ddr.keys()}
d_pca = {k: np.stack(d_pca[k]) for k in d_pca.keys()}
d_sig = {k: np.stack(d_sig[k]) for k in d_sig.keys()}

# plot results 
f, ax = plt.subplots(2, 3, figsize=(6.5, 4))
xpos = [0, 1, 2]
nevals = 20
eval_lim = 0.1
for tit, dim, x in zip(['1-D noise', '2-D noise', '3-D noise'], d_full.keys(), xpos):
    norm = np.nanmax(np.concatenate([d_ddr[dim].mean(axis=1), d_full[dim].mean(axis=1)]))
    norm = 1
    d_ddr_plot = d_ddr[dim] / norm
    d_full_plot = d_full[dim] / norm
    d_pca_plot = d_pca[dim] / norm
    d_sig_plot = d_sig[dim] / norm

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
    
    ax[1, x].plot(krange[:-1], d_sig_plot.mean(axis=1)[:-1], label=r'$\Delta \mu$', color='tab:purple')
    ax[1, x].fill_between(krange[:-1], d_sig_plot.mean(axis=1)[:-1] - d_sig_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                d_sig_plot.mean(axis=1)[:-1] + d_sig_plot.std(axis=1)[:-1] / np.sqrt(nSamples),
                                color='tab:purple', alpha=0.5, lw=0)
    ax[1, x].errorbar(krange[-2]+(2*np.diff(krange)[0]), d_sig_plot.mean(axis=1)[-1], 
                        yerr=d_sig_plot.std(axis=1)[-1], capsize=3, marker='.', color='tab:purple')

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
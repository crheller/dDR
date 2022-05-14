"""
11.05.2022 -- improved standard error estimates for across trial comparisons
    (have to deal with weird resampling from finite data)
Real data -- dataset1 (CRD004a)

1) Tuning curves for individual neurons
2) Population tuning curve
3) Decoding -- "target" detection
4) Decoding -- frequency discrimination
5) Projections of 3/4 in to the dDR / taPCA / stPCA space? Just taPCA / dDR? For the latter, could just use the same plot...

Summary panels showing ratio of dDR:taPCA, mag(dU), noise alignment across CFs?
"""
from dDR.utils.decoding import compute_dprime
from dDR.utils.dataset import Dataset, ROOT_DIR
from dDR.utils.plotting import plot_stim_pair_dDR
from dDR.PCA import PCA
from dDR.dDR import dDR

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
fig_name = os.path.join(ROOT_DIR, 'figures/fig5.svg')
fig_m1 = os.path.join(ROOT_DIR, 'figures/fig5_m1.svg')
fig_m2 = os.path.join(ROOT_DIR, 'figures/fig5_m2.svg')

# some script params
valsize = 15
nSamples = int(50 /  valsize) # how many different *unique* ways can we choose a val set of len 5
nSamples = 5    # hold fixed, but select randomly which is val
nResample = 100 # bootstrap the est sets, for a given val set
krange = [5, 10, 15, 20, 25, 30, 35] # number of trials to use for estimation set
zscore = True


# Load data, get z-score params, compute BF
data = Dataset().load(name='CRD004a')

bfsnr = '-InfdB'
rnoise = np.stack([d[:, :, data.meta['evokedBins']].sum(axis=-1) for k, d in data.spikeData[bfsnr].items()])
m = rnoise.mean(axis=(0, 1), keepdims=True)
sd = rnoise.std(axis=(0, 1), keepdims=True)
rnoise = rnoise - m
rnoise /= sd
rnoise = rnoise.mean(axis=1) # mean over trials
rnoise_mean = rnoise.mean(axis=1) # mean over neurons
bfidx = np.argsort(rnoise_mean)[::-1][0]
bf = data.cfs[bfidx]

# get resp for other snrs for the tuning curve figure
resp = {}
for snr in data.snrs:
    _r = np.stack([d[:, :, data.meta['evokedBins']].sum(axis=-1) for k, d in data.spikeData[snr].items()])
    resp[snr] = ((_r - m) / sd).mean(axis=(1, 2))

# index for sorted cellids by BF
bf_idx = np.argsort(np.argmax(rnoise, axis=0))

# choose the two stimulus pairs to compare. For both "target detection" and "freq. discrimin", use (-Inf, BF) vs. (XXX, YYY)
x1k = ('-InfdB', bf)
x2k = ('-5dB', bf)     # "target detection"
obf = data.cfs[bfidx-3]
x3k = ('-InfdB', obf) # freq discrim, ~1 octave

results = {
    'targetDetect': {
        'fullRank': np.zeros((len(krange), nResample, nSamples)),
        'ddr': np.zeros((len(krange), nResample, nSamples)),
        'tapca': np.zeros((len(krange), nResample, nSamples)),
        'stpca': np.zeros((len(krange), nResample, nSamples)),
        'tapca_norm': np.zeros((len(krange), nResample, nSamples)),
        'stpca_norm': np.zeros((len(krange), nResample, nSamples))
    },
    'freqDiscrim': {
        'fullRank': np.zeros((len(krange), nResample, nSamples)),
        'ddr': np.zeros((len(krange), nResample, nSamples)),
        'tapca': np.zeros((len(krange), nResample, nSamples)),
        'stpca': np.zeros((len(krange), nResample, nSamples)),
        'tapca_norm': np.zeros((len(krange), nResample, nSamples)),
        'stpca_norm': np.zeros((len(krange), nResample, nSamples))
    }
}
for cat, spair in zip(['targetDetect', 'freqDiscrim'], [[x1k, x2k], [x1k, x3k]]):
    x1 = data.spikeData[spair[0][0]][spair[0][1]][:, :, data.meta['evokedBins']].sum(axis=-1)
    x2 = data.spikeData[spair[1][0]][spair[1][1]][:, :, data.meta['evokedBins']].sum(axis=-1)
    X = np.stack([x1, x2])
    nUnits = X.shape[-1]
    nTrials = X.shape[1]

    # nomalize X for decoding
    if zscore:
        Xz = (X - m) / sd
    else:
        Xz = X.copy()
    # get noise covariance matrix / eigenvalues
    A0 = Xz[0] - Xz[0].mean(axis=0, keepdims=True)
    B0 = Xz[1] - Xz[1].mean(axis=0, keepdims=True)
    Xcenter = np.concatenate((A0, B0), axis=0)
    cov = np.cov(Xcenter.T)
    evals, evecs = np.linalg.eig(cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]
    evals = evals[np.argsort(evals)[::-1]]

    # save evals / noise alignement
    du = Xz[0].mean(axis=0) - Xz[1].mean(axis=0)
    du /= np.linalg.norm(du)
    align = abs(evecs.T.dot(du))
    results[cat]['evals'] = evals
    results[cat]['align'] = align

    # save (overall) projection into ddr space
    ddr = dDR(n_additional_axes=None)
    ddr.fit(Xz[0], Xz[1])
    x1ddr = ddr.transform(Xz[0]) 
    x2ddr = ddr.transform(Xz[1])
    results[cat]['x1'] = x1ddr
    results[cat]['x2'] = x2ddr
    r = compute_dprime(x1ddr.T, x2ddr.T)
    results[cat]['wopt'] = r.wopt / np.linalg.norm(r.wopt)

    # perform decoding across different sample sizes
    for jj in range(nSamples):
        tidx = np.random.choice(range(nTrials), valsize, replace=False)
        print(f"est/val set = {jj} / {nSamples}")
        for ii, k in enumerate(krange):
            # randomly select estimation trials
            est_trials = np.array(list(set(np.arange(nTrials)).difference(set(tidx))))
            X = Xz.copy() 

            print(f"bootstrapping for k = {k}")
            # get fit/test trial indices for cross-validation (project *all* left out data for unbiased comparison)
            for bb in range(nResample):
                eidx = np.random.choice(est_trials, k, replace=True)

                # FULL RANK DECODING
                try:
                    r = compute_dprime(X[0, eidx].T, X[1, eidx].T, suppress_log=True)
                    r = compute_dprime(X[0, tidx].T, X[1, tidx].T, wopt=r.wopt)
                    results[cat]['fullRank'][ii, bb, jj] = r.dprimeSquared
                except ValueError:
                    # too few samples for full rank approx.
                    results[cat]['fullRank'][ii, bb, jj] = np.nan

                # DDR
                ddr = dDR(n_additional_axes=None)
                ddr.fit(X[0, eidx], X[1, eidx])
                fit_x1ddr = ddr.transform(X[0, eidx]) 
                fit_x2ddr = ddr.transform(X[1, eidx])
                test_x1ddr = ddr.transform(X[0, tidx])
                test_x2ddr = ddr.transform(X[1, tidx])
                # compute d-prime^2 and save from val set
                rf = compute_dprime(fit_x1ddr.T, fit_x2ddr.T)
                r = compute_dprime(test_x1ddr.T, test_x2ddr.T, wopt=rf.wopt) # use fit decoding axis
                results[cat]['ddr'][ii, bb, jj] = r.dprimeSquared

                # stPCA
                pca = PCA(n_components=2)
                pca.fit(np.concatenate((X[0, eidx], X[1, eidx]), axis=0))
                Xest_pca1 = pca.transform(X[0, eidx])
                Xest_pca2 = pca.transform(X[1, eidx])
                Xval_pca1 = pca.transform(X[0, tidx])
                Xval_pca2 = pca.transform(X[1, tidx])

                r = compute_dprime(Xest_pca1.T, Xest_pca2.T)
                r = compute_dprime(Xval_pca1.T, Xval_pca2.T, wopt=r.wopt)

                results[cat]['stpca'][ii, bb, jj] = r.dprimeSquared

                # taPCA
                pca = PCA(n_components=1)
                pca.fit(np.concatenate((X[0, eidx].mean(axis=0, keepdims=True), X[1, eidx].mean(axis=0, keepdims=True)), axis=0))
                Xest_pca1 = pca.transform(X[0, eidx])
                Xest_pca2 = pca.transform(X[1, eidx])
                Xval_pca1 = pca.transform(X[0, tidx])
                Xval_pca2 = pca.transform(X[1, tidx])

                r = compute_dprime(Xval_pca1.T, Xval_pca2.T)

                results[cat]['tapca'][ii, bb, jj] = r

            results[cat]["tapca_norm"][ii, :, jj] = results[cat]['tapca'][ii, :, jj] / results[cat]['ddr'][ii, :, jj].mean()
            results[cat]["stpca_norm"][ii, :, jj] = results[cat]['stpca'][ii, :, jj] / results[cat]['ddr'][ii, :, jj].mean()

# plot projections (scatter plot, marginal on wopt, marginal on dU)

# remove outliers just for visualization
theta = 1
c = 'targetDetect'
gm = np.sqrt(results[c]['x1'][:,0]**2 + results[c]['x1'][:,1]**2)
x1td = results[c]['x1'][gm<=(np.median(gm)+(theta*np.std(gm)))]
gm = np.sqrt(results[c]['x2'][:,0]**2 + results[c]['x2'][:,1]**2)
x2td = results[c]['x2'][gm<=(np.median(gm)+(theta*np.std(gm)))]
c = 'freqDiscrim'
gm = np.sqrt(results[c]['x1'][:,0]**2 + results[c]['x1'][:,1]**2)
x1fd = results[c]['x1'][gm<=(np.median(gm)+(theta*np.std(gm)))]
gm = np.sqrt(results[c]['x2'][:,0]**2 + results[c]['x2'][:,1]**2)
x2fd = results[c]['x2'][gm<=(np.median(gm)+(theta*np.std(gm)))]

ms = 5
bins = np.arange(-10, 10, 1)
f, ax = plt.subplots(1, 3, figsize=(6, 2), sharex=True)

plot_stim_pair_dDR(x1td, 
                   x2td, 
                   xlab=r'Signal ($\Delta \mu$)',
                   ylab=r"Noise",
                   lw=1,
                   s=ms,
                   ax=ax[0])
cent = np.concatenate((x1td[:, 0], x2td[:, 0])).mean()
ax[1].hist(x1td[:, 0] - cent, histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5, bins=bins)
ax[1].hist(x2td[:, 0] - cent, histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5, bins=bins)
ax[1].set_title(r"$\Delta \mu$ projection")
ax[2].hist(x1td.dot(results['targetDetect']['wopt']).squeeze(), histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5, bins=bins)
ax[2].hist(x2td.dot(results['targetDetect']['wopt']).squeeze(), histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5, bins=bins)
ax[2].set_title(r"$w_{opt}$ projection")

f.tight_layout()

if savefig:
    f.savefig(fig_m1)

f, ax = plt.subplots(1, 3, figsize=(6, 2), sharex=True)

plot_stim_pair_dDR(x1fd, 
                   x2fd, 
                   xlab=r'Signal ($\Delta \mu$)',
                   ylab=r"Noise",
                   lw=1,
                   s=ms,
                   ax=ax[0])
cent = np.concatenate((x1fd[:, 0], x2fd[:, 0])).mean()
ax[1].hist(x1fd[:, 0] - cent, histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5, bins=bins)
ax[1].hist(x2fd[:, 0] - cent, histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5, bins=bins)
ax[1].set_title(r"$\Delta \mu$ projection")
ax[2].hist(x1fd.dot(results['freqDiscrim']['wopt']).squeeze(), histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5, bins=bins)
ax[2].hist(x2fd.dot(results['freqDiscrim']['wopt']).squeeze(), histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5, bins=bins)
ax[2].set_title(r"$w_{opt}$ projection")

f.tight_layout()

if savefig:
    f.savefig(fig_m2)

# ========================================================================================================
# overall figure -- tuning panels / decoding performance
f = plt.figure(figsize=(14, 7.5))
gs = f.add_gridspec(10, 19)
bfim = f.add_subplot(gs[:6, :4])
ftc = f.add_subplot(gs[6:, :4])
p1 = f.add_subplot(gs[:5, 4:9])
p2 = f.add_subplot(gs[5:, 4:9])
d1 = f.add_subplot(gs[:5, 9:14])
d2 = f.add_subplot(gs[5:, 9:14])
d1n = f.add_subplot(gs[:5, 14:])
d2n = f.add_subplot(gs[5:, 14:])

# plot individual neuron's tuning curves on heatmap
bfim.imshow(rnoise[:, bf_idx].T, cmap='bwr', aspect='auto', vmin=-2, vmax=2)
bfim.set_ylabel("Neuron")
bfim.set_xticks(range(len(data.cfs)))
bfim.set_xticklabels(data.cfs, rotation=45)
bfim.set_xlabel('Noise Center Frequency')

# plot the mean tuning curve over neurons
ftc.plot(data.cfs, rnoise_mean, '.-', color='k', lw=1, label='-InfdB')
cmap = cm.get_cmap('Reds', 20)
for snr, col in zip(['-10dB', '-5dB', '0dB'], [cmap(5), cmap(10), cmap(15)]):
    ftc.plot(data.cfs, resp[snr], '.-', color=col, lw=1, label=snr)
ftc.legend(frameon=False)
ftc.set_ylabel("Spike count (z-scored)")
ftc.set_xlabel('Noise Center Frequency')
ftc.set_xscale('log')

# performance for "decoding"
klab = np.array(krange) + valsize # total trials used
for a, c in zip([d1, d2], ['targetDetect', 'freqDiscrim']):
    err = (np.mean(results[c]['stpca_norm'].std(axis=1), axis=1)) / (1/np.sqrt(np.array(krange)/krange[-1])) 
    u = np.mean(np.mean(results[c]['stpca_norm'], axis=1), axis=1)
    a.plot(klab, u, '.-', color='orange', label=r"$stPCA$")
    a.fill_between(klab, u - err,
                                u + err,
                                lw=0, alpha=0.5, color='orange')

    err = (np.mean(results[c]['tapca_norm'].std(axis=1), axis=1)) / (1/np.sqrt(np.array(krange)/krange[-1])) 
    u = np.mean(np.mean(results[c]['tapca_norm'], axis=1), axis=1)
    a.plot(klab, u, '.-', color='k', label=r"$taPCA$")
    a.fill_between(klab, u - err,
                                u + err,
                                lw=0, alpha=0.5, color='k')
    a.legend(frameon=False)
    a.set_ylabel(r"Relative decoding performance")
    a.set_xlabel(r"Trials ($k$)")
    #a.set_ylim((np.max([0, a.get_ylim()[1]-6]), None))
    a.set_ylim((0, 1.1))
    a.axhline(1, linestyle="--", color="grey")

ylim = (10**-3, 1)
for a, c in zip([d1n, d2n], ['targetDetect', 'freqDiscrim']):
    evcolor = 'lightgreen'
    alcolor = 'orchid'
    ncomp = np.arange(1, nUnits+1)
    ylim = (10**-3, 1)
    a.plot(ncomp, evals / evals.sum(), '.-', markersize=10, markerfacecolor='white', color=evcolor)
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel(r"Noise component ($\mathbf{e}_1-\mathbf{e}_N$)")
    a.set_ylabel("Fract. noise var. exp.", color=evcolor)
    a.tick_params(axis='y', labelcolor=evcolor)
    a.set_ylim(ylim)
    ax2 = a.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.plot(ncomp, results[c]['align'], '.-', markersize=10, markerfacecolor='white', color=alcolor, zorder=-nUnits-1)
    ax2.set_ylabel(r"Noise-signal alignment", color=alcolor)
    ax2.tick_params(axis='y', labelcolor=alcolor)
    ax2.set_ylim(0, 1)

# plot projections
for a, c in zip([p1, p2], [[x1td, x2td], [x1fd, x2fd]]):
    # remove outliers for visualization
    cent = np.concatenate((c[0][:, 0], c[1][:, 0])).mean()
    c[0][:, 0] = c[0][:, 0] - cent
    c[1][:, 0] = c[1][:, 0] - cent
    plot_stim_pair_dDR(c[0], 
                       c[1], 
                       xlab=r'Signal ($\Delta \mu$)',
                       ylab=r"Noise",
                       lw=1,
                       s=ms,
                       ax=a)
    a.set_xlim((bins[0], bins[-1]+1))
    a.set_ylim(a.get_ylim()[0]-2, a.get_ylim()[1]+2)
f.tight_layout()

if savefig:
    f.savefig(fig_name)
    f.savefig(fig_name.replace(".svg", ".png"))

plt.show()
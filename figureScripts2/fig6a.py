"""
Real data -- dataset1 (CRD004a)

1) Tuning curves for individual neurons
2) Population tuning curve
3) Decoding -- "target" detection
4) Decoding -- frequency discrimination
5) Projections of 3/4 in to the dDR / taPCA / stPCA space? Just taPCA / dDR? For the latter, could just use the same plot...

Summary panels showing ratio of dDR:taPCA, mag(dU), noise alignment across CFs?
"""
from dDR.utils.decoding import compute_dprime
from dDR.utils.dataset import Dataset
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
fig_name = os.path.join(os.getcwd(), 'figures/fig6a.svg')

# some script params
nSamples = 100 # number of random samples for each sample size
krange = np.arange(10, 55, 5)
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
bfidx = np.argmax(rnoise_mean)
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
x2k = ('0dB', bf)     # "target detection"
obf = data.cfs[bfidx-3]
x3k = ('-InfdB', obf) # freq discrim, ~1 octave

results = {
    'targetDetect': {
        'fullRank': np.zeros((len(krange), nSamples)),
        'ddr': np.zeros((len(krange), nSamples)),
        'tapca': np.zeros((len(krange), nSamples)),
        'stpca': np.zeros((len(krange), nSamples))
    },
    'freqDiscrim': {
        'fullRank': np.zeros((len(krange), nSamples)),
        'ddr': np.zeros((len(krange), nSamples)),
        'tapca': np.zeros((len(krange), nSamples)),
        'stpca': np.zeros((len(krange), nSamples))
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
        #Xz = X - X.mean(axis=(0, 1), keepdims=True)
        #Xz = Xz / Xz.std(axis=(0, 1), keepdims=True)
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
    for ii, k in enumerate(krange):
        print(f"k = {k}")
        for jj in range(nSamples):
            trials = np.random.choice(range(nTrials), k)
            X = Xz.copy() 

            # get fit/test trial indices for cross-validation (project *all* left out data for unbiased comparison)
            eidx = np.random.choice(trials, int(k/2), replace=False)
            tidx = np.random.choice(np.array(list(set(np.arange(nTrials)).difference(set(eidx)))), int(nTrials/2), replace=False)

            # FULL RANK DECODING
            try:
                r = compute_dprime(X[0, eidx].T, X[1, eidx].T, suppress_log=True)
                r = compute_dprime(X[0, tidx].T, X[1, tidx].T, wopt=r.wopt)
                results[cat]['fullRank'][ii, jj] = r.dprimeSquared
            except ValueError:
                # too few samples for full rank approx.
                results[cat]['fullRank'][ii, jj] = np.nan

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
            results[cat]['ddr'][ii, jj] = r.dprimeSquared

            # stPCA
            pca = PCA(n_components=2)
            pca.fit(np.concatenate((X[0, eidx], X[1, eidx]), axis=0))
            Xest_pca1 = pca.transform(X[0, eidx])
            Xest_pca2 = pca.transform(X[1, eidx])
            Xval_pca1 = pca.transform(X[0, tidx])
            Xval_pca2 = pca.transform(X[1, tidx])

            r = compute_dprime(Xest_pca1.T, Xest_pca2.T)
            r = compute_dprime(Xval_pca1.T, Xval_pca2.T, wopt=r.wopt)

            results[cat]['stpca'][ii, jj] = r.dprimeSquared

            # taPCA
            pca = PCA(n_components=1)
            pca.fit(np.concatenate((X[0, eidx].mean(axis=0, keepdims=True), X[1, eidx].mean(axis=0, keepdims=True)), axis=0))
            Xest_pca1 = pca.transform(X[0, eidx])
            Xest_pca2 = pca.transform(X[1, eidx])
            Xval_pca1 = pca.transform(X[0, tidx])
            Xval_pca2 = pca.transform(X[1, tidx])

            r = compute_dprime(Xval_pca1.T, Xval_pca2.T)

            results[cat]['tapca'][ii, jj] = r

# plot projections (scatter plot, marginal on wopt, marginal on dU)
ms = 5
f, ax = plt.subplots(1, 3, figsize=(6, 2), sharex=True)

plot_stim_pair_dDR(results['targetDetect']['x1'], 
                   results['targetDetect']['x2'], 
                   xlab=r'Signal ($\Delta \mu$)',
                   ylab=r"Noise",
                   lw=1,
                   s=ms,
                   ax=ax[0])
ax[1].hist(results['targetDetect']['x1'][:, 0], histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5)
ax[1].hist(results['targetDetect']['x2'][:, 0], histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5)
ax[1].set_title(r"$\Delta \mu$ projection")
ax[2].hist(results['targetDetect']['x1'].dot(results['targetDetect']['wopt']).squeeze(), histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5)
ax[2].hist(results['targetDetect']['x2'].dot(results['targetDetect']['wopt']).squeeze(), histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5)
ax[2].set_title(r"$w_{opt}$ projection")

f, ax = plt.subplots(1, 3, figsize=(6, 2), sharex=True)

plot_stim_pair_dDR(results['freqDiscrim']['x1'], 
                   results['freqDiscrim']['x2'], 
                   xlab=r'Signal ($\Delta \mu$)',
                   ylab=r"Noise",
                   lw=1,
                   s=ms,
                   ax=ax[0])
ax[1].hist(results['freqDiscrim']['x1'][:, 0], histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5)
ax[1].hist(results['freqDiscrim']['x2'][:, 0], histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5)
ax[1].set_title(r"$\Delta \mu$ projection")
ax[2].hist(results['freqDiscrim']['x1'].dot(results['freqDiscrim']['wopt']).squeeze(), histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5)
ax[2].hist(results['freqDiscrim']['x2'].dot(results['freqDiscrim']['wopt']).squeeze(), histtype='stepfilled', lw=0, edgecolor='k', alpha=0.5)
ax[2].set_title(r"$w_{opt}$ projection")

f.tight_layout()

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
bfim.set_xlabel('CF')

# plot the mean tuning curve over neurons
ftc.plot(data.cfs, rnoise_mean, '.-', color='k', lw=1, label='-InfdB')
cmap = cm.get_cmap('Reds', 20)
for snr, col in zip(['-10dB', '-5dB', '0dB'], [cmap(5), cmap(10), cmap(15)]):
    ftc.plot(data.cfs, resp[snr], '.-', color=col, lw=1, label=snr)
ftc.legend(frameon=False)
ftc.set_ylabel("Spike count (z-scored)")
ftc.set_xlabel('CF')
ftc.set_xscale('log')

# performance for "target detection"
for a, c in zip([d1, d2], ['targetDetect', 'freqDiscrim']):
    a.plot(krange[:-1], results[c]['ddr'].mean(axis=1)[:-1], '.-', color='royalblue', label=r"$dDR$")
    a.fill_between(krange[:-1], results[c]['ddr'].mean(axis=1)[:-1] - results[c]['ddr'].std(axis=1)[:-1] / np.sqrt(nSamples),
                                results[c]['ddr'].mean(axis=1)[:-1] + results[c]['ddr'].std(axis=1)[:-1] / np.sqrt(nSamples),
                                lw=0, alpha=0.5, color='royalblue')

    a.plot(krange[:-1], results[c]['stpca'].mean(axis=1)[:-1], '.-', color='orange', label=r"$stPCA$")
    a.fill_between(krange[:-1], results[c]['stpca'].mean(axis=1)[:-1] - results[c]['stpca'].std(axis=1)[:-1] / np.sqrt(nSamples),
                                results[c]['stpca'].mean(axis=1)[:-1] + results[c]['stpca'].std(axis=1)[:-1] / np.sqrt(nSamples),
                                lw=0, alpha=0.5, color='orange')
    a.plot(krange[:-1], results[c]['tapca'].mean(axis=1)[:-1], '.-', color='k', label=r"$taPCA$")
    a.fill_between(krange[:-1], results[c]['tapca'].mean(axis=1)[:-1] - results[c]['tapca'].std(axis=1)[:-1] / np.sqrt(nSamples),
                                results[c]['tapca'].mean(axis=1)[:-1] + results[c]['tapca'].std(axis=1)[:-1] / np.sqrt(nSamples),
                                lw=0, alpha=0.5, color='k')
    a.legend(frameon=False)
    a.set_ylabel(r"$d'^2$")
    a.set_xlabel(r"Trials ($k$)")
    a.set_ylim((0, None))

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
for a, c in zip([p1, p2], ['targetDetect', 'freqDiscrim']):
    plot_stim_pair_dDR(results[c]['x1'], 
                    results[c]['x2'], 
                    xlab=r'Signal ($\Delta \mu$)',
                    ylab=r"Noise",
                    lw=1,
                    s=ms,
                    ax=a)

f.tight_layout()

plt.show()
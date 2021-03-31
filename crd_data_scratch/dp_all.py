"""
For each dataset, compute dprime using dDR / taPCA for all pairwise combos
of stimuli. Relative to BF - so all pariwise combos of onBF at each SNR to all others
"""
from dDR.utils.dataset import Dataset
from dDR.utils.decoding import compute_dprime
from dDR.PCA import PCA
from dDR.dDR import dDR

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

# load data
data = Dataset().load(name='CRD004a') # CRD002a, CRD003b, CRD004a

# get BF
bfsnr = '-InfdB'
rnoise_mean = np.stack([d[:, :, data.meta['evokedBins']].sum(axis=-1).mean(axis=(0, 1)) for k, d in data.spikeData[bfsnr].items()])
bf = data.cfs[np.argmax(rnoise_mean)]

# for each BF SNR, get d' between it and all other sounds
results = {}
for snrbf in data.snrs:
    # save results for each SNR
    results[snrbf] = pd.DataFrame()
    x1 = data.spikeData[snrbf][bf][:, :, data.meta['evokedBins']].sum(axis=-1)
    for snr in data.snrs:
        for cf in data.cfs:
            if (cf!=bf) | (snr!=snrbf):
                x2 = data.spikeData[snr][cf][:, :, data.meta['evokedBins']].sum(axis=-1)
                X = np.stack([x1, x2])
                nUnits = X.shape[-1]
                nTrials = X.shape[1]
                eidx = np.random.choice(range(nTrials), int(nTrials/2), replace=False)
                tidx = np.random.choice(np.array(list(set(np.arange(nTrials)).difference(set(eidx)))), int(nTrials/2), replace=False)

                # get evecs
                A0 = X[0] - X[0].mean(axis=0, keepdims=True)
                B0 = X[1] - X[1].mean(axis=0, keepdims=True)
                Xcenter = np.concatenate((A0, B0), axis=0)
                cov = np.cov(Xcenter.T)
                evals, evecs = np.linalg.eig(cov)
                evecs = evecs[:, np.argsort(evals)[::-1]]
                evals = evals[np.argsort(evals)[::-1]]

                # get dU
                du = X[0].mean(axis=0) - X[1].mean(axis=0)
                dumag = np.linalg.norm(du)
                du /= dumag

                # get alignment
                cosdu = evecs.T.dot(du)

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
                ddr_dprime = r.dprimeSquared

                # taPCA
                pca = PCA(n_components=1)
                pca.fit(np.concatenate((X[0, eidx].mean(axis=0, keepdims=True), X[1, eidx].mean(axis=0, keepdims=True)), axis=0))
                Xest_pca1 = pca.transform(X[0, eidx])
                Xest_pca2 = pca.transform(X[1, eidx])
                Xval_pca1 = pca.transform(X[0, tidx])
                Xval_pca2 = pca.transform(X[1, tidx])
                r = compute_dprime(Xval_pca1.T, Xval_pca2.T)
                tapca_dprime = r

                # save results
                _results = [ddr_dprime, tapca_dprime, dumag, cosdu, evals, snr, cf]
                _results = pd.DataFrame(index=['ddr', 'tapca', 'dumag', 'cosdu', 'evals', 'snr2', 'cf'],
                                        data=_results).T
                results[snrbf] = results[snrbf].append(_results)

# plot
f, ax = plt.subplots(4, 4, figsize=(12, 12))
for r, y in zip([0, 1, 2, 3], ['ratio', 'dumag', 'e1', 'evals1']):
    # -InfdB
    df = results['-InfdB']
    df['ratio'] = df['ddr'] / df['tapca']
    df['diff'] = df['ddr'] - df['tapca']
    df['e1'] = df['cosdu'].apply(lambda x: float(abs(x[0])))
    df['evals1'] = df['evals'].apply(lambda x: float(abs(x[0])))
    for snr in df.snr2.unique():
        mask = df.snr2==snr
        ax[r, 0].plot((np.log2(df[mask]['cf'].values.astype(np.int)/bf)), df[mask][y], label=snr)
    ax[r, 0].legend(frameon=False)
    ax[r, 0].set_xlabel('Octaves from BF')
    ax[r, 0].set_ylabel(y)
    if y == 'ratio':
        ax[r, 0].axhline(1, linestyle='--', color='grey')
    ax[r, 0].set_title('-InfdB')

    # -10dB
    df = results['-10dB']
    df['ratio'] = df['ddr'] / df['tapca']
    df['diff'] = df['ddr'] - df['tapca']
    df['e1'] = df['cosdu'].apply(lambda x: float(abs(x[0])))
    df['evals1'] = df['evals'].apply(lambda x: float(abs(x[0])))
    for snr in df.snr2.unique():
        mask = df.snr2==snr
        ax[r, 1].plot((np.log2(df[mask]['cf'].values.astype(np.int)/bf)), df[mask][y], label=snr)
    ax[r, 1].legend(frameon=False)
    ax[r, 1].set_xlabel('Octaves from BF')
    ax[r, 1].set_ylabel(y)
    if y == 'ratio':
        ax[r, 1].axhline(1, linestyle='--', color='grey')
    ax[r, 1].set_title('-10dB')

    # -5dB
    df = results['-5dB']
    df['ratio'] = df['ddr'] / df['tapca']
    df['diff'] = df['ddr'] - df['tapca']
    df['e1'] = df['cosdu'].apply(lambda x: float(abs(x[0])))
    df['evals1'] = df['evals'].apply(lambda x: float(abs(x[0])))
    for snr in df.snr2.unique():
        mask = df.snr2==snr
        ax[r, 2].plot((np.log2(df[mask]['cf'].values.astype(np.int)/bf)), df[mask][y], label=snr)
    ax[r, 2].legend(frameon=False)
    ax[r, 2].set_xlabel('Octaves from BF')
    ax[r, 2].set_ylabel(y)
    if y == 'ratio':
        ax[r, 2].axhline(1, linestyle='--', color='grey')
    ax[r, 2].set_title('-5dB')

    # 0dB
    df = results['0dB']
    df['ratio'] = df['ddr'] / df['tapca']
    df['diff'] = df['ddr'] - df['tapca']
    df['e1'] = df['cosdu'].apply(lambda x: float(abs(x[0])))
    df['evals1'] = df['evals'].apply(lambda x: float(abs(x[0])))
    for snr in df.snr2.unique():
        mask = df.snr2==snr
        ax[r, 3].plot((np.log2(df[mask]['cf'].values.astype(np.int)/bf)), df[mask][y], label=snr)
    ax[r, 3].legend(frameon=False)
    ax[r, 3].set_xlabel('Octaves from BF')
    ax[r, 3].set_ylabel(y)
    if y == 'ratio':
        ax[r, 3].axhline(1, linestyle='--', color='grey')
    ax[r, 3].set_title('0dB')

f.tight_layout()

plt.show()
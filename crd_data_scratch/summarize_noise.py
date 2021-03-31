"""
Plot noise space for all cordy datasets.

1 - BF stims only: show scree plot and alignment
"""

from dDR.utils.dataset import Dataset
from dDR.utils.decoding import compute_dprime
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

f, ax = plt.subplots(1, 1, figsize=(4.5, 4))
for i, (site, color) in enumerate(zip(['CRD002a', 'CRD003b', 'CRD004a'], ['tab:blue', 'tab:orange', 'tab:green'])):
    data = Dataset().load(name=site)  # CRD002a, CRD003b, CRD004a

    # get population best frequency using the first mean response across all neurons to the noise alone (-Inf)
    rnoise_mean = np.stack([d[:, :, data.meta['evokedBins']].sum(axis=-1).mean(axis=(0, 1)) for k, d in data.spikeData['-InfdB'].items()])
    bf = data.cfs[np.argmax(rnoise_mean)]
    # x1 / x2 / x3 / x4 are shape rep X neuron
    x1 = data.spikeData['-InfdB'][bf][:, :, data.meta['evokedBins']].sum(axis=-1)
    x2 = data.spikeData['-10dB'][bf][:, :, data.meta['evokedBins']].sum(axis=-1)
    x3 = data.spikeData['-5dB'][bf][:, :, data.meta['evokedBins']].sum(axis=-1)
    x4 = data.spikeData['0dB'][bf][:, :, data.meta['evokedBins']].sum(axis=-1)
    X = np.stack([x1, x2, x3, x4])
    nUnits = X.shape[-1]
    nTrials = X.shape[1]

    Xz = X - X.mean(axis=(0, 1), keepdims=True)
    #sd = Xz.std(axis=(0, 1), keepdims=True)
    #sd[sd==0] = 1
    #Xz = Xz / sd
    # get noise covariance matrix / eigenvalues
    cov = np.cov((Xz.reshape(-1, nUnits) - Xz.reshape(-1, nUnits).mean(axis=0)).T)
    evals, evecs = np.linalg.eig(cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]
    evals = evals[np.argsort(evals)[::-1]]

    # plot eigenspectrum / alignement
    evcolor = 'lightgreen'
    alcolor = 'orchid'
    ncomp = np.arange(1, nUnits+1)
    ylim = (10**-3, 1)
    du = Xz[0].mean(axis=0) - Xz[3].mean(axis=0)
    du /= np.linalg.norm(du)
    ax.plot(ncomp, evals / evals.sum(), '.-', markersize=10, markerfacecolor='white', color=color)

    if i == 0:
        ax2 = ax.twinx()
    ax2.plot(ncomp, abs(evecs.T.dot(du)), color=color, zorder=-nUnits-1)


ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r"Noise component ($\mathbf{e}_1-\mathbf{e}_N$)")
ax.set_ylabel("Fract. noise var. exp.", color=evcolor)
ax.tick_params(axis='y', labelcolor=evcolor)
ax.set_ylim(ylim)
ax2.spines['right'].set_visible(True)
ax2.set_ylabel(r"Noise-signal alignment", color=alcolor)
ax2.tick_params(axis='y', labelcolor=alcolor)
ax2.set_ylim(0, 1)

f.tight_layout()

plt.show()
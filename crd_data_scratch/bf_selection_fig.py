"""
Scratch file developing figure illustrating the stimuli and BF selection
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

# load data
data = Dataset().load(name='CRD003b') # CRD002a, CRD003b, CRD004a

# get BF
bfsnr = '-InfdB'
rnoise = np.stack([d[:, :, data.meta['evokedBins']].sum(axis=-1) for k, d in data.spikeData[bfsnr].items()])
m = rnoise.mean(axis=(0, 1), keepdims=True)
sd = rnoise.std(axis=(0, 1), keepdims=True)
rnoise = rnoise - m
rnoise /= sd
rnoise = rnoise.mean(axis=1) # mean over trials
rnoise_mean = rnoise.mean(axis=1) # mean over neurons
bf = data.cfs[np.argmax(rnoise_mean)]

# get resp for other snrs
r = {}
for snr in data.snrs:
    _r = np.stack([d[:, :, data.meta['evokedBins']].sum(axis=-1) for k, d in data.spikeData[snr].items()])
    r[snr] = ((_r - m) / sd).mean(axis=(1, 2))
    

# sort cellids
idx = np.argsort(np.argmax(rnoise, axis=0))

# plot
f = plt.figure(figsize=(4, 8))

hm = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
lp = plt.subplot2grid((5, 1), (3, 0), rowspan=2)

# plot individual neuron's tuning curves on heatmap
hm.imshow(rnoise[:, idx].T, cmap='bwr', aspect='auto', vmin=-2, vmax=2)
hm.set_ylabel("Neuron")
hm.set_xticks(range(len(data.cfs)))
hm.set_xticklabels(data.cfs, rotation=45)
hm.set_xlabel('CF')

# plot the mean tuning curve over neurons
lp.plot(data.cfs, rnoise_mean, '.-', color='k', lw=1, label='-InfdB')
cmap = cm.get_cmap('Reds', 20)
for snr, col in zip(['-10dB', '-5dB', '0dB'], [cmap(5), cmap(10), cmap(15)]):
    lp.plot(data.cfs, r[snr], '.-', color=col, lw=1, label=snr)
lp.legend(frameon=False)
lp.set_ylabel("Spike count")
lp.set_xlabel('CF')
lp.set_xscale('log')


f.tight_layout()

plt.show()
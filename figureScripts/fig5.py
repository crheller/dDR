'''
Point here is to illustrate dDR method.
Use two cartoon datasets (1-D / 2-D) and a more realistic (1/f) dataset
For each dataset, perform dDR with 1, 2, and 3 noise dimensions.
Higher-D approaches overfit more early on, but reach greater max performance (for the 2-D & 1/n data)

Normalize to full rank performance at 10k trials (approx true dprime)
'''
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
fig_name = os.path.join(os.getcwd(), 'figures/fig5.svg')

nUnits = 100
# define mean response to each stimulus
u1 = np.random.normal(4, 0.2, nUnits)
u2 = np.random.normal(4, 0.2, nUnits)

# independent noise
ind_noise = 0.6

# generate "eigenvectors"
evecs = np.concatenate([sh.generate_lv_loading(nUnits, mean_loading=0, variance=2, mag=1) for i in range(nUnits)], axis=1)
evecs = sh.orthonormal(evecs)
evecs *= 3

# DATASET 1
svs = np.append(0.6, 0.3 / np.arange(2, nUnits+1)**(1/2))
cov1 = sh.generate_full_rank_cov(evecs * svs)

# DATASET 2
svs = np.append(0.6, np.append(0.4, 0.3 / np.arange(3, nUnits+1)**(1/2)))
cov2 = sh.generate_full_rank_cov(evecs * svs)

# DATASET 3
svs = 1 / np.arange(1, nUnits+1)**(1/2)
cov1n = sh.generate_full_rank_cov(evecs * svs)

# measure decoding performance for a range of sample sizes
# for each sample size, generate "nSamples" datasets to measure uncertainty of d-prime estimate
nSamples = 500
stepSize = 25
krange = np.arange(10, 150, stepSize)
krange = np.append(krange, 10000)
results = {k: {d: np.zeros((len(krange), nSamples)) for d in ['ddr1', 'ddr2', 'ddr3', 'fullRank']} for k in ['1d', '2d', '1n']}
for ii, k in enumerate(krange):
    print(f"k = {k}")
    for d, c in zip(results.keys(), [cov1, cov2, cov1n]):
        for jj in range(nSamples):
            x1 = np.random.multivariate_normal(u1, c, k)
            x2 = np.random.multivariate_normal(u2, c, k)
            X = np.stack([x1, x2])
            if ind_noise != 0:
                X += np.random.normal(0, ind_noise, X.shape)

            # get fit/test trial indices for cross-validation
            eidx = np.random.choice(np.arange(k), int(k/2), replace=False)
            tidx = np.array(list(set(np.arange(k)).difference(set(eidx))))

            # do the decoding analysis
            # FULL RANK
            try:
                r = compute_dprime(X[0, eidx].T, X[1, eidx].T, suppress_log=True)
                r = compute_dprime(X[0, tidx].T, X[1, tidx].T, wopt=r.wopt)
                results[d]['fullRank'][ii, jj] = r.dprimeSquared
            except ValueError:
                # too few samples for full rank approx.
                results[d]['fullRank'][ii, jj] = np.nan

            # DDR
            for dd, ddr_key in zip(np.arange(3), ['ddr1', 'ddr2', 'ddr3']):
                if dd==0:
                    ndim = None
                else:
                    ndim = dd
                ddr = dDR(n_additional_axes=ndim)
                ddr.fit(X[0, eidx], X[1, eidx])
                fit_x1ddr = ddr.transform(X[0, eidx]) 
                fit_x2ddr = ddr.transform(X[1, eidx])
                test_x1ddr = ddr.transform(X[0, tidx])
                test_x2ddr = ddr.transform(X[1, tidx])
                # compute d-prime^2 and save from val set
                r = compute_dprime(fit_x1ddr.T, fit_x2ddr.T)
                r = compute_dprime(test_x1ddr.T, test_x2ddr.T, wopt=r.wopt) # use fit decoding axis
                results[d][ddr_key][ii, jj] = r.dprimeSquared

# top row:    eigenspectrum of the noise for each dataset
# bottom row: dDR performance, normalized to fullRank
f, ax = plt.subplots(2, 3, figsize=(6, 4))

color = 'springgreen'
ylim = (10**-3.2, 1)
ncomp = np.arange(1, nUnits+1)
for i, (a, c, tit) in enumerate(zip([ax[0, 0], ax[0, 1], ax[0, 2]], [cov1, cov2, cov1n], ['1-D noise', '2-D noise', r'1/$n$ noise'])):
    evals, evecs = np.linalg.eig(c)
    evals = evals[np.argsort(evals)[::-1]]
    a.plot(ncomp, evals/evals.sum(), '.-', markersize=10, markerfacecolor='white', color=color)
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel(r'Noise component ($\mathbf{e}_1-\mathbf{e}_N$)')
    a.set_ylim(ylim)
    if i == 0:
        a.set_ylabel('Fraction var. exp.')
    a.set_title(tit)

ylim = (0, 1.1)
ms = 3
cmap = cm.get_cmap('Blues', 100)
for i, (a, c) in enumerate(zip([ax[1, 0], ax[1, 1], ax[1, 2]], ['1d', '2d', '1n'])):
    norm = results[c]['fullRank'][-1, :].mean()
    for d, tit, col in zip(['ddr1', 'ddr2', 'ddr3', 'fullRank'], [r'$dDR_1$', r'$dDR_2$', r'$dDR_3$', 'Full Rank'], [cmap(50), cmap(75), cmap(99), 'grey']):
        m = results[c][d].mean(axis=1)[:-1] / norm
        sd = (results[c][d].std(axis=1)[:-1] / norm) / np.sqrt(nSamples)
        a.plot(krange[:-1], m, '.-', color=col, label=tit, markersize=ms)    
        a.fill_between(krange[:-1], m-sd, m+sd, lw=0, alpha=0.5, color=col)
        xval = krange[-2] + stepSize
        a.errorbar(xval, results[c][d][-1].mean() / norm, yerr=results[c][d][-1].std() / norm, marker='.', markersize=ms, color=col, capsize=2)

    if i == 0:
        a.set_ylabel(r"$d'^2$ (normalized)")
        a.legend(frameon=False)
    a.set_xlabel(r"Sample size ($k$)")
    a.set_ylim(ylim)
f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()
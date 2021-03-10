"""
Simple two-neuron schematic of single trial responses and decoding axis for high / low rep conditions

Shuffled distribution of pairwise covariance for the two neurons in each conditions

Covariance matrix

cross-validated d'^2 vs. k and var(covariance) vs. k. On same axes.
"""
from dDR.utils.decoding import compute_dprime
from dDR.utils.plotting import compute_ellipse

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

np.random.seed(123)

savefig = True
fig_name = os.path.join(os.getcwd(), 'figures/fig1.svg')

# generate pseudo data, X, for decoding
k1 = 10
k2 = 100
u1 = [2, 0]
u2 = [-2, 0]
cov = np.array([[1, 0.4], [0.4, 1]])
x11 = np.random.multivariate_normal(u1, cov, k1)
x21 = np.random.multivariate_normal(u2, cov, k1)
X1 = np.stack([x11, x21])
x12 = np.random.multivariate_normal(u1, cov, k2)
x22 = np.random.multivariate_normal(u2, cov, k2)
X2 = np.stack([x12, x22])

# get decoding axis and projections
d = compute_dprime(x11.T, x21.T)
wopt1 = d.wopt * 2
d = compute_dprime(x12.T, x22.T)
wopt2 = d.wopt * 2

# generate list of shuffled covariance values for k=k1/k2
nshuff = 500
scov1 = []
scov2 = []
for i in range(0, nshuff):
    x = np.random.choice(x11[:, 0], x11.shape[0], replace=False)
    y = np.random.choice(x11[:, 1], x11.shape[0], replace=False)
    scov1.append(np.cov(x, y)[0, 1])
    x = np.random.choice(x12[:, 0], x12.shape[0], replace=False)
    y = np.random.choice(x12[:, 1], x12.shape[0], replace=False)
    scov2.append(np.cov(x, y)[0, 1])

# for a range of k, generate shuffled distros
kscov = []
dp = []
krange = np.arange(10, 400, 2)
_x1 = np.random.multivariate_normal(u1, cov, 1000)
for i, _k in enumerate(krange):
    _s = []
    _dp = []
    print(f"{int((i/len(krange)) * 100)} percent finished generating shuffles")
    for i in range(0, nshuff):
        x = np.random.choice(_x1[:, 0], _k, replace=False)
        y = np.random.choice(_x1[:, 1], _k, replace=False)
        _s.append(np.cov(x, y)[0, 1])
        r1 = np.random.multivariate_normal(u1, cov, _k)
        r2 = np.random.multivariate_normal(u2, cov, _k)
        # get cv dprime w/ simple 50/50 split of data
        est = np.random.choice(np.arange(_k), int(_k/2), replace=False)
        val = np.array(list(set(np.arange(_k)).difference(set(est))))
        # train decoder
        resf = compute_dprime(r1[est, :].T, r2[est, :].T)
        rest = compute_dprime(r1[val, :].T, r2[val, :].T, wopt=resf.wopt)
        _dp.append(abs(rest.dprimeSquared - resf.dprimeSquared))
    dp.append(_dp)
    kscov.append(_s)
dp = np.stack(dp)
kscov = np.stack(kscov)

# plot the results
f, ax = plt.subplots(2, 3, figsize=(6, 4))

# plot decoding figure1
ax[0, 0].scatter(X1[0, :, 0], X1[0, :, 1], s=3, alpha=0.5, lw=0)
ax[0, 0].scatter(X1[1, :, 0], X1[1, :, 1], s=3, alpha=0.5, lw=0)
x, y = compute_ellipse(X1[0, :, 0], X1[0, :, 1])
ax[0, 0].plot(x, y, color='tab:blue', label='Stim. a')
x, y = compute_ellipse(X1[1, :, 0], X1[1, :, 1])
ax[0, 0].plot(x, y, color='tab:orange', label='Stim. b')
# plot decoding axis
ref = np.array(u1) - ((np.array(u1) - np.array(u2)) / 2)
wopt1 = (wopt1 / np.linalg.norm(wopt1)) * 2
ax[0, 0].plot([ref[0]-wopt1[0], ref[0]+wopt1[0]], 
            [ref[1]-wopt1[1], ref[1]+wopt1[1]], 'k-')
ax[0, 0].legend(frameon=False)
ax[0, 0].set_xlabel(r"$n_1$ spike counts")
ax[0, 0].set_ylabel(r"$n_2$ spike counts")
ax[0, 0].axis('equal')
ax[0, 0].set_title(r"$k=%s$"%str(k1))

# plot decoding figure2
ax[0, 1].scatter(X2[0, :, 0], X2[0, :, 1], s=3, alpha=0.5, lw=0)
ax[0, 1].scatter(X2[1, :, 0], X2[1, :, 1], s=3, alpha=0.5, lw=0)
x, y = compute_ellipse(X2[0, :, 0], X2[0, :, 1])
ax[0, 1].plot(x, y, color='tab:blue', label='Stim. a')
x, y = compute_ellipse(X2[1, :, 0], X2[1, :, 1])
ax[0, 1].plot(x, y, color='tab:orange', label='Stim. b')
# plot decoding axis
ref = np.array(u1) - ((np.array(u1) - np.array(u2)) / 2)
wopt2 = (wopt2 / np.linalg.norm(wopt2)) * 2
ax[0, 1].plot([ref[0]-wopt2[0], ref[0]+wopt2[0]], 
            [ref[1]-wopt2[1], ref[1]+wopt2[1]], 'k-')
ax[0, 1].legend(frameon=False)
ax[0, 1].set_xlabel(r"$n_1$ spike counts")
ax[0, 1].set_ylabel(r"$n_2$ spike counts")
ax[0, 1].axis('equal')
ax[0, 1].set_title(r"$k=%s$"%str(k2))

# force a/b to share axes
ax[0, 0].set_xlim((-5, 5)); ax[0, 0].set_ylim((-5, 5))
ax[0, 1].set_xlim((-5, 5)); ax[0, 1].set_ylim((-5, 5))

# plot covariance matrix
ax[0, 2].imshow(cov, cmap='Reds', aspect='auto')
ax[0, 2].set_xticks([0, 1])
ax[0, 2].set_yticks([0, 1])
ax[0, 2].set_xticklabels([r"$n_1$", r"$n_2$"])
ax[0, 2].set_yticklabels([r"$n_1$", r"$n_2$"])
for c1 in np.arange(cov.shape[0]):
    for c2 in np.arange(cov.shape[0]):
        if c1!=c2:
            c = 'k'
        else: 
            c = 'white'
        ax[0, 2].text(c1, c2, str(cov[c1, c2]), 
                        ha='center', va='center', color=c)
ax[0, 2].set_title(r"$\Sigma=\Sigma_a=\Sigma_b$")

# plot shuffled distro for k=k in decoding figure
llim, hlim = (-2, 2)
ax[1, 0].hist(scov1, bins=np.arange(llim, hlim, 0.1), histtype='stepfilled', edgecolor='k', alpha=0.5)
ax[1, 0].axvline(cov[0, 1], color='r')
ax[1, 0].set_xlim((llim, hlim))
ax[1, 0].set_title(r"$k=%s, \sigma^2=%s$" % (str(k1), str(round(np.var(scov1), 3))))
ax[1, 0].set_xlabel(r"$cov(n_1, n_2)$ after shuffling")

ax[1, 1].hist(scov2, bins=np.arange(llim, hlim, 0.1), histtype='stepfilled', edgecolor='k', alpha=0.5)
ax[1, 1].axvline(cov[0, 1], color='r')
ax[1, 1].set_xlim((llim, hlim))
ax[1, 1].set_title(r"$k=%s, \sigma^2=%s$" % (str(k2), str(round(np.var(scov2), 3))))
ax[1, 1].set_xlabel(r"$cov(n_1, n_2)$ after shuffling")

# plot range of shuffle var for different values of k
ax[1, 2].plot(krange, kscov.var(axis=-1))
ax[1, 2].axhline(0, linestyle='--', color='k')
ax[1, 2].set_ylabel(r"$\sigma^2$ of shuff. distribution", color='tab:blue')
ax[1, 2].set_xlabel(r"No. of samples ($k$)")
ax[1, 2].tick_params(axis='y', labelcolor='tab:blue')

# generate twin axes for d'^2
ax2 = ax[1, 2].twinx()
ax2.spines['right'].set_visible(True)
ax2.plot(krange, dp.mean(axis=-1), color='orange')
ax2.set_ylabel(r"Error in $d'^2$ estimate"+"\nabs(Est. - Val.)", color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

f.tight_layout()

if savefig:
    f.savefig(fig_name)

plt.show()

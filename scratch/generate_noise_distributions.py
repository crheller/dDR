"""
WIP to generate different surrogate datasets with specific types of noise distributions.

Eventually, idea would be to write a function that can generate each type
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

# set up population size / parameters
N = 100
u = [4, 1]           # mean of 1, 2
sigu = [1, 1]        # sd of 1, 2 (across neurons). Bigger this is, bigger dU is
# variance of random background "noise" (centered on 0.05) to add to cov. 
# Think of these as generating small, positive correlations to data (small eigenvectors that don't go above noise floor)
cov_delta = 0.1  
spike_noise = 1      # add random noise to spikes that washes out non-sig correlations

# ================================= "GLOBAL" / FIXED properties ================================
# generate mean responses
u1 = np.random.normal(u[0], sigu[0], (N, 1))
u2 = np.random.normal(u[1], sigu[1], (N, 1))

# generate baseline noise level to be added to covariance in all cases
cov_noise = np.random.normal(0.01, 0.1, (N, N))

# generate latent variables
lv1 = generate_lv_loading(N, mean_loading=0.5, variance=1, mag=2)
lv2 = generate_lv_loading(N, mean_loading=u1[:,0]-u2[:,0], variance=0.5, mag=1.8) # one dim roughly like the signal axis
lv3 = generate_lv_loading(N, mean_loading=0.5, variance=1, mag=1.5)
lv_ortho_set = orthonormal(np.concatenate([lv1, lv2, lv3], axis=1))

# =============================== GENERATE DATASETS ====================================
# for each dataset, plot eigenspectrum / covariance matrix, and params used to generate it

# NO SIG DIMENSIONS
cov = cov_noise
cov = cov.dot(cov.T)
evals, evecs = np.linalg.eig(cov)
evals = evals[np.argsort(evals)[::-1]]
f, ax = plt.subplots(1, 3, figsize=(6, 2))
ax[0].imshow(cov, aspect='auto', vmin=-2, vmax=2, cmap='bwr')
ax[1].plot(evals, '.')
ax[2].plot(evals / evals.sum())
f.tight_layout()


# "BASELINE" case -- 1-D, HIGH VARIANCE
cov = generate_low_rank_cov(lv_ortho_set[:, [0]]) + cov_noise
cov = cov.dot(cov.T)
evals, evecs = np.linalg.eig(cov)
evals = evals[np.argsort(evals)[::-1]]
f, ax = plt.subplots(1, 3, figsize=(6, 2))
ax[0].imshow(cov, aspect='auto', vmin=-2, vmax=2, cmap='bwr')
ax[1].plot(evals, '.')
ax[2].plot(evals / evals.sum())
f.tight_layout()

# 1-D, (LOW VARIANCE)
cov = generate_low_rank_cov(lv_ortho_set[:, [0]]*0.8) + cov_noise
cov = cov.dot(cov.T)
evals, evecs = np.linalg.eig(cov)
evals = evals[np.argsort(evals)[::-1]]
f, ax = plt.subplots(1, 3, figsize=(6, 2))
ax[0].imshow(cov, aspect='auto', vmin=-2, vmax=2, cmap='bwr')
ax[1].plot(evals, '.')
ax[2].plot(evals / evals.sum())
f.tight_layout()

# 2-D, (2 sig dims)
cov = generate_low_rank_cov(lv_ortho_set[:, :2]) + cov_noise
cov = cov.dot(cov.T)
evals, evecs = np.linalg.eig(cov)
evals = evals[np.argsort(evals)[::-1]]
f, ax = plt.subplots(1, 3, figsize=(6, 2))
ax[0].imshow(cov, aspect='auto', vmin=-2, vmax=2, cmap='bwr')
ax[1].plot(evals, '.')
ax[2].plot(evals / evals.sum())
f.tight_layout()

# 3-D, (3 sig dims)
cov = generate_low_rank_cov(lv_ortho_set[:, :3]) + cov_noise
cov = cov.dot(cov.T)
evals, evecs = np.linalg.eig(cov)
evals = evals[np.argsort(evals)[::-1]]
f, ax = plt.subplots(1, 3, figsize=(6, 2))
ax[0].imshow(cov, aspect='auto', vmin=-2, vmax=2, cmap='bwr')
ax[1].plot(evals, '.')
ax[2].plot(evals / evals.sum())
f.tight_layout()

f.tight_layout()

plt.show()


def generate_low_rank_cov(lvs):
    """
    Given <dim> loading vectors (N x dim) take outerproduct 
    to generate low-rank cov matrix
    Input:
        lvs: numpy array: (N x dim) where dim is the number of LVs and N is number of neurons
    """
    # make sure lvs form orthonormal set. Never change e1. Just make sure others are orthogonal
    return lvs.dot(lvs.T)

def orthonormal(lvs):
    """
    Force all latent variables to be orthogonal, nut *not* mag 1
    """
    # start at 1, so the 1st (0th) lv is not changes
    for e1 in np.arange(1, lvs.shape[1]):
        # make sure e1 is ortho to all others
        mag = np.linalg.norm(lvs[:, e1])
        _e1 = lvs[:, e1] / mag
        for e2 in range(0, e1):
            if e1!=e2:
                mag2 = np.linalg.norm(lvs[:, e2])
                _e2 = lvs[:, e2] / mag2
                _e1 -= _e1.dot(_e2) * _e2
        lvs[:, e1] = _e1 * mag
    return lvs

def generate_lv_loading(N, mean_loading=0, variance=1, mag=1):
    """
    Input:
        mean_loading: mean loading weight of the latent variables. Can be int/float, or a vector of length N (to make lv aligned with something)
        variance: variance of each loading weight w.r.t "mean_loading"
        mag: magnitude of LV loading weight vector
    Return
        loading weights for a latent variable, shape (N x 1)
    """
    if (type(mean_loading)!=int) & (type(mean_loading)!=float):
        if mean_loading.shape[0] != N:
            raise ValueError("Length of mean_loading must = N")
        else:
            ml = mean_loading / np.linalg.norm(mean_loading)
            lv = np.random.normal(mean_loading, variance)
            lv /= np.linalg.norm(lv)
            lv = lv[:, np.newaxis]
            return lv * mag
    else:
        lv = np.random.normal(mean_loading, variance, (N, 1))
        lv /= np.linalg.norm(lv)
        return lv * mag


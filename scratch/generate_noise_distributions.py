"""
WIP to generate different surrogate datasets with specific types of noise distributions.

Eventually, idea would be to write a function that can generate each type
"""

import numpy as np
import matplotlib.pyplot as plt
import dDR.utils.surrogate_helpers as sh

np.random.seed(123)

N = 100

# mean responses
u = np.random.normal(4, 0.25, (N, 1))

# generate covariance matrix
# step 1 - orthonormal basis
evecs = np.concatenate([sh.generate_lv_loading(N, mean_loading=0, variance=1, mag=1) for n in range(N)], axis=1)
evecs = sh.orthonormal(evecs) * 3

# step 2 - Design eigenspectrum (scale each dimension to shape noise)
evs = 1 / np.arange(1, N+1)**(1/2)
ind = np.random.normal()
# step 3 - scale evecs
# scale correlated components
sevecs = evecs * evs
# scale indep. variance

# step 4 - generate cov matrix with outer product of evecs (inner product would give identity)
cov = sevecs.dot(sevecs.T)
ind = np.ones(cov.shape)
#np.fill_diagonal(ind, 10)
#cov *= ind

# generate stim response data, and add independent noise
k = 100 # "trials"

X = np.random.multivariate_normal(u.squeeze(), cov, k)
#X += np.random.normal(0, .5, X.shape)

# plot the results
f, ax = plt.subplots(1, 3, figsize=(12, 4))

# cov before noise
ax[0].imshow(cov, aspect='auto', vmin=-1, vmax=1, cmap='bwr')
ax[0].set_title("Before adding indep. noise")

# cov after adding indep. noise
ax[1].imshow(np.cov(X.T), aspect='auto', vmin=-1, vmax=1, cmap='bwr')
ax[1].set_title("After adding indep. noise")

# plot 1/f for reference
ax[2].plot(np.arange(1, N+1), 1 / np.arange(1, N+1), '-', label=r"1/f")

# plot "true" eigenspectrum
evals, evecs = np.linalg.eig(cov)
evals = evals[np.argsort(evals)[::-1]]
ax[2].plot(np.arange(1, N+1), evals / evals.sum(), '.', label=r"Eigenvalues (no noise)")

# plot eigenspectrum after adding random noise
evals, evecs = np.linalg.eig(np.cov(X.T))
evals = evals[np.argsort(evals)[::-1]]
ax[2].plot(np.arange(1, N+1), evals / evals.sum(), '.', label=r"Eigenvalues (after ind. noise)")

ax[2].set_yscale('log')
ax[2].set_xscale('log')
ax[2].set_ylabel("Fract. Var. Exp.")
ax[2].set_xlabel('PC')
ax[2].legend(frameon=False)

f.tight_layout()

plt.show()



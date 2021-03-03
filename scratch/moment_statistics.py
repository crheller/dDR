"""
How reliabe are estimates of sample mean / covariance of a multivariate gaussian as a function of sample size?
plot mse of each over sample sizes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# one-D case
nsamps = 1000
u = 5
sd = 2
mean = []
var = []
var_var = []
mean_var = []
n = np.arange(2, 100, 1)
for ii, ss in enumerate(n):
    print(f"Sample {ii} /{len(n)}")
    _mean = []
    _variance = []
    for i in range(0, nsamps):
        sample = np.random.normal(u, sd, ss)
        _mean.append(np.mean(sample))
        _variance.append(np.var(sample, ddof=1))
    mean.append(np.mean(_mean))
    var.append(np.mean(_variance))
    var_var.append(np.var(_variance, ddof=1))
    mean_var.append(np.var(_mean, ddof=1))

f, ax = plt.subplots(2, 2, figsize=(6, 6))

ax[0, 0].plot(n, mean_var, label=f'mean: {u}')
ax[0, 0].plot(n, var_var, label=f'variance: {(sd**2):.3f}')
ax[0, 0].plot(n, (sd**2)/n, lw=5, color='tab:blue', alpha=0.5, zorder=-1, label=r'$\sigma^2$/$n$')
ax[0, 0].plot(n, (2*(sd**2)**2 / ((n-1))), lw=5, color='tab:orange', alpha=0.5, zorder=-1, label=r'$2(\sigma^2)^2$/$(n-1)$')
#ax[0, 0].plot(n, (n * (sd**2)**2) / (n - 1)**2, lw=5, color='tab:orange', alpha=0.5, zorder=-1, label=r'$2(\sigma^2)^2$/$(n-1)$')
ax[0, 0].set_ylabel(f'Var. of param estimate\nacross {nsamps} samples')
ax[0, 0].set_xlabel('Sample size')
ax[0, 0].legend(frameon=False)

ax[0, 1].plot(n, np.array(var_var) / np.array(mean_var))
ax[0, 1].set_ylabel('Ratio')
ax[0, 1].set_xlabel('Sample size')

# ratio of variance in estimate to true param value
ax[1, 0].plot(n, np.array(mean_var) / u, color='tab:blue')
ax[1, 0].set_ylabel('var(mean) / mean')
ax[1, 0].set_xlabel('Sample size')

ax[1, 1].plot(n, np.array(var_var) / (sd**2), color='tab:orange')
ax[1, 1].set_ylabel('var(var) / var')
ax[1, 1].set_xlabel('Sample size')

f.tight_layout()

plt.show()

# two-D case -- do covariance values behave like variance?
nsamps = 1000
u = [2, 2] # "z-scored data"
cov = np.array([[1, 0.4], [0.4, 1]])
mean = []
var = []
var_var = []
mean_var = []
n = np.arange(2, 100, 1)
for ii, ss in enumerate(n):
    print(f"Sample {ii} /{len(n)}")
    _mean = []
    _variance = []
    for i in range(0, nsamps):
        sample = np.random.multivariate_normal(u, cov, ss)
        _mean.append(np.mean(sample, axis=0))
        _variance.append(np.cov(sample.T))
    mean.append(np.stack(_mean).mean(axis=0))
    var.append(np.stack(_variance).mean(axis=0))
    var_var.append(np.stack(_variance).var(axis=0))
    mean_var.append(np.stack(_mean).var(axis=0))

mean = np.stack(mean)
var = np.stack(var)
var_var = np.stack(var_var)
mean_var = np.stack(mean_var)

f, ax = plt.subplots(2, 3, figsize=(13, 8))

# mean
ax[0, 0].plot(n, mean_var[:, 0], label=f'mean[0]: {u[0]}')
ax[0, 0].plot(n, mean_var[:, 1], label=f'mean[1]: {u[1]}')
# variance
ax[0, 0].plot(n, var_var[:, 0, 0], label=f'variance[0, 0]: {(cov[0, 0]):.3f}')
ax[0, 0].plot(n, ((cov[0, 0])**2 / (0.5*(n-1))), lw=5, color='tab:green', alpha=0.5, zorder=-1, label=r'$2(\sigma^2)^2$/$(n-1)$')
ax[0, 0].plot(n, var_var[:, 1, 1], label=f'variance[1, 1]: {(cov[1, 1]):.3f}')
# covariance
ax[0, 0].plot(n, var_var[:, 0, 1], label=f'covariance[0, 1]: {(cov[0, 1]):.3f}')
ax[0, 0].plot(n, (2*cov[0, 1]**2 + (cov[0, 0]*cov[1,1]) - cov[0, 1] ) / (n-1), lw=5, color='tab:purple', alpha=0.5, zorder=-1, \
                                label=r'$(2\Sigma_{0,1}^2 + \sigma_0^2\sigma_1^2 - \Sigma_{0, 1})$/$(n-1)$')
#ax[0, 0].plot(n, (n * (cov[0, 0] * cov[1, 1]) / (n-1)**2), lw=5, color='tab:purple', alpha=0.5, zorder=-1, label=r'$(\sigma_0^2\sigma_1^2)$/$(n-1)$')
ax[0, 0].plot(n, var_var[:, 1, 0], label=f'covariance[1, 0]: {(cov[1, 0]):.3f}')
ax[0, 0].set_ylabel(f'Var. of param estimate\nacross {nsamps} samples')
ax[0, 0].set_xlabel('Sample size')
ax[0, 0].legend(frameon=False)

ax[0, 1].plot(n, var_var[:, 0, 0] / mean_var[:, 0], label='cov[0, 0]:mean[0]')
ax[0, 1].plot(n, var_var[:, 0, 1] / mean_var[:, 0], label='cov[0, 1]:mean[0]')
ax[0, 1].plot(n, var_var[:, 0, 1] / mean_var[:, 1], label='cov[0, 1]:mean[1]')
ax[0, 1].set_ylabel('Ratio')
ax[0, 1].set_xlabel('Sample size')
ax[0, 1].legend(frameon=False)

# ratio of variance in estimate to true param value
ax[1, 0].plot(n, mean_var[:, 0] / u[0], color='tab:blue', label='mean[0]')
ax[1, 0].set_ylabel('var(mean) / mean')
ax[1, 0].set_xlabel('Sample size')
ax[1, 0].legend(frameon=False)

ax[1, 1].plot(n, var_var[:, 0, 0] / cov[0, 0], color='tab:green', label='cov[0, 0]')
ax[1, 1].plot(n, var_var[:, 0, 1] / cov[0, 1], color='tab:purple', label='cov[0, 1]')
ax[1, 1].set_ylabel('var(var) / var')
ax[1, 1].set_xlabel('Sample size')
ax[1, 1].legend(frameon=False)

# ratio of % errs
ax[1, 2].plot(n, (mean_var[:, 0] / u[0]) / (var_var[:, 0, 0] / cov[0, 0]), label='mean[0]:cov[0, 0]')
ax[1, 2].plot(n, (mean_var[:, 0] / u[0]) / (var_var[:, 0, 1] / cov[0, 1]), label='mean[0]:cov[0, 1]')
ax[1, 2].plot(n, (var_var[:, 0, 0] / cov[0, 0]) / (var_var[:, 0, 1] / cov[0, 1]), label='cov[0, 0]:cov[0, 1]')

ax[1, 2].legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')

ax[1, 2].set_xlabel('Sample size')
ax[1, 2].set_ylabel('Ratio')

f.tight_layout()

plt.show()


# final two panel figure just illustrating the variance in estiamte of mean, variance, and covariance.
# second panel shows covariance matrix
f, ax = plt.subplots(1, 2, figsize=(8, 4))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# arbitrary-D case -- how do pairwise covariance values behave?
nsamps = 1000
u = [0, 0, 0] # "z-scored data"
cov = np.array([[.4, 0.2, -0.2], [0.2, .2, 0.2], [-0.2, 0.2, .1]])
u = np.zeros(10)
lv = abs(np.random.normal(0, 1, (10, 1)))
lv /= np.linalg.norm(lv)
cov = lv.dot(lv.T)
cov += np.random.normal(0, 0.1, cov.shape)
np.fill_diagonal(cov, np.random.randint(1, 2, size=cov.shape[0]))
cov = cov.dot(cov.T)
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
    var_var.append(np.stack(_variance).var(axis=0, ddof=1))
    mean_var.append(np.stack(_mean).var(axis=0, ddof=1))

mean = np.stack(mean)
var = np.stack(var)
var_var = np.stack(var_var)
mean_var = np.stack(mean_var)

# plot variance in u (mean_var), sigma (diag(var_var)), and cov (off diagonal var_var) for different sample sizes
f, ax = plt.subplots(1, 2, figsize=(8, 4))

# mean
ax[0].plot(n, mean_var[:, 0], label=r"$\mu_0$")
ax[0].plot(n, (abs(cov[0, 0]) / (n)), lw=5, color='tab:blue', alpha=0.5, zorder=-1, label=r'$\sigma_0^2$/$n$')

# variance
ax[0].plot(n, var_var[:, 0, 0], label=r"$\Sigma_{0, 0}$")
ax[0].plot(n, (2*(abs(cov[0, 0]))**2 / ((n-1))), lw=5, color='tab:orange', alpha=0.5, zorder=-1, label=r'$2\sigma_0^4$/$(n-1)$')

# covariance
ax[0].plot(n, var_var[:, 0, 1], label=r"$\Sigma_{0, 1}$")
ax[0].plot(n, abs(2*cov[0, 1]**2 + (cov[0, 0]*cov[1,1]) - (cov[0, 1])) / (n-1), lw=5, color='tab:green', alpha=0.5, zorder=-1, \
                                label=r'$(2\Sigma_{0,1}^2 + \sigma_0^2\sigma_1^2 - \Sigma_{0, 1})$/$(n-1)$')

ax[0].set_ylabel(f'Var. of param estimate\nacross {nsamps} samples')
ax[0].set_xlabel('Sample size')
ax[0].legend(frameon=False)

# plot covariance matrix
sns.heatmap(cov, cmap='Blues', annot=True, ax=ax[1], vmin=0, vmax=1, xticklabels=False, yticklabels=False, cbar=False)
ax[1].set_title(r"$\Sigma$")

f.tight_layout()

plt.show()
"""
How reliabe are estimates of sample mean / covariance of a multivariate gaussian as a function of sample size?
plot mse of each over sample sizes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

N = 10
size = np.arange(N, 1000, 10)
u = np.random.normal(0, 1, N) # True means
lv = np.abs(np.random.normal(0, 1, (N, 1)))
lv /= np.linalg.norm(lv)
cov = lv.dot(lv.T)
cov += np.random.normal(0, 0.1, cov.shape) # true covariance
merr = []
varerr = []
pc1 = []
for n in size:
    print(f"n {n}")
    m = []
    v = []
    p = []
    for i in range(0, 100):
        vals = np.random.multivariate_normal(u, cov, n)
        m.append(u - vals.mean(axis=0))
        v.append(cov.flatten() - np.cov(vals.T).flatten())
        _, e = np.linalg.eig(np.cov(vals.T))
        p.append((e[:, 0].dot(lv)[0]))
    merr.append(np.stack(np.abs(m)).mean(axis=0))
    varerr.append(np.stack(np.abs(v)).mean(axis=0))
    pc1.append(np.mean(np.abs(p)))

merr = np.stack(merr)
varerr = np.stack(varerr)

f, ax = plt.subplots(1, 1, figsize=(5, 5))

m = (merr / abs(u)).mean(axis=-1)
sd = (merr / abs(u)).std(axis=-1)
ax.plot(size, m)
ax.fill_between(size, m-sd, m+sd, alpha=0.5, lw=0)
m = (varerr / abs(cov.flatten())).mean(axis=-1)
sd = (varerr / abs(cov.flatten())).std(axis=-1)
ax.plot(size, m)
ax.fill_between(size, m-sd, m+sd, alpha=0.5, lw=0)
ax.plot(size, pc1)
#ax.set_ylim((0, 1))
ax.set_ylabel("Percentage error")
ax.axvline(N, linestyle='--', color='k')

ax.set_title(f"Cov: {cov[0,1]:.4f}, Mean: {u[0]:.4f}")

f.tight_layout()

plt.show()

# one-D case
nsamps = 1000
u = 5
sd = 2
mean = []
var = []
var_var = []
mean_var = []
n = np.arange(2, 100, 1)
for ss in n:
    _mean = []
    _variance = []
    for i in range(0, nsamps):
        sample = np.random.normal(u, sd, ss)
        _mean.append(np.mean(sample))
        _variance.append(np.var(sample))
    mean.append(np.mean(_mean))
    var.append(np.mean(_variance))
    var_var.append(np.var(_variance))
    mean_var.append(np.var(_mean))

f, ax = plt.subplots(2, 2, figsize=(6, 6))

ax[0, 0].plot(n, mean_var, label=f'mean: {u}')
ax[0, 0].plot(n, var_var, label=f'variance: {(sd**2):.3f}')
ax[0, 0].set_ylabel(f'Var. of param estimate\nacross {nsamps} samples')
ax[0, 0].set_xlabel('Sample size')
ax[0, 0].legend(frameon=False)

ax[0, 1].plot(n, np.array(var_var) / np.array(mean_var))
ax[0, 1].set_ylabel('Ratio')
ax[0, 1].set_xlabel('Sample size')

# ratio of variance in estimate to true param value
ax[1, 0].plot(n, np.array(mean_var)/mean, color='tab:blue')
ax[1, 0].set_ylabel('var(mean) / mean')
ax[1, 0].set_xlabel('Sample size')

ax[1, 1].plot(n, np.array(var_var)/(sd**2), color='tab:orange')
ax[1, 1].set_ylabel('var(var) / var')
ax[1, 1].set_xlabel('Sample size')

f.tight_layout()

plt.show()


# two-D case -- do covariance values behave like variance?
nsamps = 1000
u = [5, 5]
cov = np.array([[2, 0.4], [0.4, 2]])
mean = []
var = []
var_var = []
mean_var = []
n = np.arange(2, 100, 1)
for ss in n:
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

f, ax = plt.subplots(2, 2, figsize=(6, 6))

ax[0, 0].plot(n, mean_var, label=f'mean: {u}')
ax[0, 0].plot(n, var_var, label=f'variance: {(sd**2):.3f}')
ax[0, 0].set_ylabel(f'Var. of param estimate\nacross {nsamps} samples')
ax[0, 0].set_xlabel('Sample size')
ax[0, 0].legend(frameon=False)

ax[0, 1].plot(n, np.array(var_var) / np.array(mean_var))
ax[0, 1].set_ylabel('Ratio')
ax[0, 1].set_xlabel('Sample size')

# ratio of variance in estimate to true param value
ax[1, 0].plot(n, np.array(mean_var)/mean, color='tab:blue')
ax[1, 0].set_ylabel('var(mean) / mean')
ax[1, 0].set_xlabel('Sample size')

ax[1, 1].plot(n, np.array(var_var)/(sd**2), color='tab:orange')
ax[1, 1].set_ylabel('var(var) / var')
ax[1, 1].set_xlabel('Sample size')

f.tight_layout()

plt.show()

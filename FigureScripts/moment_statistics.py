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
        m.append(u[0] - vals[:, 0].mean())
        v.append(cov[0, 1] - np.cov(vals.T)[0, 1])
        _, e = np.linalg.eig(np.cov(vals.T))
        p.append((e[:, 0].dot(lv)[0]))
    merr.append(np.mean(np.abs(m)))
    varerr.append(np.mean(np.abs(v)))
    pc1.append(np.mean(np.abs(p)))

f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(size, np.array(merr) / abs(u[0]))
ax.plot(size, np.array(varerr) / abs(cov[0, 1]))
ax.plot(size, pc1)
#ax.set_ylim((0, 1))
ax.set_ylabel("Percentage error")
ax.axvline(N, linestyle='--', color='k')

ax.set_title(f"Cov: {cov[0,1]:.4f}, Mean: {u[0]:.4f}")

f.tight_layout()

plt.show()
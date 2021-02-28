import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def compute_ellipse(x, y):
    inds = np.isfinite(x) & np.isfinite(y)
    x= x[inds]
    y = y[inds]
    data = np.vstack((x, y))
    mu = np.mean(data, 1)
    data = data.T - mu

    D, V = np.linalg.eig(np.divide(np.matmul(data.T, data), data.shape[0] - 1))
    # order = np.argsort(D)[::-1]
    # D = D[order]
    # V = abs(V[:, order])

    t = np.linspace(0, 2 * np.pi, 100)
    e = np.vstack((np.sin(t), np.cos(t)))  # unit circle
    VV = np.multiply(V, np.sqrt(D))  # scale eigenvectors
    e = np.matmul(VV, e).T + mu  # project circle back to orig space
    e = e.T

    return e[0], e[1]


def plot_stim_pair_dDR(A, B, lab1=None, lab2=None, xlab=r"$dDR_1 (\Delta \mu)$", ylab=r"$dDR_2$", s=10, lw=1, ax=None):
    """
    Plot stimulus pair in dDR space (first two dims only, if more than that)
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    else:
        ax.scatter(A[:, 0], A[:, 1], s=s)
        x, y = compute_ellipse(A[:, 0], A[:, 1])
        ax.plot(x, y, lw=lw, label=lab1)
        ax.scatter(B[:, 0], B[:, 1], s=s)
        x, y = compute_ellipse(B[:, 0], B[:, 1])
        ax.plot(x, y, lw=lw, label=lab1)
    
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    return ax
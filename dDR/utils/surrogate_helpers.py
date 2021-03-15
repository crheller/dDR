"""
Helper functions for generating surrogate data
"""
import numpy as np

def generate_full_rank_cov(lvs):
    """
    Using latent variables, generate low rank cov matrix, then add noise to make full rank
    Force positive semi-definite
    """
    cov = lvs.dot(lvs.T)
    return cov

def orthonormal(lvs):
    """
    Force all latent variables to be orthogonal, and mag 1
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
        lvs[:, e1] = _e1 / np.linalg.norm(_e1)
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


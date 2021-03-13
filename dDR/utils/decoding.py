"""
Functions for computing d-prime from a pair of response distributions
"""
from collections import namedtuple
import numpy as np
import logging

log = logging.getLogger()

def compute_dprime(A, B, diag=False, wopt=None, suppress_log=False):
    """
    Compute discriminability between matrix A and matrix B
    where both are shape N neurons X N reps.

    A / B: N x k numpy arrays
    diag: bool, if True, compute dprime assuming diagonal covariance matrix
    wopt: N x 1 numpy array -- if not None, use this axis as the decoding axis 

    Return:
        dprime ** 2 (with sign preserved... so that we can evaluate consitency of sign btwn est/val sets)
        decoding axis (with norm 1)
        evals: (of mean covariance matrix)
        evecs: (of mean covariance matrix)
        evec_sim: similarity of first eigenvector between the two stimuli, A and B
        dU: <A> - <B>
    """
    if (A.shape[0] > (A.shape[1] + B.shape[1])) & (wopt is None):
        if not suppress_log:
            log.warning("Number of dimensions greater than number of observations. Unstable performance!")

    if A.shape[0] != B.shape[0]:
        raise ValueError("Number of dimensions do not match between conditions")

    if A.shape[0] == 1:
        # single dimension
        dprime = _dprime_single_dim(A, B)
        return dprime

    if diag:
        dprime, wopt, evals, evecs, evec_sim, dU = _dprime_diag(A, B)

    else:
        dprime, wopt, evals, evecs, evec_sim, dU = _dprime(A, B, wopt=wopt)

    dprime_results = {
        'dprimeSquared': dprime,
        'wopt': wopt,
        'evals': evals,
        'evecs': evecs,
        'evecSim': evec_sim,
        'dU': dU
    }

    return namedtuple('DecodingResults', dprime_results.keys())(**dprime_results)


def _dprime_single_dim(A, B):
    """
    Simplest case of single neuron (dimension) dprime -- classic signal detection theory
    u1 - u2 / sqrt(0.5 * (sig1 + sig2))
    """
    num = A.mean() - B.mean()
    den = np.sqrt(0.5 * (A.var() + B.var()))
    if num == 0:
        return 0
    elif den == 0:
        return np.inf
    else:
        return num / den


def _dprime(A, B, wopt=None):
    """
    See Rumyantsev et. al 2020, Nature for nice derivation
    """

    sigA = np.cov((A - A.mean(axis=-1, keepdims=True)))
    sigB = np.cov((B - B.mean(axis=-1, keepdims=True)))

    usig = 0.5 * (sigA + sigB)
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]    

    try:
        valA, vecA = np.linalg.eig(sigA)
        valB, vecB = np.linalg.eig(sigB)
        evec_sim = abs(vecB[:, np.argsort(valB)[::-1][0]].dot(vecA[:, np.argsort(valA)[::-1][0]]))
    except:
        evec_sim = np.nan

    if wopt is not None:
        wopt_train = wopt / np.linalg.norm(wopt)
        A = A.T.dot(wopt_train).T
        B = B.T.dot(wopt_train).T

        usig_ = 0.5 * (np.cov((A - A.mean(axis=-1, keepdims=True))) + np.cov((B - B.mean(axis=-1, keepdims=True))))
        u_vec_ = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]
    
    try:
        if wopt is not None:
            wopt = (1 / usig_) * u_vec_
            dp2 = np.matmul(u_vec_, wopt)[0][0]
            try:
                # if wopt is passed, could still compute dpirme but can't compute 
                # evecs/ evals
                evals, evecs = np.linalg.eig(usig)
                # make sure evals / evecs are sorted
                idx_sort = np.argsort(evals)[::-1]
                evals = evals[idx_sort]
                evecs = evecs[:, idx_sort]
            except:
                wopt = np.nan * np.ones((A.shape[0], 1))
                evals = np.nan * np.ones((A.shape[0], ))
                evecs = np.nan * np.ones((A.shape[0], A.shape[0]))

        else:
            inv = np.linalg.inv(usig)
            wopt = inv @ u_vec.T
            dp2 = np.matmul(u_vec, wopt)[0][0]

            evals, evecs = np.linalg.eig(usig)
            # make sure evals / evecs are sorted
            idx_sort = np.argsort(evals)[::-1]
            evals = evals[idx_sort]
            evecs = evecs[:, idx_sort]

    except:
        log.info('WARNING, Singular Covariance, dprime infinite, set to np.nan')
        wopt_nan = np.nan * np.ones((A.shape[0], 1))
        evals_nan = np.nan * np.ones((A.shape[0], ))
        evecs_nan = np.nan * np.ones((A.shape[0], A.shape[0]))
        u_vec_nan =  np.nan * np.ones((1, A.shape[0]))
        return np.nan, wopt_nan, evals_nan, evecs_nan, np.nan, u_vec_nan

    return dp2, wopt, evals, evecs, evec_sim, u_vec


def _dprime_diag(A, B):
    """
    See Rumyantsev et. al 2020, Nature  and Averbeck 2006, JNP for nice derivations.
        Note typo in Rumyantsev though!
    """

    sigA = np.cov((A - A.mean(axis=-1, keepdims=True)))
    sigB = np.cov((B - B.mean(axis=-1, keepdims=True)))

    usig = 0.5 * (sigA + sigB)
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]    

    try:
        valA, vecA = np.linalg.eig(sigA)
        valB, vecB = np.linalg.eig(sigB)
        evec_sim = abs(vecB[:, np.argsort(valB)[::-1][0]].dot(vecA[:, np.argsort(valA)[::-1][0]]))
    except:
        evec_sim = np.nan

    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]

    try:
        # get diagonal covariance matrix
        usig_diag = np.zeros(usig.shape)
        np.fill_diagonal(usig_diag, np.diagonal(usig))

        # compute numerator
        numerator = (u_vec @ np.linalg.inv(usig_diag) @ u_vec.T) ** 2

        # compute denominator
        denominator = u_vec @ np.linalg.inv(usig_diag) @ usig @ np.linalg.inv(usig_diag) @ u_vec.T
        denominator = denominator[0][0]
    except np.linalg.LinAlgError:
        log.info('WARNING, Singular Covariance, dprime infinite, set to np.nan')
        wopt_nan = np.nan * np.ones((A.shape[0], 1))
        evals_nan = np.nan * np.ones((A.shape[0], ))
        evecs_nan = np.nan * np.ones((A.shape[0], A.shape[0]))
        u_vec_nan =  np.nan * np.ones((1, A.shape[0]))
        return np.nan, wopt_nan, evals_nan, evecs_nan, np.nan, u_vec_nan

    dp2 = (numerator / denominator).squeeze()

    evals, evecs = np.linalg.eig(usig)
    # make sure evals / evecs are sorted
    idx_sort = np.argsort(evals)[::-1]
    evals = evals[idx_sort]
    evecs = evecs[:, idx_sort]

    # best decoding axis ignoring correlations (reduces to direction of u_vec)
    wopt_diag = np.linalg.inv(usig_diag) @ u_vec.T

    return dp2, wopt_diag, evals, evecs, evec_sim, u_vec

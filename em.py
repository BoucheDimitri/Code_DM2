import numpy as np
import scipy.stats as stats


def g_pdfs(mus, sigmas):
    """
    List of Gaussian pdf functions for the different parameters: [f(., mu_1, sigma_1),..., f(., mu_k, sigma_k)]

    Params:
        mus (np.ndarray): the mus stacked in columns (nfeatures, ngaussians)
        sigmas (list): list of covariance matrices, len(sigmas) = ngaussians and sigmas[j] = sigma_j

    Returns:
        list: a list of func
    """
    k = len(sigmas)
    pdfs = []
    for j in range(0, k):
        gj = lambda xvec: stats.multivariate_normal.pdf(xvec, mus[:, j], sigmas[j])
        pdfs.append(gj)
    return pdfs


def log_g_pdfs(mus, sigmas):
    """
    List of Gaussian logpdf functions for the different parameters: [log f(., mu_1, sigma_1),..., log f(., mu_k, sigma_k)]

    Params:
        mus (np.ndarray): the mus stacked in columns (nfeatures, ngaussians)
        sigmas (list): list of covariance matrices, len(sigmas) = ngaussians and sigmas[j] = sigma_j

    Returns:
        list: a list of func
    """
    k = len(sigmas)
    pdfs = []
    for j in range(0, k):
        gj = lambda xvec: stats.multivariate_normal.logpdf(xvec, mus[:, j], sigmas[j])
        pdfs.append(gj)
    return pdfs


def log_pz_given_x(x, pi, mus, sigmas):
    """
    p(z|x, pi, mus, sigmas)

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        pi (np.ndarray): multinomial mixture distribution, (ngaussians, )
        mus (np.ndarray): the mus stacked in columns (nfeatures, ngaussians)
        sigmas (list): list of covariance matrices, len(sigmas) = ngaussians and sigmas[j] = sigma_j

    Returns:
        np.ndarray: matrix which ij-th entry is p(z=j|x_i, pi, mus, sigmas)
    """
    k = pi.shape[0]
    n = x.shape[1]
    pzgx = np.zeros((k, n))
    pdfs = g_pdfs(mus, sigmas)
    for i in range(0, n):
        sumj = 0
        for j in range(0, k):
            pzgx_ij = pi[j] * pdfs[j](x[:, i])
            pzgx[j, i] = pzgx_ij
            sumj += pzgx_ij
        pzgx[:, i] *= (1 / sumj)
    return pzgx


def log_gmatrix(x, k, mus, sigmas):
    n = x.shape[1]
    log_pdfs = log_g_pdfs(mus, sigmas)
    gmat = np.zeros((k, n))
    for i in range(0, n):
        for j in range(0, k):
            gmat[j, i] = log_pdfs[j](x[:, i])
    return gmat


def e_computation(x, pi_t, mus_t, sigmas_t, pi_tplus1, mus_tplus1, sigmas_tplus1):
    k = pi_t.shape[0]
    pzgx = log_pz_given_x(x, pi_t, mus_t, sigmas_t)
    pi_term = np.dot(np.sum(pzgx, axis=1), np.log(pi_tplus1))
    gmat = log_gmatrix(x, k, mus_tplus1, sigmas_tplus1)
    mus_sigs_term = np.sum(gmat * pzgx)
    return pi_term + mus_sigs_term


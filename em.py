import numpy as np


def multi_gaussian(xvec, mu, sigma):
    d = xvec.shape[0]
    norm = np.power(np.pi, 0.5 * d) * np.sqrt(np.linalg.det(sigma))
    sigma_inv = np.linalg.inv(sigma)
    intraexp = - 0.5 * np.dot(np.dot((xvec - mu).T, sigma_inv), xvec - mu)
    return (1 / norm) * np.exp(intraexp)


def pz_given_x(x, pi, mus, sigmas):
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
    for j in range(k):
        for i in range(n):
            pzgx[j, i] = pi[j] * multi_gaussian(x[:, i], mus[:, j], sigmas[j])
    for i in range(0, n):
        pzgx[:, i] *= (1 / np.sum(pzgx[:, i]))
    pzgx /= pzgx.sum(0)
    return pzgx


def log_gmatrix(x, mus, sigmas):
    """
    Matrix which entries are the log p(x_i|z=j, mus, sigmas)

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        mus (np.ndarray): the mus stacked in columns (nfeatures, ngaussians)
        sigmas (list): list of covariance matrices, len(sigmas) = ngaussians and sigmas[j] = sigma_j

    Returns:
        np.ndarray: Matrix which entries are the log p(x_i|z=j, mus, sigmas)
    """
    n = x.shape[1]
    k = len(sigmas)
    gmat = np.zeros((k, n))
    for i in range(0, n):
        for j in range(0, k):
            gmat[j, i] = np.log(multi_gaussian(x[:, i], mus[:, j], sigmas[j]))
    return gmat


def test_propto_eye(sigmas):
    test = (1 / sigmas[0][0, 0]) * sigmas[0] - np.eye(sigmas[0].shape[0]) == np.zeros(sigmas[0].shape)
    if np.all(test):
        return 1
    else:
        return 0


def e_computation(x, pi_t, mus_t, sigmas_t, pi_tplus1, mus_tplus1, sigmas_tplus1):
    """
    Computation of E_(z | x, mus_tplus1, sigmas_tplus1) [log p(x, z|pi_t, mus_t, sigmas_t)]

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        pi_t (np.ndarray): multinomial mixture distribution at t (ngaussians, )
        mus_t (np.ndarray): the mus at t stacked in columns (nfeatures, ngaussians)
        sigmas_t (list): list of covariance matrices at t, len(sigmas) = ngaussians
        pi_tplus1 (np.ndarray): multinomial mixture distribution at t + 1 (ngaussians, )
        mus_tplus1 (np.ndarray): the mus at t + 1 stacked in columns (nfeatures, ngaussians)
        sigmas_tplus1 (list): list of covariance matrices at t + 1, len(sigmas) = ngaussians

    Returns:
        float: E_(z | x, mus_tplus1, sigmas_tplus1) [log p(x, z|pi_t, mus_t, sigmas_t)]
    """
    pzgx = pz_given_x(x, pi_t, mus_t, sigmas_t)
    pi_term = np.dot(np.sum(pzgx, axis=1), np.log(pi_tplus1))
    gmat = log_gmatrix(x, mus_tplus1, sigmas_tplus1)
    mus_sigs_term = np.sum(gmat * pzgx)
    return pi_term + mus_sigs_term


def m_step_pi(pzgx):
    pi_tplus1 = (1 / np.sum(pzgx)) * np.sum(pzgx, axis=1)
    return pi_tplus1


def m_step_mus(x, pzgx):
    d = x.shape[0]
    k = pzgx.shape[0]
    mus_tplus1 = np.zeros((d, k))
    for j in range(0, k):
        mus_tplus1[:, j] = (1 / np.sum(pzgx[j, :])) * np.sum(pzgx[j, :] * x, axis=1)
    return mus_tplus1


def m_step_sigmas_diag(x, pzgx, mus_tplus1):
    n = x.shape[1]
    k = pzgx.shape[0]
    d = x.shape[0]
    sigmas1d = np.zeros((k, ))
    for j in range(0, k):
        for i in range(0, n):
            xcij = x[:, i] - mus_tplus1[:, j]
            sigmas1d[j] += pzgx[j, i] * np.dot(xcij.T, xcij)
        sigmas1d[j] *= (1 / np.sum(pzgx[j, :]))
    sigmas_tplus1 = []
    for j in range(0, k):
        sigmas_tplus1.append(sigmas1d[j] * np.eye(d))
    return sigmas_tplus1


def m_step_sigmas(x, pzgx, mus_tplus1):
    n = x.shape[1]
    k = pzgx.shape[0]
    d = x.shape[0]
    sigmas_tplus1 = []
    for j in range(0, k):
        sigmas_tplus1.append(np.zeros((d, d)))
    for j in range(0, k):
        for i in range(0, n):
            xcij = (x[:, i] - mus_tplus1[:, j]).reshape(d, 1)
            sigmas_tplus1[j] += pzgx[j, i] * np.dot(xcij, xcij.T)
        sigmas_tplus1[j] *= (1 / np.sum(pzgx[j, :]))
    return sigmas_tplus1


def m_step(x, pi_t, mus_t, sigmas_t):
    pzgx = pz_given_x(x, pi_t, mus_t, sigmas_t)
    pi_tplus1 = m_step_pi(pzgx)
    mus_tplus1 = m_step_mus(x, pzgx)
    sigmas_tplus1 = m_step_sigmas(x, pzgx, mus_tplus1)
    return pi_tplus1, mus_tplus1, sigmas_tplus1


def m_step_diag(x, pi_t, mus_t, sigmas_t):
    pzgx = pz_given_x(x, pi_t, mus_t, sigmas_t)
    pi_tplus1 = m_step_pi(pzgx)
    mus_tplus1 = m_step_mus(x, pzgx)
    sigmas_tplus1 = m_step_sigmas_diag(x, pzgx, mus_tplus1)
    return pi_tplus1, mus_tplus1, sigmas_tplus1


def iterate_em(x, pi_0, mus_0, sigmas_0, maxit, epsilon, diag=False):
    qexpecs = [np.inf]
    pi_t, mus_t, sigmas_t = pi_0, mus_0, sigmas_0
    for t in range (0, maxit):
        if diag:
            pi_tplus1, mus_tplus1, sigmas_tplus1 = m_step_diag(x, pi_t, mus_t, sigmas_t)
        else:
            pi_tplus1, mus_tplus1, sigmas_tplus1 = m_step(x, pi_t, mus_t, sigmas_t)
        qexpec = e_computation(x, pi_t, mus_t, sigmas_t, pi_tplus1, mus_tplus1, sigmas_tplus1)
        qexpecs.append(qexpec)
        if np.abs(qexpecs[t + 1] - qexpecs[t]) < epsilon:
            return pi_tplus1, mus_tplus1, sigmas_tplus1, qexpecs[1:]
        pi_t, mus_t, sigmas_t = pi_tplus1, mus_tplus1, sigmas_tplus1
        print(t)
    return pi_tplus1, mus_tplus1, sigmas_tplus1, qexpecs[1:]


def assign_cluster(x, pi, mus, sigmas):
    pzgx = pz_given_x(x, pi, mus, sigmas)
    maxz = np.argmax(pzgx, axis=0)
    return maxz


# cov[j]=np.sum(tau[:,j] * np.sum((x-np.tensordot(mu[j],np.ones(n),0))**2,axis=0))/(p*np.sum(tau[:,j]))*np.eye(p)
import numpy as np


def haar_father(x):
    return ((x >= 0) & (x <= 1)) * 1.0


def haar_mother(x):
    return np.where(x > 0.5, 1, -1) * ((x >= 0) & (x <= 1))


phi = haar_father


def psi(j, k, x):
    return 2 ** (j / 2) * haar_mother(2 ** j * x - k)


def approx_beta_wavelet(j, x, y):
    x = x.reshape(-1, 1)
    k = np.arange(2 ** j).reshape(1, -1)
    return np.mean(psi(j, k, x) * y.reshape(-1, 1), axis=0)


def wavelet_approx(alpha, beta):
    def func(xs):
        return alpha * phi(xs) + sum(
            np.sum(beta_j.reshape(1, -1) * psi(j, np.arange(2 ** j).reshape(1, -1), xs.reshape(-1, 1)), axis=1) for j, beta_j in enumerate(beta)
        )
    return func


def fit_wavelet(J, x, y):
    n = x.shape[0]
    alpha_hat = np.mean(phi(x) * y)
    D = [approx_beta_wavelet(j, x, y) for j in range(J)]
    sigma_hat = np.sqrt(n) * np.median(np.abs(D[-1])) / 0.6745
    beta_hat = [
        np.where(np.abs(d) > sigma_hat * np.sqrt(2 * np.log(n) / n), d, 0) for d in D
    ]
    return wavelet_approx(alpha_hat, beta_hat)

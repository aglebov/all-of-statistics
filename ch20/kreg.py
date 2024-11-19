import numpy as np
import scipy.stats as stats


def w(K, xi, h, x):
    u = (x.reshape(-1, 1) - xi.reshape(1, -1)) / h
    w = K(u)
    return w / np.sum(w, axis=1, keepdims=True)


def r(K, xi, yi, h, x):
    wi = w(K, xi, h, x)
    return np.sum(yi.reshape(1, -1) * wi, axis=1)


def cross_validation_score_reg(K, xi, yi, h):
    u = (xi.reshape(-1, 1) - xi.reshape(1, -1)) / h
    ri = r(K, xi, yi, h, xi)
    return np.sum(
        (yi - ri) ** 2 / 
        (1 - K(0) / np.sum(K(u), axis=1)) ** 2
    )


def kreg_conf_int(K, xi, yi, h, alpha, xs):
    n = xi.shape[0]
    r_hat = r(K, xi, yi, h, xs)
    m = (np.max(xi) - np.min(xi)) / 3 / h
    q = stats.norm.ppf((1 + (1 - alpha) ** (1 / m)) / 2)
    sigma = np.sqrt(np.sum((yi[1:] - yi[:-1]) ** 2) / 2 / (n - 1))
    se = sigma * np.sqrt(np.sum(w(K, xi, h, xs) ** 2, axis=1))
    l = r_hat - q * se
    u = r_hat + q * se
    return r_hat, l, u

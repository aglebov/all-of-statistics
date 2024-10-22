import numpy as np
import scipy.stats as stats


def fisher_transform(r):
    return (np.log(1 + r) - np.log(1 - r)) / 2


def fisher_transform_inv(z):
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def fisher_conf_int(rho_hat, n, alpha):
    return fisher_transform_inv(fisher_transform(rho_hat) + np.array([-1, 1]) * stats.norm.ppf(1 - alpha / 2) / np.sqrt(n - 3))

import numpy as np

from graphviz import Digraph

import logging
logger = logging.getLogger(__name__)


class Node:
    pass


class Branch(Node):
    def __init__(self, covariate, threshold, left, right, counts=None):
        self.covariate = covariate
        self.threshold = threshold
        self.left = left
        self.right = right
        self.counts = counts

    def predict(self, X):
        next_node = self.left if X[self.covariate] < self.threshold else self.right
        return next_node.predict(X)


class Leaf(Node):
    def __init__(self, value, counts=None):
        self.value = value
        self.counts = counts

    def predict(self, X):
        return self.value


def visualise_tree(tree):
    dot = Digraph()

    def add_nodes_edges(tree, parent, left):
        if isinstance(tree, Leaf):
            dot.node(
                name=str(tree),
                label=f'Class: {tree.value}\nCounts: {tree.counts}',
            )
        else:
            dot.node(
                name=str(tree),
                label=f'Covariate: {tree.covariate}\nCounts: {tree.counts}',
                shape='box',
            )
            add_nodes_edges(tree.left, tree, True)
            add_nodes_edges(tree.right, tree, False)

        if parent is not None:
            label = f'< {parent.threshold}' if left else f'>= {parent.threshold}'
            dot.edge(str(parent), str(tree), label=label)
    
    add_nodes_edges(tree, None, False)
    
    return dot


def impurity(y):
    return 1 - ((y.value_counts() / y.shape[0]) ** 2).sum()


def impurity_split(y, X, covariate, threshold):
    selector1 = X[covariate] < threshold
    selector2 = X[covariate] >= threshold
    n1 = np.sum(selector1)
    n2 = np.sum(selector2)
    return impurity(y[selector1]) * n1 + impurity(y[selector2]) * n2, n1, n2


def find_best_split(y, X, covariate, min_leaf_size=None):
    unique_covariates = np.unique(X[covariate])
    thresholds = (unique_covariates[:-1] + unique_covariates[1:]) / 2
    impurities = np.stack([impurity_split(y, X, covariate, t) for t in thresholds])
    if min_leaf_size is not None:
        # remove from consideration those entries that would result in leaves with small numbers of samples
        leaf_size_filter = np.min(impurities[:, 1:], axis=1) >= min_leaf_size
        impurities = impurities[leaf_size_filter]
        thresholds = thresholds[leaf_size_filter]
    if impurities.shape[0] > 0:
        min_impurity_idx = np.argmin(impurities[:, 0])
        return thresholds[min_impurity_idx], impurities[min_impurity_idx, 0]
    else:
        return 0.0, np.inf


def find_best_covariate_split(y, X, min_leaf_size=None):
    valid_covariates = np.array([c for c in X.columns if len(np.unique(X[c])) > 1])
    res = np.stack([find_best_split(y, X, c, min_leaf_size) for c in valid_covariates])
    # find minimum impurity
    covariate_idx = np.argmin(res[:, 1])
    if np.isinf(res[covariate_idx, 1]):
        # if all impurities are +inf, no split is possible
        return None, np.nan
    else:
        return valid_covariates[covariate_idx], res[covariate_idx, 0]


def fit_tree(y, X, min_leaf_size):
    assert X.shape[0] > 0

    # if all samples are in the same class, stop splitting
    y_distinct, counts = np.unique(y, return_counts=True)
    if len(y_distinct) == 1:
        return Leaf(y_distinct[0], counts)

    # attempt splitting
    covariate, threshold = find_best_covariate_split(y, X, min_leaf_size)
    if covariate is None:
        # no split would results in leaves having the required number of samples
        return Leaf(y_distinct[np.argmax(counts)], counts)

    split_left = X[covariate] < threshold
    split_right = X[covariate] >= threshold

    return Branch(
        covariate,
        threshold,
        fit_tree(y[split_left], X[split_left], min_leaf_size=min_leaf_size),
        fit_tree(y[split_right], X[split_right], min_leaf_size=min_leaf_size),
        counts,
    )


class DecisionTreeModel:
    def __init__(self, min_leaf_size):
        self.min_leaf_size = min_leaf_size

    def fit(self, X, y):
        self.tree = fit_tree(y, X, self.min_leaf_size)
        return self

    def predict(self, X):
        return X.apply(self.tree.predict, axis=1)

    def get_params(self, deep=True):
        return {
            'min_leaf_size': self.min_leaf_size,
        }

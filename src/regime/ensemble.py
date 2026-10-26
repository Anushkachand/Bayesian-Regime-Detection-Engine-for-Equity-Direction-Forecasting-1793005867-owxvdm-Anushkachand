"""Ensembling: out-of-sample log-score -> pseudo-BMA weights, plus scoring helpers."""
import numpy as np
from scipy.optimize import minimize
from . import NAMES


def log_score(probs, y, eps=1e-9):
    return float(np.mean(np.log(np.clip(probs[np.arange(len(y)), y], eps, 1))))


def brier(probs, y):
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(((probs - onehot) ** 2).sum(1)))


def pseudo_bma_weights(member_probs, y):
    """weights_m  proportional to exp(N * mean held-out log score)  (pseudo-BMA, Yao et al. 2018 sec. 2)."""
    n = len(y)
    scores = np.array([log_score(p, y) for p in member_probs]) * n
    scores -= scores.max()
    w = np.exp(scores)
    return w / w.sum()


def combine(member_probs, w):
    p = sum(wi * pi for wi, pi in zip(w, member_probs))
    p = np.clip(p, 1e-9, None)
    return p / p.sum(1, keepdims=True)


def stacking_weights(member_probs, y):
    """Constrained stacking: simplex weights maximising held-out mean log score (Yao et al. 2018).
    Softmax parameterisation keeps weights >= 0 and summing to 1."""
    P = np.stack([p[np.arange(len(y)), y] for p in member_probs])      # (M, n) prob of the true label
    M = P.shape[0]
    def neg(theta):
        w = np.exp(theta - theta.max()); w /= w.sum()
        return -np.mean(np.log(np.clip(w @ P, 1e-9, None)))
    res = minimize(neg, np.zeros(M), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-8, maxiter=4000))
    w = np.exp(res.x - res.x.max())
    return w / w.sum()

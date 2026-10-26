"""Calibration tools: ECE, split-conformal APS sets, and Adaptive Conformal Inference (ACI)."""
import numpy as np


def ece(probs, y, bins=10):
    """Expected calibration error on the top-label confidence."""
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def aps_scores_all(probs):
    """s(x, y) = total prob mass of labels at least as likely as y (non-randomised APS). Shape (n,K)."""
    order = np.argsort(-probs, axis=1)
    sp = np.take_along_axis(probs, order, 1)
    cum = np.cumsum(sp, 1)
    out = np.empty_like(probs)
    np.put_along_axis(out, order, cum, 1)
    return out


def calibrate_aps(probs_cal, y_cal, alpha=0.1):
    """Split-conformal threshold: finite-sample (1-alpha) quantile of the true-label APS scores."""
    s = aps_scores_all(probs_cal)[np.arange(len(y_cal)), y_cal]
    n = len(s)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(s, q, method="higher"))


def prediction_sets(probs, qhat):
    S = aps_scores_all(probs) <= qhat
    S[np.arange(len(probs)), probs.argmax(1)] = True      # never return an empty set
    return S


def coverage_and_size(S, y):
    return float(S[np.arange(len(y)), y].mean()), float(S.sum(1).mean())


def aci_sets(probs_cal, y_cal, probs_test, y_test, alpha=0.1, gamma=0.01):
    """Adaptive Conformal Inference (Gibbs & Candes 2021): alpha_t updated online from realised misses,
    which restores long-run coverage under distribution shift."""
    cal_scores = aps_scores_all(probs_cal)[np.arange(len(y_cal)), y_cal]
    test_scores = aps_scores_all(probs_test)
    a_t, alphas, sets = alpha, [], []
    for i in range(len(probs_test)):
        q = np.quantile(cal_scores, np.clip(1 - a_t, 0.0, 1.0), method="higher")
        s = test_scores[i] <= q
        s[probs_test[i].argmax()] = True
        sets.append(s)
        err = 0.0 if s[y_test[i]] else 1.0
        a_t = a_t + gamma * (alpha - err)
        alphas.append(a_t)
    return np.array(sets), np.array(alphas)

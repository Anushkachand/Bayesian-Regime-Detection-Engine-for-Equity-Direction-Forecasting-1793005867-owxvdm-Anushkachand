"""Bayesian Online Changepoint Detection (Adams & MacKay 2007), Normal-Inverse-Gamma model."""
import numpy as np
from scipy.stats import t as student_t


def bocpd(x, hazard=1 / 200.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1e-4, max_run=600):
    """Return P(run length <= 10 | data up to t) for each t and the MAP run length."""
    x = np.asarray(x, float)
    T = len(x)
    R = np.array([1.0])
    mu, kappa, alpha, beta = (np.array([v], float) for v in (mu0, kappa0, alpha0, beta0))
    p_recent = np.zeros(T)
    map_rl = np.zeros(T, dtype=int)
    for t in range(T):
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        pred = student_t.pdf(x[t], df=2 * alpha, loc=mu, scale=scale)       # predictive per run length
        growth = R * pred * (1 - hazard)
        cp = (R * pred * hazard).sum()
        R = np.concatenate([[cp], growth])
        R /= R.sum()
        # update sufficient statistics
        mu_new = (kappa * mu + x[t]) / (kappa + 1)
        beta_new = beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa + 1])
        alpha = np.concatenate([[alpha0], alpha + 0.5])
        beta = np.concatenate([[beta0], beta_new])
        if len(R) > max_run:   # truncate tail for speed
            R, mu, kappa, alpha, beta = (a[:max_run] for a in (R, mu, kappa, alpha, beta))
            R /= R.sum()
        p_recent[t] = R[:11].sum()
        map_rl[t] = int(R.argmax())
    return p_recent, map_rl


def detect(p_recent, thresh=0.5, min_gap=20):
    """Turn the changepoint-probability signal into discrete detection dates (indices)."""
    out, last = [], -10 ** 9
    for t, p in enumerate(p_recent):
        if p > thresh and t - last >= min_gap:
            out.append(t); last = t
    return out


def score_detections(det, truth_idx, tol=15):
    """Precision/recall of detections vs true changepoints within +/- tol days (detections can lag)."""
    det, truth_idx = np.array(det), np.array(truth_idx)
    hits = sum(any(0 <= d - c <= tol for d in det) for c in truth_idx)
    good = sum(any(0 <= d - c <= tol for c in truth_idx) for d in det)
    return dict(n_detections=int(len(det)), n_true=int(len(truth_idx)),
                recall=hits / max(len(truth_idx), 1), precision=good / max(len(det), 1))

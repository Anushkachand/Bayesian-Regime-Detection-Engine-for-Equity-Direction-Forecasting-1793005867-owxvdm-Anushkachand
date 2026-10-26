"""Gaussian HMM (diagonal covariance) fitted by Baum-Welch, written in NumPy.

Extras over a vanilla HMM:
  * Dirichlet prior on each transition row (MAP-EM) with a 'sticky' diagonal boost
  * transition_posterior(): Dirichlet concentrations -> credible intervals on persistence
  * filter(): causal forward probabilities (what you would know *today*)
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist
from sklearn.cluster import KMeans
from . import NAMES


class GaussianHMM:
    def __init__(self, K=5, n_iter=100, tol=1e-4, seed=0, prior_conc=1.0, sticky=20.0):
        self.K, self.n_iter, self.tol, self.seed = K, n_iter, tol, seed
        self.prior_conc, self.sticky = prior_conc, sticky

    # ---- emissions ------------------------------------------------------
    def _logB(self, X):
        d = X[:, None, :] - self.means[None]                       # (T,K,D)
        return -0.5 * (np.log(2 * np.pi * self.vars)[None] + d ** 2 / self.vars[None]).sum(-1)

    # ---- inference ------------------------------------------------------
    def _forward(self, logB):
        T, K = logB.shape
        logA = np.log(self.A)
        la = np.empty((T, K))
        la[0] = np.log(np.clip(self.pi, 1e-12, None)) + logB[0]
        for t in range(1, T):
            la[t] = logsumexp(la[t - 1][:, None] + logA, axis=0) + logB[t]
        return la

    def _backward(self, logB):
        T, K = logB.shape
        logA = np.log(self.A)
        lb = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(logA + logB[t + 1][None] + lb[t + 1][None], axis=1)
        return lb

    def fit(self, X):
        X = np.asarray(X, float)
        T, D = X.shape
        km = KMeans(self.K, n_init=10, random_state=self.seed).fit(X)
        self.means = km.cluster_centers_.copy()
        self.vars = np.stack([X[km.labels_ == k].var(0) + 1e-4 for k in range(self.K)])
        self.A = np.full((self.K, self.K), 0.02 / (self.K - 1))
        np.fill_diagonal(self.A, 0.98)
        self.pi = np.full(self.K, 1.0 / self.K)
        prior = np.full((self.K, self.K), self.prior_conc) + self.sticky * np.eye(self.K)
        prev = -np.inf
        for _ in range(self.n_iter):
            logB = self._logB(X)
            la, lb = self._forward(logB), self._backward(logB)
            ll = logsumexp(la[-1])
            lg = la + lb - ll
            gamma = np.exp(lg)
            logA = np.log(self.A)
            xi = np.zeros((self.K, self.K))
            for t in range(T - 1):
                xi += np.exp(la[t][:, None] + logA + logB[t + 1][None] + lb[t + 1][None] - ll)
            # M-step (MAP for transitions)
            self.xi_counts = xi
            self.A = (xi + prior - 1.0 + 1e-12)
            self.A /= self.A.sum(1, keepdims=True)
            w = gamma.sum(0)[:, None]
            self.means = gamma.T @ X / w
            self.vars = (gamma.T @ (X ** 2)) / w - self.means ** 2 + 1e-4
            self.vars = np.maximum(self.vars, 1e-4)
            self.pi = gamma[0] / gamma[0].sum()
            if abs(ll - prev) < self.tol * abs(ll):
                break
            prev = ll
        self.loglik_ = ll
        self.prior_ = prior
        return self

    def filter(self, X):
        """Causal filtered state probabilities P(s_t | x_1..t)."""
        la = self._forward(self._logB(np.asarray(X, float)))
        return np.exp(la - logsumexp(la, axis=1, keepdims=True))

    def smooth(self, X):
        X = np.asarray(X, float)
        logB = self._logB(X)
        la, lb = self._forward(logB), self._backward(logB)
        lg = la + lb
        return np.exp(lg - logsumexp(lg, axis=1, keepdims=True))

    def predict_h(self, filt, h):
        """Propagate filtered probs h steps ahead through the transition matrix."""
        return filt @ np.linalg.matrix_power(self.A, h)

    # ---- Bayesian pieces ------------------------------------------------
    def transition_posterior(self):
        """Dirichlet concentration per row = prior + expected transition counts."""
        return self.prior_ + self.xi_counts

    def persistence_ci(self, level=0.95):
        """Credible interval for each regime's self-transition probability (Beta marginal)."""
        conc = self.transition_posterior()
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        out = []
        for k in range(self.K):
            a, b = conc[k, k], conc[k].sum() - conc[k, k]
            out.append((a / (a + b), beta_dist.ppf(lo, a, b), beta_dist.ppf(hi, a, b)))
        return np.array(out)

    # ---- labelling ------------------------------------------------------
    def name_map(self, ret_col=0):
        """Map HMM state index -> regime name from its mean/vol signature.
        Sequential rule: Risk-Off = lowest mean; Risk-On = best mean/vol;
        Post-Shock = highest vol among the rest; Late-Cycle = lower vol; Transitional = last."""
        mu, sd = self.means[:, ret_col], np.sqrt(self.vars[:, ret_col])
        left, m = list(range(self.K)), {}
        k = min(left, key=lambda i: mu[i]);            m[k] = "Risk-Off";     left.remove(k)
        k = max(left, key=lambda i: mu[i] / sd[i]);    m[k] = "Risk-On";      left.remove(k)
        k = max(left, key=lambda i: sd[i]);            m[k] = "Post-Shock";   left.remove(k)
        k = min(left, key=lambda i: sd[i]);            m[k] = "Late-Cycle";   left.remove(k)
        m[left[0]] = "Transitional"
        return m

    def to_named(self, probs, ret_col=0):
        """Reorder probability columns into the canonical NAMES order."""
        m = self.name_map(ret_col)
        out = np.zeros_like(probs)
        for idx, name in m.items():
            out[:, NAMES.index(name)] = probs[:, idx]
        return out

    def named_transition(self, ret_col=0):
        m = self.name_map(ret_col)
        order = [next(i for i, n in m.items() if n == name) for name in NAMES]
        return self.A[np.ix_(order, order)]

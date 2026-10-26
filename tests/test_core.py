import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from regime.hmm import GaussianHMM
from regime.conformal import calibrate_aps, prediction_sets, coverage_and_size, ece
from regime.bocpd import bocpd, detect
from regime.ensemble import stacking_weights


def _two_regime_data(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < 0.97 else 1 - s[t - 1]
    x = np.where(s == 0, rng.normal(0.5, 0.5, n), rng.normal(-0.5, 2.0, n))[:, None]
    return x, s


def test_hmm_recovers_two_regimes():
    x, s = _two_regime_data()
    h = GaussianHMM(K=2, seed=0).fit(x)
    pred = h.smooth(x).argmax(1)
    acc = max((pred == s).mean(), (pred != s).mean())     # label switching
    assert acc > 0.85
    assert np.allclose(h.A.sum(1), 1)


def test_filter_rows_sum_to_one():
    x, _ = _two_regime_data()
    h = GaussianHMM(K=2, seed=0).fit(x)
    assert np.allclose(h.filter(x).sum(1), 1)


def test_conformal_coverage_is_valid():
    rng = np.random.default_rng(1)
    def draw(n):
        p = rng.dirichlet(np.ones(4) * 0.6, size=n)
        y = np.array([rng.choice(4, p=pi) for pi in p])   # labels truly drawn from p -> p is calibrated
        return p, y
    pc, yc = draw(2000); pt, yt = draw(2000)
    q = calibrate_aps(pc, yc, alpha=0.1)
    cov, _ = coverage_and_size(prediction_sets(pt, q), yt)
    assert cov >= 0.88


def test_ece_zero_for_perfect_confidence():
    p = np.eye(3)[[0, 1, 2, 0]]
    assert ece(p, np.array([0, 1, 2, 0])) < 1e-9


def test_bocpd_fires_on_variance_break():
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(0, 0.01, 400), rng.normal(0, 0.04, 400)]
    det = detect(bocpd(x)[0])
    assert any(400 <= d <= 430 for d in det)


def test_stacking_weights_on_simplex():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 300)
    good = np.eye(3)[y] * 0.6 + 0.4 / 3
    bad = np.full((300, 3), 1 / 3)
    w = stacking_weights([good, bad], y)
    assert abs(w.sum() - 1) < 1e-9 and w[0] > w[1]

"""Data loading: synthetic regime generator (known truth) + Yahoo Finance loader."""
import numpy as np
import pandas as pd
from . import NAMES

# daily (mean, vol) of Nifty-like returns and typical India-VIX level, per regime
_PARAMS = {
    "Risk-On":      (0.0008, 0.007, 13.0),
    "Risk-Off":     (-0.0015, 0.020, 30.0),
    "Transitional": (0.0000, 0.012, 18.0),
    "Late-Cycle":   (0.0003, 0.009, 15.0),
    "Post-Shock":   (0.0010, 0.014, 22.0),
}

def _transition_matrix(persist=0.975):
    K = len(NAMES)
    # off-diagonals favour plausible moves (e.g. Risk-Off -> Post-Shock)
    off = np.array([
        [0, .15, .35, .45, .05],   # Risk-On
        [.05, 0, .15, .00, .80],   # Risk-Off
        [.35, .30, 0, .15, .20],   # Transitional
        [.10, .30, .50, 0, .10],   # Late-Cycle
        [.45, .05, .40, .10, 0],   # Post-Shock
    ], dtype=float)
    off = off / off.sum(1, keepdims=True)
    A = off * (1 - persist)
    A[np.diag_indices(K)] = persist
    return A

def make_synthetic(n=3500, seed=7):
    """Return (DataFrame[close, vix], true_state Series of regime names)."""
    rng = np.random.default_rng(seed)
    A = _transition_matrix()
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = rng.choice(len(NAMES), p=A[s[t - 1]])
    mu = np.array([_PARAMS[k][0] for k in NAMES])[s]
    sd = np.array([_PARAMS[k][1] for k in NAMES])[s]
    ret = mu + sd * rng.standard_normal(n)
    vix_lvl = np.array([_PARAMS[k][2] for k in NAMES])[s]
    vix = np.empty(n); vix[0] = vix_lvl[0]
    for t in range(1, n):   # mean-reverting toward the regime's VIX level
        vix[t] = vix[t - 1] + 0.15 * (vix_lvl[t] - vix[t - 1]) + 0.6 * rng.standard_normal()
    vix = np.clip(vix, 9, None)
    idx = pd.bdate_range("2011-01-03", periods=n)
    close = 5000 * np.exp(np.cumsum(ret))
    df = pd.DataFrame({"close": close, "vix": vix}, index=idx)
    truth = pd.Series([NAMES[i] for i in s], index=idx, name="true_regime")
    return df, truth

def load_yahoo(start="2010-01-01"):
    """Real Nifty 50 + India VIX from Yahoo Finance (needs internet + yfinance)."""
    import yfinance as yf
    nifty = yf.download("^NSEI", start=start, auto_adjust=True, progress=False)["Close"]
    vix = yf.download("^INDIAVIX", start=start, auto_adjust=True, progress=False)["Close"]
    nifty = nifty.squeeze(); vix = vix.squeeze()
    df = pd.concat([nifty.rename("close"), vix.rename("vix")], axis=1).dropna()
    return df

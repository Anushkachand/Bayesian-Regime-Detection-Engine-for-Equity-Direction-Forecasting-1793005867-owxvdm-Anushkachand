"""Feature engineering: returns, trend, volatility, drawdown, VIX features."""
import numpy as np
import pandas as pd

FEATURES = ["ret", "ret5", "vol20", "vol60", "trend", "dd", "vix", "dvix5"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["close"], df["vix"]
    f = pd.DataFrame(index=df.index)
    f["ret"] = np.log(c).diff()
    f["ret5"] = f["ret"].rolling(5).sum()
    f["vol20"] = f["ret"].rolling(20).std()
    f["vol60"] = f["ret"].rolling(60).std()
    f["trend"] = c.rolling(50).mean() / c.rolling(200).mean() - 1   # 50-DMA vs 200-DMA
    f["dd"] = c / c.rolling(252, min_periods=60).max() - 1          # drawdown
    f["vix"] = v
    f["dvix5"] = v.diff(5)
    return f.dropna()

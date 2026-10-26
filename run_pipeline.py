"""End-to-end pipeline:  data -> features -> HMM + ML members -> ensemble -> conformal -> BOCPD -> report.

Usage:
    python run_pipeline.py              # synthetic data with known truth (works offline)
    python run_pipeline.py --real       # real Nifty 50 + India VIX via Yahoo Finance
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from regime import NAMES
from regime.data import make_synthetic, load_yahoo
from regime.features import build_features, FEATURES
from regime.hmm import GaussianHMM
from regime.bocpd import bocpd, detect, score_detections
from regime.conformal import (ece, calibrate_aps, prediction_sets, coverage_and_size, aci_sets)
from regime.ensemble import log_score, brier, pseudo_bma_weights, stacking_weights, combine

H = 5          # forecast horizon in trading days
ALPHA = 0.10   # target miscoverage -> 90% prediction sets
OUT = "outputs"


def full_proba(model, X, K=len(NAMES)):
    """predict_proba padded to all K classes (a class may be absent from the training fold)."""
    p = np.zeros((len(X), K))
    p[:, model.classes_] = model.predict_proba(X)
    return p


def main(real=False):
    os.makedirs(OUT, exist_ok=True)
    # ---------- 1. data & features ----------
    if real:
        df, truth = load_yahoo(), None
    else:
        df, truth = make_synthetic()
    F = build_features(df)
    n = len(F)
    print(f"[data] {n} usable days  {F.index[0].date()} -> {F.index[-1].date()}")

    # ---------- 2. chronological splits (no shuffling, no leakage) ----------
    i1, i2, i3 = int(.55 * n), int(.70 * n), int(.85 * n)
    tr, va, ca, te = slice(0, i1), slice(i1, i2), slice(i2, i3), slice(i3, n - H)

    # ---------- 3. HMM (emissions: scaled return + standardised VIX; fit on TRAIN only) ----------
    vm, vs = F["vix"].iloc[tr].mean(), F["vix"].iloc[tr].std()
    Xh = np.c_[F["ret"].values * 100, (F["vix"].values - vm) / vs]
    hmm = GaussianHMM(K=5, seed=1).fit(Xh[tr])
    filt = hmm.to_named(hmm.filter(Xh))                       # causal P(regime_t | data up to t)
    names_at = np.array(NAMES)[filt.argmax(1)]
    print("[hmm] state signatures:", hmm.name_map())

    # ---------- 4. targets: regime at t+H ----------
    if truth is not None:
        lab = truth.loc[F.index].map({k: i for i, k in enumerate(NAMES)}).values
        label_kind = "TRUE synthetic regime"
    else:   # real data has no ground truth -> HMM smoothed labels (reference only, circular!)
        lab = hmm.to_named(hmm.smooth(Xh)).argmax(1)
        label_kind = "HMM-smoothed reference label (not ground truth)"
    y = np.full(n, -1); y[:n - H] = lab[H:]                  # y_t = regime at t+H

    # ---------- 5. members ----------
    sc = StandardScaler().fit(F[FEATURES].iloc[tr])
    Xs = sc.transform(F[FEATURES])
    members = {}
    members["HMM (filtered, A^h)"] = hmm.to_named(hmm.predict_h(hmm.filter(Xh), H))
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(Xs[tr][: -H], y[tr][: -H])
    members["Logistic"] = full_proba(lr, Xs)
    mlps = [MLPClassifier((32, 16), alpha=1e-2, max_iter=400, early_stopping=True, random_state=s)
            .fit(Xs[tr][: -H], y[tr][: -H]) for s in range(5)]
    members["Deep ensemble (5 MLPs)"] = np.mean([full_proba(m, Xs) for m in mlps], axis=0)
    prior = np.bincount(y[tr][: -H], minlength=5) / len(y[tr][: -H])
    members["Class-frequency baseline"] = np.tile(prior, (n, 1))

    # ---------- 6. ensemble weights on VALIDATION, conformal calibration on CAL, report on TEST ----------
    real_members = {k: v for k, v in members.items() if "baseline" not in k}
    vp = [p[va] for p in real_members.values()]
    w_bma = pseudo_bma_weights(vp, y[va])
    w = stacking_weights(vp, y[va])                      # used for the final ensemble
    ens = combine(list(real_members.values()), w)
    members["ENSEMBLE (stacking)"] = ens
    print("[ensemble] pseudo-BMA weights:", dict(zip(real_members, np.round(w_bma, 3).tolist())))
    print("[ensemble] stacking weights  :", dict(zip(real_members, np.round(w, 3).tolist())))

    rows = []
    for name, p in members.items():
        rows.append(dict(model=name,
                         log_score=log_score(p[te], y[te]), brier=brier(p[te], y[te]),
                         ece=ece(p[te], y[te]), accuracy=float((p[te].argmax(1) == y[te]).mean())))
    table = pd.DataFrame(rows).set_index("model").round(4)
    print("\n[test-set metrics] horizon = %d days, labels = %s" % (H, label_kind))
    print(table.to_string())

    # ---------- 7. conformal prediction sets on the ensemble ----------
    qhat = calibrate_aps(ens[ca], y[ca], ALPHA)
    S = prediction_sets(ens[te], qhat)
    cov, size = coverage_and_size(S, y[te])
    S_aci, alphas = aci_sets(ens[ca], y[ca], ens[te], y[te], ALPHA, gamma=0.01)
    cov_aci, size_aci = coverage_and_size(S_aci, y[te])
    print(f"\n[conformal] target coverage {1-ALPHA:.0%}")
    print(f"  split-APS : coverage={cov:.3f}  avg set size={size:.2f}")
    print(f"  ACI       : coverage={cov_aci:.3f}  avg set size={size_aci:.2f}")

    # ---------- 8. HMM Bayesian outputs ----------
    ci = hmm.persistence_ci()
    m = hmm.name_map()
    order = [next(i for i, nm in m.items() if nm == name) for name in NAMES]
    pers = pd.DataFrame(ci[order], index=NAMES, columns=["P(stay) mean", "CI 2.5%", "CI 97.5%"]).round(4)
    pers["expected duration (days)"] = (1 / (1 - pers["P(stay) mean"])).round(1)
    print("\n[hmm] posterior persistence (Dirichlet->Beta marginal):")
    print(pers.to_string())

    # ---------- 9. BOCPD on returns ----------
    p_recent, _ = bocpd(F["ret"].values)
    det = detect(p_recent)
    bocpd_stats = None
    if truth is not None:
        tv = truth.loc[F.index].values
        true_cp = np.where(tv[1:] != tv[:-1])[0] + 1
        # only count 'major' breaks: moves between regimes with different vol/mean (exclude Late-Cycle<->Risk-On)
        bocpd_stats = score_detections(det, true_cp, tol=15)
        print("\n[bocpd] vs ALL true regime switches (+15d lag tolerance):", {k: round(v, 3) if isinstance(v, float) else v for k, v in bocpd_stats.items()})

    # ---------- 10. figures ----------
    idx = F.index[te]
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(idx, df["close"].reindex(idx), color="k", lw=1); ax[0].set_ylabel("Index level")
    ax[0].set_title("Ensemble regime probabilities (test period, horizon=%dd)" % H)
    ax[1].stackplot(idx, ens[te].T, labels=NAMES, alpha=.85); ax[1].legend(loc="upper left", ncol=5, fontsize=8)
    ax[1].set_ylabel("P(regime)"); ax[1].set_ylim(0, 1)
    ax[2].plot(idx, S.sum(1), color="C3", lw=.8); ax[2].set_ylabel("conformal set size")
    plt.tight_layout(); plt.savefig(f"{OUT}/regime_probabilities.png", dpi=110); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    conf, hit = members["ENSEMBLE (stacking)"][te].max(1), (ens[te].argmax(1) == y[te])
    edges = np.linspace(0, 1, 11); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (conf > lo) & (conf <= hi)
        if mk.sum() >= 5: xs.append(conf[mk].mean()); ys.append(hit[mk].mean())
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].plot(xs, ys, "o-"); ax[0].set_xlabel("confidence"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("Reliability diagram (ensemble)")
    win = 63
    roll = pd.Series(S[np.arange(len(S)), y[te]].astype(float)).rolling(win).mean()
    ax[1].plot(roll.values, label="split-APS"); 
    roll2 = pd.Series(S_aci[np.arange(len(S_aci)), y[te]].astype(float)).rolling(win).mean()
    ax[1].plot(roll2.values, label="ACI"); ax[1].axhline(1 - ALPHA, color="k", ls="--")
    ax[1].set_title("Rolling 63-day conformal coverage"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/calibration.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(F.index, p_recent, lw=.7); ax.axhline(.5, color="r", ls="--")
    for d in det: ax.axvline(F.index[d], color="g", alpha=.3)
    ax.set_title("BOCPD: P(run length <= 10) with detections (green)")
    plt.tight_layout(); plt.savefig(f"{OUT}/bocpd.png", dpi=110); plt.close()

    # ---------- 11. save artefacts ----------
    table.to_csv(f"{OUT}/model_comparison.csv")
    pers.to_csv(f"{OUT}/hmm_persistence.csv")
    pd.DataFrame(hmm.named_transition(), index=NAMES, columns=NAMES).round(4).to_csv(f"{OUT}/hmm_transition_matrix.csv")
    latest = pd.Series(ens[n - H - 1], index=NAMES)
    latest_set = [NAMES[i] for i in np.where(prediction_sets(ens[[n - H - 1]], qhat)[0])[0]]
    summary = dict(mode="real" if real else "synthetic", horizon_days=H, labels=label_kind,
                   ensemble_weights_stacking=dict(zip(real_members, map(float, w))),
                   ensemble_weights_pseudo_bma=dict(zip(real_members, map(float, w_bma))),
                   conformal=dict(target=1 - ALPHA, split_aps=dict(coverage=cov, avg_set_size=size),
                                  aci=dict(coverage=cov_aci, avg_set_size=size_aci)),
                   bocpd=bocpd_stats,
                   latest_call=dict(date=str(F.index[n - H - 1].date()), probs=latest.round(4).to_dict(),
                                    conformal_set_90=latest_set))
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    print("\n[latest call]", json.dumps(summary["latest_call"]))
    print(f"[done] artefacts written to ./{OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--real", action="store_true")
    main(ap.parse_args().real)

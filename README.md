# Bayesian Regime Detection Engine — Indian Equities

Classifies the market into five regimes (**Risk-On, Risk-Off, Transitional, Late-Cycle, Post-Shock**)
and outputs *calibrated probabilities with conformal uncertainty sets* instead of point forecasts.

## Pipeline
```
data -> features -> HMM (Baum-Welch, Dirichlet-sticky MAP) ─┐
                    Logistic regression                      ├─> stacking ensemble -> conformal (APS + ACI)
                    Deep ensemble (5 MLPs)                  ─┘
returns -> BOCPD (online changepoints)
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py            # synthetic data with known truth (offline)
python run_pipeline.py --real     # real Nifty 50 + India VIX (Yahoo Finance)
python -m pytest -q tests
```
Outputs land in `outputs/` (metrics CSVs, `summary.json`, three PNG figures).

## Results (synthetic data, 5-day-ahead regime, chronological test split)
| model | log-score | ECE | accuracy |
|---|---|---|---|
| HMM (filtered, A^h) | -1.667 | 0.265 | 0.540 |
| Logistic | -1.352 | 0.096 | 0.523 |
| Deep ensemble | -1.124 | 0.063 | 0.562 |
| **Stacking ensemble** | -1.135 | **0.044** | **0.572** |

Conformal 90% sets: split-APS coverage 0.927 (avg size 2.79); ACI coverage 0.900 (avg size 2.55).
BOCPD: precision 0.94, recall 0.19 against *all* true switches (many switches are between near-identical regimes).

**Honest reading:** the ensemble has the best calibration (ECE) and accuracy, but *not* the best log-score
(the deep ensemble is marginally better), so the "ensemble beats every member" criterion is **not** met here.
The HMM is overconfident (ECE 0.27) and receives ~0 stacking weight.

## Scope vs. the project brief
| Brief item | Status |
|---|---|
| Frequentist HMM, 5 states, labelling, transition matrix | Done (from scratch, NumPy) |
| Bayesian transition posterior (Dirichlet -> Beta credible intervals) | Done (analytic, not full MCMC) |
| Bayesian HMM via PyMC/NUTS, RS-VAR, R-hat/ESS diagnostics | **TODO** (needs pymc/numpyro) |
| MC-dropout / variational BNN | Partial: deep ensemble of MLPs only |
| Foundation models (Chronos + TimesFM/Moirai) | **TODO** |
| Ensembling: stacking + pseudo-BMA | Done (WAIC/LOO via ArviZ TODO) |
| Conformal: split-APS + ACI, ECE, reliability, rolling coverage | Done |
| BOCPD | Done; particle filter TODO |
| TDA/GNN features, breadth/FII/DII/macro features | **TODO** (only price/vol/VIX features) |
| Monte Carlo + allocation backtest (IR, TE), case studies, R codebase | **TODO** |
| Report (40+ pages), model card, 18-slide deck, demo video | **TODO** |

## Caveats
* Results above are on **synthetic** data with known labels. On real data there is no ground truth, so
  `--real` scores against HMM-smoothed labels, which is circular and only a sanity check.
* Educational/research code; not investment advice.

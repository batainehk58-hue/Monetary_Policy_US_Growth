import json
import warnings

import numpy as np
import openpyxl
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order

warnings.filterwarnings("ignore")

RAW = "data/"                       # <- directory holding the raw downloads
START, END = "2003-01-01", "2025-10-01"   # 2003:Q1 - 2025:Q4
K = 2                               # lagged differences (three lags in levels)
RANK = 1                            # cointegrating rank
HORIZON = 20                        # quarters of impulse responses
NBOOT = 1000                        # bootstrap replications (Table VII, Fig. 4)
OUT = {}


# ======================================================================
# 1. BUILD THE DATASET
# ======================================================================
def load(fname, col):
    d = pd.read_csv(RAW + fname, parse_dates=["observation_date"])
    d = d.rename(columns={"observation_date": "date"})
    return d.set_index("date")[[col]]


def build_dataset():
    gdp = load("GDPC1.csv", "GDPC1")
    inf = load("DPCCRV1Q225SBEA.csv", "DPCCRV1Q225SBEA")
    unr = load("UNRATE.csv", "UNRATE")
    m2 = load("M2SL.csv", "M2SL")
    wal = load("WALCL.csv", "WALCL")
    ffr = load("FEDFUNDS.csv", "FEDFUNDS")

    # ---- Wu-Xia shadow rate (monthly, last business day; ends 2022-02) ----
    ws = openpyxl.load_workbook(RAW + "WuXiaShadowRate.xlsx", data_only=True)["Data"]
    rows = [(r[0], r[1], r[2]) for r in ws.iter_rows(min_row=2, values_only=True)
            if r[0] is not None]
    wx = pd.DataFrame(rows, columns=["date", "ffr_eom", "shadow"]).set_index("date")
    wx.index = pd.to_datetime(wx.index)

    # ---- Splice: shadow rate where published, effective funds rate thereafter ----
    mm = ffr.join(wx[["shadow"]], how="left")
    mm["SHADOW"] = mm["shadow"].where(mm["shadow"].notna(), mm["FEDFUNDS"])

    used = mm.loc[START:"2025-12-01", "shadow"].notna().sum()
    n_months = len(mm.loc[START:"2025-12-01"])
    print(f"[splice] Wu-Xia months used: {used} of {n_months} "
          f"(published through {wx['shadow'].last_valid_index():%Y-%m})")

    # how close are the two series OUTSIDE the ZLB? (Section 3.1 of the paper)
    j = mm.join(wx[["shadow"]], rsuffix="_x")
    for lo, hi in [("2003", "2007"), ("2016", "2019")]:
        dd = (j.loc[lo:hi, "shadow"] - j.loc[lo:hi, "FEDFUNDS"]).abs()
        print(f"[splice] |shadow - EFFR| {lo}-{hi}: "
              f"mean {dd.mean():.3f}, max {dd.max():.3f}")

    # ---- Quarterly averages ----
    Q = lambda d: d.resample("QS").mean()
    df = pd.concat(
        [Q(gdp), Q(inf), Q(unr), Q(m2), Q(wal), Q(ffr), Q(mm[["SHADOW"]])], axis=1
    )
    df.columns = ["GDPC1", "INF", "UNRATE", "M2SL", "WALCL", "FEDFUNDS", "SHADOW"]
    df = df.loc[START:END].copy()

    df["LGDP"] = np.log(df["GDPC1"])
    df["LM2"] = np.log(df["M2SL"])
    df["LWALCL"] = np.log(df["WALCL"])
    df["FFR"] = df["FEDFUNDS"]
    df["SSR"] = df["SHADOW"]

    # ---- FOUR separate impulse dummies (NOT spanning dummies) ----
    idx = df.index
    for q in ["2008-10-01", "2009-01-01", "2020-04-01", "2020-07-01"]:
        df["D_" + q[:7]] = (idx == pd.Timestamp(q)).astype(float)

    # ---- spanning dummies, kept only to reproduce the rejected specification ----
    df["D_GFC"] = ((idx >= "2008-10-01") & (idx <= "2009-01-01")).astype(float)
    df["D_COVID"] = ((idx >= "2020-04-01") & (idx <= "2020-07-01")).astype(float)

    df.index.name = "date"
    df.index.freq = "QS-JAN"
    df.to_csv("dataset_quarterly_with_shadow_rate.csv")
    print(f"[data] sample {df.index.min():%Y-%m} -> {df.index.max():%Y-%m}, n = {len(df)}\n")
    return df


# ======================================================================
# 2. TABLE II - DESCRIPTIVE STATISTICS  (on the ESTIMATION SAMPLE)
# ======================================================================
def descriptives(df):
    print("### TABLE II: Descriptive statistics (estimation sample only) ###")
    spec = [("Real GDP (bn $)", df.GDPC1, 0),
            ("Core PCE inflation (%)", df.INF, 2),
            ("Unemployment (%)", df.UNRATE, 2),
            ("M2 (bn $)", df.M2SL, 0),
            ("Fed total assets (tn $)", df.WALCL / 1e6, 2),
            ("Shadow rate SSR (%)", df.SSR, 2),
            ("Federal funds rate (%)", df.FFR, 2)]
    tab = {}
    for name, s, dp in spec:
        tab[name] = [round(s.mean(), dp), round(s.std(), dp),
                     round(s.min(), dp), round(s.max(), dp)]
        print(f"  {name:24} mean={tab[name][0]:>10} sd={tab[name][1]:>8} "
              f"min={tab[name][2]:>9} max={tab[name][3]:>9}")
    print(f"  -> sd(SSR)/sd(FFR) = {df.SSR.std() / df.FFR.std():.2f}\n")
    return tab


# ======================================================================
# 3. TABLE III - THE REVERSE-CAUSALITY EVIDENCE
# ======================================================================
def reverse_causality(df):
    print("### TABLE III: reverse causality ###")
    d = df[["UNRATE", "LWALCL"]].diff().dropna()
    crisis = pd.to_datetime(["2008-10-01", "2009-01-01", "2020-04-01", "2020-07-01"])
    mask = d.index.isin(crisis)
    full = d.UNRATE.corr(d.LWALCL)
    excl = d[~mask].UNRATE.corr(d[~mask].LWALCL)
    print(d[mask].round(3).to_string())
    print(f"  corr(dUNRATE, dLWALCL) full sample     = {full:+.3f}")
    print(f"  corr(dUNRATE, dLWALCL) excl. 4 quarters= {excl:+.3f}   "
          f"(n = {len(d)} first differences)\n")
    return {"corr_full": round(full, 3), "corr_excl": round(excl, 3),
            "n_diffs": int(len(d))}


# ======================================================================
# 4. TABLE IV - ADF UNIT-ROOT TESTS
# ======================================================================
def adf_table(df):
    print("### TABLE IV: ADF ###")
    res = {}
    for v in ["LGDP", "INF", "UNRATE", "LM2", "LWALCL", "SSR", "FFR"]:
        a0 = adfuller(df[v].dropna(), autolag="AIC")
        a1 = adfuller(df[v].diff().dropna(), autolag="AIC")
        res[v] = (round(a0[0], 2), round(a0[1], 3), round(a1[0], 2), round(a1[1], 3))
        print(f"  {v:7} level t={a0[0]:6.2f} p={a0[1]:.3f} | "
              f"diff t={a1[0]:7.2f} p={a1[1]:.3f}")
    print()
    return res


# ======================================================================
# 5. THE VECM
# ======================================================================
def estimate(df, policy, exog, tag, order=None, verbose=True):
    """Estimate one VECM. Returns a dict of every statistic reported."""
    order = order or ["LGDP", "INF", "UNRATE", "LM2", "LWALCL", policy]
    sys = df[order]
    ix = {n: i for i, n in enumerate(order)}

    # --- Johansen trace (note: computed WITHOUT the dummies; see paper 4.1) ---
    jo = coint_johansen(sys, det_order=0, k_ar_diff=K)
    trace = [(round(float(jo.lr1[i]), 2), round(float(jo.cvt[i, 1]), 2))
             for i in range(len(order))]

    res = VECM(sys, k_ar_diff=K, coint_rank=RANK,
               deterministic="ci", exog=exog).fit()

    # --- long-run vector, normalised on LGDP ---
    beta = res.beta[:len(order), 0] / res.beta[0, 0]
    beta_d = {n: round(float(b), 4) for n, b in zip(order, beta)}

    # --- adjustment coefficients ---
    alpha = {n: (round(float(res.alpha[i, 0]), 4),
                 round(float(res.tvalues_alpha[i, 0]), 2))
             for n, i in ix.items()}

    # --- Granger causality ---
    gc = {}
    for caused in ["LGDP", "UNRATE"]:
        for causing in [policy, "LWALCL", "LM2"]:
            t = res.test_granger_causality(caused=caused, causing=causing)
            gc[f"{causing}->{caused}"] = (round(float(t.test_statistic), 2),
                                          round(float(t.pvalue), 3))

    # --- residual diagnostics ---
    # (i) equation-level Ljung-Box, with the estimated parameters netted out of the df
    n_par = len(order) * K + (exog.shape[1] if exog is not None else 0) + 1
    r = pd.DataFrame(res.resid)
    lb = min(acorr_ljungbox(r[c], lags=[8], model_df=min(n_par, 7),
                            return_df=True)["lb_pvalue"].iloc[0]
             for c in r.columns)
    lb4 = min(acorr_ljungbox(r[c], lags=[4], return_df=True)["lb_pvalue"].iloc[0]
              for c in r.columns)            # the figure quoted in the paper
    # (ii) multivariate portmanteau test for the SYSTEM (this is the honest one:
    #      it rejects, and Section 4.4 says so)
    wh = {}
    for nl in (4, 8, 12):
        try:
            w = res.test_whiteness(nlags=nl, adjusted=True)
            wh[nl] = (round(float(w.test_statistic), 1), round(float(w.pvalue), 3))
        except Exception:
            wh[nl] = (float("nan"), float("nan"))
    try:
        nm = res.test_normality()
        norm_p = float(nm.pvalue)
    except Exception:
        norm_p = float("nan")

    # --- stability of the level VAR representation ---
    A = res.var_rep
    p, Kv = A.shape[0], A.shape[1]
    C = np.zeros((Kv * p, Kv * p))
    C[:Kv, :] = np.hstack([A[i] for i in range(p)])
    if p > 1:
        C[Kv:, :-Kv] = np.eye(Kv * (p - 1))
    ev = np.sort(np.abs(np.linalg.eigvals(C)))[::-1]
    max_nonunit = round(float(ev[Kv - RANK]), 3)   # largest root below the unit roots

    # ------------------------------------------------------------------
    # IMPULSE RESPONSES
    # NOTE: VECMResults.irf() returns responses of the LEVELS of the variables
    # (it is built on res.var_rep, the level VAR representation).  orth_irfs[h]
    # is therefore the level deviation at horizon h -- for UNRATE, directly in
    # percentage points.  Do NOT cumulate these: cumulating a level response
    # yields "percentage-point-quarters", not percentage points.
    # ------------------------------------------------------------------
    orth = res.irf(HORIZON).orth_irfs

    irf = {}
    for resp, shock in [("LGDP", policy), ("UNRATE", policy),
                        ("LGDP", "LM2"), ("UNRATE", "LM2"),
                        ("LGDP", "LWALCL"), ("UNRATE", "LWALCL")]:
        y = orth[:, ix[resp], ix[shock]]
        h_pk = int(np.argmax(np.abs(y)))
        irf[f"{shock}->{resp}"] = {
            "h4": round(float(y[4]), 4),
            "h8": round(float(y[8]), 4),
            "peak": round(float(y[h_pk]), 4),
            "peak_h": h_pk,
        }

    # --- FEVD at 8 quarters (horizons 0..8 inclusive) ---
    def fevd(target):
        ti = ix[target]
        c = np.cumsum(orth[:, ti, :] ** 2, axis=0)
        s = c / c.sum(axis=1, keepdims=True)
        return {n: round(float(s[8, j]) * 100, 1) for j, n in enumerate(order)}

    out = {"order": order, "trace": trace, "beta": beta_d, "alpha": alpha,
           "granger": gc, "ljung_box_min_p": round(float(lb4), 3),
           "ljung_box_min_p_df_adj": round(float(lb), 3),
           "portmanteau": wh, "normality_p": norm_p,
           "max_nonunit_root": max_nonunit,
           "fevd_LGDP": fevd("LGDP"), "fevd_UNRATE": fevd("UNRATE"), "irf": irf}

    if verbose:
        print(f"=== {tag} ===")
        print("  Johansen trace:", " | ".join(
            f"r<={i}: {t[0]} vs {t[1]}" for i, t in enumerate(trace)))
        print("  beta (norm. LGDP):", beta_d)
        print("  alpha:", alpha)
        print("  Granger:", gc)
        print(f"  Ljung-Box(4) min p (equation level) = {lb4:.3f}")
        print("  Portmanteau (system): " + " | ".join(
            f"h{n}: chi2={v[0]} p={v[1]}" for n, v in wh.items())
            + f" | normality p = {norm_p:.2e}")
        print(f"  largest non-unit root = {max_nonunit}")
        print("  FEVD LGDP  :", out["fevd_LGDP"])
        print("  FEVD UNRATE:", out["fevd_UNRATE"])
        print("  IRF (LEVEL responses, pp for UNRATE / log pts for LGDP):")
        for k, v in irf.items():
            print(f"     {k:16} h4={v['h4']:+.4f}  h8={v['h8']:+.4f}  "
                  f"peak={v['peak']:+.4f} @h{v['peak_h']}")
        print()
    return out


# ======================================================================
# 6. TABLE VII - BOOTSTRAP CONFIDENCE INTERVALS FOR THE IMPULSE RESPONSES
#    Recursive residual bootstrap: the data are regenerated from the fitted
#    VECM with resampled residuals, and the ENTIRE system (cointegrating
#    vector, alpha, Gamma, dummies) is re-estimated at every replication.
#    This is what Section 3.3 describes and what Fig. 4 plots.
# ======================================================================
def bootstrap_irf(df, policy="SSR", exog=None, order=None, B=NBOOT, seed=20260713):
    order = order or ["LGDP", "INF", "UNRATE", "LM2", "LWALCL", policy]
    ix = {n: i for i, n in enumerate(order)}
    Y = df[order].values
    T, Kv = Y.shape

    res = VECM(df[order], k_ar_diff=K, coint_rank=RANK,
               deterministic="ci", exog=exog).fit()
    alpha, beta = res.alpha, res.beta
    c0 = float(res.det_coef_coint[0, 0])
    gamma, phi = res.gamma, (res.exog_coefs if exog is not None else None)
    resid = np.asarray(res.resid)
    point = res.irf(HORIZON).orth_irfs

    def simulate(e):
        sim = np.zeros((T, Kv))
        sim[:K + 1] = Y[:K + 1]                      # initial values held fixed
        for t in range(K + 1, T):
            ecm = beta[:, 0] @ sim[t - 1] + c0
            dlags = np.concatenate([sim[t - 1 - i] - sim[t - 2 - i] for i in range(K)])
            dy = alpha[:, 0] * ecm + gamma @ dlags + e[t - K - 1]
            if phi is not None:
                dy = dy + phi @ exog[t]
            sim[t] = sim[t - 1] + dy
        return sim

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        e = resid[rng.integers(0, len(resid), len(resid))]
        sd = pd.DataFrame(simulate(e), index=df.index, columns=order)
        sd.index.freq = "QS-JAN"
        try:
            rb = VECM(sd, k_ar_diff=K, coint_rank=RANK,
                      deterministic="ci", exog=exog).fit()
            draws.append(rb.irf(HORIZON).orth_irfs)
        except Exception:
            pass
    D = np.array(draws)
    print(f"### TABLE VII: bootstrap ({len(draws)}/{B} replications converged) ###")

    pairs = [("UNRATE", "LWALCL"), ("UNRATE", "LM2"),
             ("LGDP", "LWALCL"), ("LGDP", "LM2"),
             ("LGDP", policy), ("UNRATE", policy)]
    tab, bands = {}, {}
    for resp, shock in pairs:
        a = D[:, :, ix[resp], ix[shock]]
        bands[f"{shock}->{resp}"] = {
            "point": point[:, ix[resp], ix[shock]].tolist(),
            "lo90": np.percentile(a, 5, axis=0).tolist(),
            "hi90": np.percentile(a, 95, axis=0).tolist(),
            "lo95": np.percentile(a, 2.5, axis=0).tolist(),
            "hi95": np.percentile(a, 97.5, axis=0).tolist()}
        for h in (4, 8):
            p = float(point[h, ix[resp], ix[shock]])
            l90, u90 = np.percentile(a[:, h], [5, 95])
            l95, u95 = np.percentile(a[:, h], [2.5, 97.5])
            tab[f"{shock}->{resp}_h{h}"] = {
                "point": round(p, 3),
                "ci90": [round(float(l90), 3), round(float(u90), 3)],
                "ci95": [round(float(l95), 3), round(float(u95), 3)]}
            print(f"  {shock:7}->{resp:7} h{h:<2} {p:+.3f}  "
                  f"90% [{l90:+.3f}, {u90:+.3f}]  95% [{l95:+.3f}, {u95:+.3f}]")
    print()
    json.dump(bands, open("irf_bands.json", "w"))
    return tab


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    df = build_dataset()

    OUT["descriptives"] = descriptives(df)
    OUT["reverse_causality"] = reverse_causality(df)
    OUT["adf"] = adf_table(df)

    IMPULSE = df[[c for c in df.columns if c.startswith("D_2")]].values   # 4 dummies
    SPANNING = df[["D_GFC", "D_COVID"]].values                            # 2 dummies

    # ---- the two models reported in the paper ----
    OUT["main"] = estimate(df, "SSR", IMPULSE,
                           "MAIN: shadow rate + 4 impulse dummies")
    OUT["robustness_ffr"] = estimate(df, "FFR", IMPULSE,
                                     "ROBUSTNESS I: nominal FFR + 4 impulse dummies")

    # ---- the two rejected specifications cited in Sections 3.2 and 4.3 ----
    OUT["no_dummies"] = estimate(df, "SSR", None,
                                 "REJECTED: no dummies (sign of QE flips)")
    OUT["spanning"] = estimate(df, "SSR", SPANNING,
                               "REJECTED: two spanning dummies (fails Ljung-Box)")

    # ---- TABLE VII: bootstrap intervals for the central responses (Section 4.3) ----
    OUT["bootstrap"] = bootstrap_irf(df, "SSR", IMPULSE)

    # ---- lag-order justification (Section 3.3) and ROBUSTNESS III (Section 4.7) ----
    print("### Information criteria (Section 3.3) ###")
    print(select_order(df[["LGDP", "INF", "UNRATE", "LM2", "LWALCL", "SSR"]],
                       maxlags=6, deterministic="ci").summary())
    print("\n### Lag-order robustness (Sections 3.3 and 4.7) ###")
    lag_tab = []
    for k in [1, 2, 3, 4]:
        K = k
        r = estimate(df, "SSR", IMPULSE, "", verbose=False)
        qe4 = r["irf"]["LWALCL->UNRATE"]["h4"]
        qe8 = r["irf"]["LWALCL->UNRATE"]["h8"]
        lag_tab.append({"k": k, "lb_min_p": r["ljung_box_min_p"],
                        "portmanteau_p4": r["portmanteau"][4][1],
                        "QE->U_h4": qe4, "QE->U_h8": qe8})
        print(f"  k = {k}: Ljung-Box(4) min p = {r['ljung_box_min_p']:.3f} | "
              f"portmanteau(4) p = {r['portmanteau'][4][1]} | "
              f"QE->U h4 = {qe4:+.3f}  h8 = {qe8:+.3f}")
    OUT["lag_robustness"] = lag_tab
    K = 2
    print()

    # ---- TABLE IX: sensitivity to the recursive ordering (Section 4.6) ----
    print("### TABLE X: Cholesky ordering sensitivity (Section 4.6) ###")
    orderings = [
        ["LGDP", "INF", "UNRATE", "LM2", "LWALCL", "SSR"],   # baseline
        ["LGDP", "INF", "UNRATE", "LWALCL", "LM2", "SSR"],
        ["LGDP", "INF", "UNRATE", "SSR", "LWALCL", "LM2"],
        ["LGDP", "UNRATE", "INF", "LM2", "LWALCL", "SSR"],
        ["LWALCL", "LM2", "SSR", "LGDP", "INF", "UNRATE"],   # policy first
    ]
    ord_tab = []
    for o in orderings:
        r = estimate(df, "SSR", IMPULSE, "", order=o, verbose=False)
        qe = r["irf"]["LWALCL->UNRATE"]["h8"]
        m2 = r["irf"]["LM2->UNRATE"]["h8"]
        ord_tab.append({"order": o, "QE->U_h8": qe, "M2->U_h8": m2})
        print(f"  {', '.join(o):48}  QE->U h8 = {qe:+.3f}   M2->U h8 = {m2:+.3f}")
    OUT["ordering_robustness"] = ord_tab
    print()

    with open("results_final.json", "w") as f:
        json.dump(OUT, f, indent=1)
    print("SAVED dataset_quarterly_with_shadow_rate.csv and results_final.json")

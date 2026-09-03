import sys, json, warnings
import numpy as np
import pandas as pd
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox

import os

# =============================================================================
# LOCAL DATA PATH  --  CLEAR THIS BEFORE PUBLISHING THE REPLICATION PACKAGE
#
# Set this to run the script directly from an editor without passing a
# command-line argument.  It is a convenience for local work only: it hard-codes
# one machine's folder layout and will fail on anyone else's computer.  Set it
# back to "" before uploading, so that the script falls back to looking for the
# data beside itself, which is what a referee needs.
# =============================================================================
LOCAL_DATA_PATH = ""

# Resolution order: command-line argument, then LOCAL_DATA_PATH, then the
# folder this script sits in.
if len(sys.argv) > 1:
    U = sys.argv[1]
elif LOCAL_DATA_PATH:
    U = LOCAL_DATA_PATH
    print("!" * 74)
    print("! WARNING: running with a hard-coded LOCAL_DATA_PATH:")
    print("!   " + LOCAL_DATA_PATH)
    print("! Set LOCAL_DATA_PATH = \"\" near the top of this file before you")
    print("! publish the replication package, or it will fail for everyone else.")
    print("!" * 74)
else:
    U = os.path.dirname(os.path.abspath(__file__))
if not U.endswith(("/", "\\")):
    U += os.sep
# Write all output (data_final.csv, results_final.json, the .npy arrays and the
# figures) into the same folder as the data, rather than into whatever
# directory the script happened to be launched from.
os.chdir(U)
SEED = 11
NBOOT = 1000
H = 20
K_AR_DIFF = 2
RANK = 1
OUT = {}
CHECKS = []


def check(label, got, want, tol):
    ok = got is not None and abs(float(got) - want) <= tol
    CHECKS.append((label, got, want, tol, ok))
    return ok


# =============================================================================
# 1. DATA
# =============================================================================
def load(f, c):
    d = pd.read_csv(U + f, parse_dates=["observation_date"]).rename(
        columns={"observation_date": "date"})
    return d.set_index("date")[[c]]


gdp = load("GDPC1.csv", "GDPC1")
ffr = load("FEDFUNDS.csv", "FEDFUNDS")
m2 = load("M2SL.csv", "M2SL")
unr = load("UNRATE.csv", "UNRATE")
inf = load("DPCCRV1Q225SBEA.csv", "DPCCRV1Q225SBEA")
wal = load("WALCL.csv", "WALCL")

# --- Wu-Xia monthly shadow rate (Atlanta Fed workbook) -----------------------
wb = openpyxl.load_workbook(U + "WuXiaShadowRate.xlsx", data_only=True)
ws = wb["Data"]
rows = [(r[0], r[1], r[2]) for r in ws.iter_rows(min_row=2, values_only=True)
        if r[0] is not None]
wx = pd.DataFrame(rows, columns=["date", "ffr_m", "shadow_m"]).set_index("date")
wx.index = pd.to_datetime(wx.index)

# --- splice: Wu-Xia where published, effective funds rate thereafter ---------
mm = ffr.join(wx[["shadow_m"]], how="left")
mm["SHADOW"] = mm["shadow_m"].where(mm["shadow_m"].notna(), mm["FEDFUNDS"])
print("Wu-Xia months used:", int(mm["shadow_m"].notna().sum()), "of", len(mm),
      "| last published:", wx["shadow_m"].last_valid_index().date())

# splice quality (quoted in Section 3.1)
sub = wx.join(ffr, how="inner")
for a, b in [("2003-01-01", "2007-12-31"), ("2016-01-01", "2019-12-31")]:
    dd = (sub.loc[a:b, "shadow_m"] - sub.loc[a:b, "FEDFUNDS"]).abs()
    print(f"  splice check {a[:4]}-{b[:4]}: MAD={dd.mean():.3f} max={dd.max():.3f}")
    if a.startswith("2003"):
        check("splice MAD 2003-2007", dd.mean(), 0.23, 0.005)
        check("splice max 2003-2007", dd.max(), 0.80, 0.01)
    else:
        check("splice MAD 2016-2019", dd.mean(), 0.10, 0.005)
        check("splice max 2016-2019", dd.max(), 0.27, 0.01)

# NEW: maximum divergence inside the two ZLB windows (Section 3.1 quotes ~3 pp)
zlb = pd.concat([sub.loc["2008-12-01":"2015-12-31"], sub.loc["2020-03-01":"2022-02-28"]])
zlb_gap = (zlb["shadow_m"] - zlb["FEDFUNDS"]).abs().max()
print(f"  max |shadow - FFR| inside the ZLB windows: {zlb_gap:.2f} pp")
check("max ZLB divergence (pp)", zlb_gap, 3.0, 0.35)

Q = lambda d: d.resample("QS").mean()
df = pd.concat([Q(gdp), Q(inf), Q(unr), Q(m2), Q(wal), Q(ffr), Q(mm[["SHADOW"]])], axis=1)
df.columns = ["GDPC1", "INF", "UNRATE", "M2SL", "WALCL", "FEDFUNDS", "SHADOW"]
df = df.loc["2003-01-01":"2025-10-01"].copy()
try:
    df.index.freq = "QS-JAN"
except (ValueError, AttributeError):
    pass

df["LGDP"] = np.log(df["GDPC1"])
df["LM2"] = np.log(df["M2SL"])
df["LWALCL"] = np.log(df["WALCL"])
df["FFR"] = df["FEDFUNDS"]
df["SSR"] = df["SHADOW"]

# --- crisis dummies ----------------------------------------------------------
CRISIS = ["2008-10-01", "2009-01-01", "2020-04-01", "2020-07-01"]  # 08Q4 09Q1 20Q2 20Q3
IMPULSE = []
for q in CRISIS:                       # FOUR separate impulse dummies (paper)
    name = "D_" + q[:7]
    df[name] = (df.index == pd.Timestamp(q)).astype(float)
    IMPULSE.append(name)
# TWO spanning dummies, kept only for the Section 3.2 comparison
df["D_GFC"] = df.index.isin(pd.to_datetime(CRISIS[:2])).astype(float)
df["D_COVID"] = df.index.isin(pd.to_datetime(CRISIS[2:])).astype(float)
SPAN = ["D_GFC", "D_COVID"]

assert df[IMPULSE].sum().eq(1).all(), "impulse dummies mis-dated"
assert not df.isna().any().any(), "missing values in estimation sample"
df.to_csv("data_final.csv")
print(f"\nSample {df.index.min().date()} -> {df.index.max().date()}  n={len(df)}")
check("sample size n", len(df), 92, 0)

VARS = ["LGDP", "INF", "UNRATE", "LM2", "LWALCL", "SSR", "FFR"]
EX = df[IMPULSE].values
EXS = df[SPAN].values

# --- Table II ---------------------------------------------------------------
desc = pd.DataFrame({
    "mean": [df.GDPC1.mean(), df.INF.mean(), df.UNRATE.mean(), df.M2SL.mean(),
             df.WALCL.mean() / 1e6, df.SSR.mean(), df.FFR.mean()],
    "sd": [df.GDPC1.std(), df.INF.std(), df.UNRATE.std(), df.M2SL.std(),
           df.WALCL.std() / 1e6, df.SSR.std(), df.FFR.std()],
    "min": [df.GDPC1.min(), df.INF.min(), df.UNRATE.min(), df.M2SL.min(),
            df.WALCL.min() / 1e6, df.SSR.min(), df.FFR.min()],
    "max": [df.GDPC1.max(), df.INF.max(), df.UNRATE.max(), df.M2SL.max(),
            df.WALCL.max() / 1e6, df.SSR.max(), df.FFR.max()]},
    index=["RealGDP", "INF", "UNRATE", "M2", "FedAssets(tn)", "SSR", "FFR"])
print("\n### Table II: descriptive statistics ###")
print(desc.round(2))
ratio = df.SSR.std() / df.FFR.std()
print(f"SSR/FFR dispersion ratio = {ratio:.3f}")
print(f"SSR trough = {df.SSR.min():.2f} in {df.SSR.idxmin():%Y}Q{df.SSR.idxmin().quarter}"
      f"; FFR that quarter = {df.FFR[df.SSR.idxmin()]:.2f}")
OUT["desc"] = desc.round(3).to_dict()
# Table II, every cell quoted in the paper
for _lab, _ser, _m, _sd, _lo, _hi in [
        ("Real GDP", df.GDPC1, 18795, 2603, 14614, 24056),
        ("Core PCE inflation", df.INF, 2.16, 1.22, -0.80, 6.10),
        ("Unemployment", df.UNRATE, 5.75, 2.02, 3.53, 13.00),
        ("M2", df.M2SL, 12644, 5426, 5843, 22292),
        ("Fed assets (tn)", df.WALCL / 1e6, 3.92, 2.59, 0.72, 8.93),
        ("SSR", df.SSR, 1.28, 2.41, -2.92, 5.33),
        ("FFR", df.FFR, 1.76, 1.90, 0.06, 5.33)]:
    _tol = 1.0 if abs(_m) > 100 else 0.006   # paper rounds to 2 dp
    check(f"Table II mean, {_lab}", _ser.mean(), _m, _tol)
    check(f"Table II sd,   {_lab}", _ser.std(), _sd, _tol)
    check(f"Table II min,  {_lab}", _ser.min(), _lo, _tol)
    check(f"Table II max,  {_lab}", _ser.max(), _hi, _tol)
check("SSR trough in 2014Q2", float(df.SSR.idxmin() == pd.Timestamp("2014-04-01")), 1.0, 0)
check("SSR sd", df.SSR.std(), 2.41, 0.005)
check("FFR sd", df.FFR.std(), 1.90, 0.005)
check("SSR/FFR dispersion ratio", ratio, 1.27, 0.005)
check("SSR minimum", df.SSR.min(), -2.92, 0.005)
check("FFR in 2014Q2", df.FFR[df.SSR.idxmin()], 0.09, 0.005)

# --- Table III: the reverse-causality arithmetic ----------------------------
d1 = df.diff().dropna()
mask = ~d1.index.isin(pd.to_datetime(CRISIS))
c_all = d1.UNRATE.corr(d1.LWALCL)
c_ex = d1.loc[mask, "UNRATE"].corr(d1.loc[mask, "LWALCL"])
print("\n### Table III ###")
print(d1.loc[pd.to_datetime(CRISIS), ["UNRATE", "LWALCL"]].round(3))
print(f"corr(dUNRATE,dLWALCL): full={c_all:.3f} (n={len(d1)})  excl. 4 = {c_ex:.3f}")
print(f"corr(LM2,LWALCL): levels={df.LM2.corr(df.LWALCL):.3f} diffs={d1.LM2.corr(d1.LWALCL):.3f}")
print("largest |d log GDP| (%):")
print((d1.LGDP.abs() * 100).sort_values(ascending=False).head(5).round(2))
OUT["tableIII"] = {"corr_full": round(c_all, 3), "corr_excl4": round(c_ex, 3),
                   "n_diff": int(len(d1))}
check("corr full sample", c_all, 0.45, 0.005)
check("corr excluding 4", c_ex, -0.16, 0.006)
check("n first differences", len(d1), 91, 0)
check("corr(LM2,LWALCL) levels", df.LM2.corr(df.LWALCL), 0.95, 0.005)
check("corr(LM2,LWALCL) diffs", d1.LM2.corr(d1.LWALCL), 0.54, 0.005)
# NEW in v5: the individual cells of Table III were printed but never checked.
for _q, _du, _dw in [("2008-10-01", 0.87, 0.77), ("2009-01-01", 1.40, -0.03),
                     ("2020-04-01", 9.17, 0.45), ("2020-07-01", -4.20, 0.04)]:
    check(f"Table III dUNRATE {_q[:7]}", d1.UNRATE[pd.Timestamp(_q)], _du, 0.006)
    check(f"Table III dLWALCL {_q[:7]}", d1.LWALCL[pd.Timestamp(_q)], _dw, 0.006)
# NEW in v5: Section 3.2 asserts the 2020 observations are "more than three
# times the next largest".  Verified rather than asserted.
_g = (d1.LGDP.abs() * 100).sort_values(ascending=False)
_ratio = _g.iloc[1] / _g.iloc[2]          # smaller of the two 2020 quarters / next
print(f"2020 leverage: top two = {_g.iloc[0]:.2f}, {_g.iloc[1]:.2f}; "
      f"next = {_g.iloc[2]:.2f}; ratio = {_ratio:.2f}")
check("2020 quarters > 3x the next largest", float(_ratio > 3.0), 1.0, 0)

# =============================================================================
# 2. UNIT ROOTS  (Table IV + the trend specification discussed in 4.1)
# =============================================================================
print("\n### Table IV: ADF (constant, no trend) ###")
adf_res = {}
TAB4 = {"LGDP": -0.08, "INF": -2.46, "UNRATE": -2.70, "LM2": -0.40,
        "LWALCL": -1.42, "SSR": -2.42, "FFR": -2.99}
TAB4P = {"LGDP": 0.952, "INF": 0.125, "UNRATE": 0.074, "LM2": 0.909,
         "LWALCL": 0.573, "SSR": 0.137, "FFR": 0.036}
TAB4D = {"LGDP": -11.35, "INF": -9.89, "UNRATE": -11.19, "LM2": -5.67,
         "LWALCL": -8.62, "SSR": -3.86, "FFR": -3.93}
# Section 4.1 quotes these ADF-with-trend p-values in the running text
ADFCT_P = {"LGDP": 0.274, "INF": 0.193, "UNRATE": 0.142, "LM2": 0.066,
           "LWALCL": 0.933, "SSR": 0.365, "FFR": 0.140}
for v in VARS:
    a0 = adfuller(df[v].dropna(), autolag="AIC")
    a1 = adfuller(df[v].diff().dropna(), autolag="AIC")
    adf_res[v] = dict(lvl_t=round(a0[0], 2), lvl_p=round(a0[1], 3),
                      d_t=round(a1[0], 2), d_p=round(a1[1], 3))
    print(f" {v:7} lvl t={a0[0]:7.2f} p={a0[1]:.3f} | diff t={a1[0]:7.2f} p={a1[1]:.3f}")
    check(f"ADF(c) level t, {v}", a0[0], TAB4[v], 0.006)
    check(f"ADF(c) level p, {v}", a0[1], TAB4P[v], 0.001)
    check(f"ADF(c) diff  t, {v}", a1[0], TAB4D[v], 0.006)
print("\n### ADF with a linear trend, and KPSS(trend) ###")
kp = {}
for v in VARS:
    a = adfuller(df[v].dropna(), regression="ct", autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # p-value interpolation warning
        ks, kp_p, kl, kcv = kpss(df[v].dropna(), regression="ct", nlags="auto")
    kp[v] = dict(adf_ct_t=round(a[0], 2), adf_ct_p=round(a[1], 3),
                 kpss=round(ks, 3), kpss_p=round(kp_p, 3),
                 rej5=bool(ks > kcv["5%"]), rej10=bool(ks > kcv["10%"]))
    print(f" {v:7} ADF(ct) t={a[0]:7.2f} p={a[1]:.3f} | KPSS={ks:.3f} "
          f"(5% cv {kcv['5%']}) reject5%={ks>kcv['5%']} reject10%={ks>kcv['10%']}")
    check(f"ADF(ct) p, {v}", a[1], ADFCT_P[v], 0.001)
# Section 4.1: no series rejects the unit root at 5% under the trend
# specification, and KPSS rejects trend-stationarity at 5% for all but LM2.
n_adf_ct_rej5 = sum(kp[v]["adf_ct_p"] < 0.05 for v in VARS)
n_kpss_rej5 = sum(kp[v]["rej5"] for v in VARS)
print(f" -> ADF(ct) rejects at 5% for {n_adf_ct_rej5} of 7 series (paper: 0)")
print(f" -> KPSS(ct) rejects at 5% for {n_kpss_rej5} of 7 series (paper: 6, all but LM2)")
check("ADF(ct) 5% rejections", n_adf_ct_rej5, 0, 0)
check("KPSS(ct) 5% rejections", n_kpss_rej5, 6, 0)
check("KPSS(ct) LM2 rejects at 5%", float(kp["LM2"]["rej5"]), 0.0, 0)
OUT["adf"], OUT["adf_ct_kpss"] = adf_res, kp

# =============================================================================
# 2b. LAG-ORDER INFORMATION CRITERIA  (asserted in Section 3.3, never computed
#     in v2).  Reported on the levels VAR, as select_order does.
# =============================================================================
print("\n### Section 3.3: information criteria, levels VAR ###")
try:
    sel = VAR(df[["LGDP", "INF", "UNRATE", "LM2", "LWALCL", "SSR"]]).select_order(
        maxlags=7)
    print(sel.summary())
    ic_pick = {c: int(getattr(sel, c)) for c in ["aic", "bic", "hqic", "fpe"]}
    print(" levels-VAR order chosen:", ic_pick,
          "\n -> in k_ar_diff terms:", {k: v - 1 for k, v in ic_pick.items()})
    OUT["lag_ic"] = ic_pick
    # Section 3.3: Schwarz and Hannan-Quinn pick k = 1; AIC and FPE keep falling
    # to the edge of the searchable range, i.e. no interior optimum.
    check("BIC k_ar_diff", ic_pick["bic"] - 1, 1, 0)
    check("HQ  k_ar_diff", ic_pick["hqic"] - 1, 1, 0)
except Exception as e:
    print(f" (select_order unavailable: {type(e).__name__}: {e})")

# =============================================================================
# 3. HELPERS
# =============================================================================
def fevd_at(orth, names, target, h=8):
    """Share of the h-step forecast-error variance of `target`, horizons 0..h."""
    ti = names.index(target)
    c = np.cumsum(orth[:, ti, :] ** 2, axis=0)
    s = c / c.sum(axis=1, keepdims=True)
    return {n: round(s[h, j] * 100, 1) for j, n in enumerate(names)}


def multivariate_portmanteau(resid, h, k_ar_diff=K_AR_DIFF, rank=RANK):
    """Adjusted (Hosking) multivariate Ljung-Box statistic and p-value.

    Degrees of freedom follow Luetkepohl (2005, ch. 8) as implemented in
    statsmodels' VECMResults.test_whiteness:

        df = K^2 * (h - p + 1) - K * r,     p = k_ar_diff + 1 (VAR order in levels)

    The K*r term for the loadings must not be omitted: with K = 6, h = 4,
    k_ar_diff = 2, rank = 1 the correct df is 66, not 72, and the p-value of
    the main model is 0.020 (reject at 5%) rather than 0.058.  df <= 0 means
    the statistic is not defined at that lag order (this happens at k = 4).
    """
    E = np.asarray(resid)
    n, Kd = E.shape
    C0i = np.linalg.inv(E.T @ E / n)
    Q = 0.0
    for i in range(1, h + 1):
        Ci = E[i:].T @ E[:-i] / n
        Q += np.trace(Ci.T @ C0i @ Ci @ C0i) / (n - i)
    Q *= n ** 2
    dfree = Kd * Kd * (h - (k_ar_diff + 1) + 1) - Kd * rank
    p = float(stats.chi2.sf(Q, dfree)) if dfree > 0 else None
    return Q, dfree, p


def companion_roots(res, Kd):
    """Moduli of the eigenvalues of the implied VAR companion matrix."""
    A = list(res.var_rep)
    p = len(A)
    C = np.zeros((Kd * p, Kd * p))
    C[:Kd] = np.hstack(A)
    if p > 1:
        C[Kd:, :Kd * (p - 1)] = np.eye(Kd * (p - 1))
    return np.sort(np.abs(np.linalg.eigvals(C)))[::-1]


def fit(order, k=K_AR_DIFF, exog=EX, data=None):
    d = df if data is None else data
    return VECM(d[order], k_ar_diff=k, coint_rank=RANK,
                deterministic="ci", exog=exog).fit()


def irf(res, names, shock, resp, h, horizon=H):
    o = res.irf(horizon).orth_irfs
    return o[h, names.index(resp), names.index(shock)]


def granger_wald(names, caused, causing, k=K_AR_DIFF, exog=EX, data=None):
    """Wald test on the short-run Gamma block, conditional on the estimated beta.

    Self-contained on purpose.  v3 tried to slice statsmodels'
    `cov_params_wo_det`, but that matrix is np.kron(mat1, sigma_u), i.e.
    PARAMETER-major (flat index of parameter j in equation i is j*K + i), and
    its per-equation block also contains the exogenous dummy coefficients.
    v3 assumed an equation-major layout with a block length of RANK + K*k,
    omitting the dummies, and therefore extracted the wrong sub-matrix.

    Here the regression is rebuilt from the data instead:

        dY_t = alpha (beta' [Y_{t-1}; 1]) + sum_i Gamma_i dY_{t-i} + Phi D_t + e

    Given beta the system is a seemingly-unrelated regression with identical
    regressors, so equation-by-equation OLS is efficient and
    Var(vec B) = Sigma (x) (X'X)^-1.  The null is that the k coefficients on
    `causing` in the `caused` equation are jointly zero.  Both the ML variant
    (Sigma = e'e / T, matching statsmodels' convention) and the
    df-corrected variant are returned.
    """
    d = df if data is None else data
    Y = d[names].values
    T, K = Y.shape
    dY = np.diff(Y, axis=0)
    dY0 = dY[k:]
    n = len(dY0)
    y_lag1 = np.hstack([Y[k:-1], np.ones((n, 1))])          # restricted constant
    Z = np.hstack([dY[k - i:len(dY) - i] for i in range(1, k + 1)])
    if exog is not None:
        Z = np.hstack([Z, np.asarray(exog, float)[k + 1:]])

    # concentrate out Z, then solve the reduced-rank problem for beta
    def resid_on(M):
        b, *_ = np.linalg.lstsq(Z, M, rcond=None)
        return M - Z @ b
    R0, R1 = resid_on(dY0), resid_on(y_lag1)
    S00 = R0.T @ R0 / n
    S11 = R1.T @ R1 / n
    S01 = R0.T @ R1 / n
    C = np.linalg.cholesky(S11)
    Ci = np.linalg.inv(C)
    Msym = Ci @ S01.T @ np.linalg.solve(S00, S01) @ Ci.T
    lam, V = np.linalg.eigh((Msym + Msym.T) / 2)
    V = V[:, np.argsort(lam)[::-1]]
    beta_star = Ci.T @ V[:, :RANK]
    beta_star = beta_star @ np.linalg.inv(beta_star[:RANK, :])   # eye(r) on top

    X = np.hstack([y_lag1 @ beta_star, Z])
    coef, *_ = np.linalg.lstsq(X, dY0, rcond=None)
    e = dY0 - X @ coef
    npar = X.shape[1]
    XtXi = np.linalg.inv(X.T @ X)
    ci, xi = names.index(caused), names.index(causing)
    idx = [RANK + i * K + xi for i in range(k)]     # Gamma_i[caused, causing]
    b = coef[idx, ci]
    out = {}
    for lab, S in [("ml", e.T @ e / n), ("df", e.T @ e / (n - npar))]:
        V_ = XtXi[np.ix_(idx, idx)] * S[ci, ci]
        W = float(b @ np.linalg.solve(V_, b))
        out[lab] = (W, float(stats.chi2.sf(W, k)),
                    float(stats.f.sf(W / k, k, n - npar)))
    return out


# =============================================================================
# 4. MAIN AND ROBUSTNESS MODELS  (Tables V, VI, VII, VIII, IX)
# =============================================================================
def estimate(pol, tag):
    names = ["LGDP", "INF", "UNRATE", "LM2", "LWALCL", pol]
    sysdf = df[names]
    print(f"\n{'='*70}\n=== {tag} ===\n{'='*70}")

    # Johansen. det_order=0 -> unrestricted constant; the VECM below restricts
    # the constant to the cointegration space, so the two deterministic
    # specifications differ. Both are reported; see Section 4.1.
    jo = coint_johansen(sysdf, det_order=0, k_ar_diff=K_AR_DIFF)
    trace = [(round(jo.lr1[i], 2), round(jo.cvt[i, 1], 2), round(jo.cvt[i, 2], 2))
             for i in range(len(names))]
    print(" Johansen trace (stat, 5% cv, 1% cv), det_order=0:")
    for i, (s, c5, c1) in enumerate(trace):
        print(f"   r<={i}: {s:8.2f}  5% {c5:7.2f}  1% {c1:7.2f}  "
              f"{'reject' if s > c5 else 'do not reject'}")
    jo_ci = coint_johansen(sysdf, det_order=-1, k_ar_diff=K_AR_DIFF)
    print(" (det_order=-1 comparison:", np.round(jo_ci.lr1, 2), ")")

    res = fit(names)
    # statsmodels normalises beta so that eye(r) forms its first r rows, i.e.
    # beta[0, 0] == 1 identically.  `scale` is therefore always +1; it is kept
    # only so the code stays correct if that normalisation ever changes.
    scale = float(res.beta[0, 0])
    bn = res.beta[:, 0] / scale
    print("\n cointegrating vector, b'y = 0, normalised on LGDP:")
    print("   ", {n: round(b, 4) for n, b in zip(names, bn)},
          "| restricted const:", round(float(res.det_coef_coint[0, 0] / scale), 4))
    # (B) the paper reports the SOLVED relation; its coefficients are -bn
    print(" solved long-run relation (Section 4.2), LGDP = ...")
    print("   ", {n: round(-b, 4) for n, b in zip(names[1:], bn[1:])})
    print(" alpha, statsmodels t (ML covariance) and df-corrected t:")
    # statsmodels forms Var(alpha) from Sigma_ML = resid'resid / T, with no
    # degrees-of-freedom correction.  Table VI reports those t-ratios; the
    # df-corrected ones (factor sqrt(T / (T - npar))) are printed beside them
    # because the DLWALCL loading is significant at 10% only without it.
    sgn = np.sign(scale)
    npar_eq = RANK + len(names) * K_AR_DIFF + (0 if EX is None else EX.shape[1])
    dfadj = np.sqrt((res.nobs - npar_eq) / res.nobs)
    alpha = [(round(res.alpha[i, 0] * scale, 4),
              round(res.tvalues_alpha[i, 0] * sgn, 2))
             for i in range(len(names))]
    for n, (a, t) in zip(names, alpha):
        print(f"   {n:7} alpha={a:8.4f}  t={t:6.2f}  t(df-corr)={t * dfadj:6.2f}")

    # diagnostics
    r = pd.DataFrame(res.resid)
    lb = {c: float(acorr_ljungbox(r[c], lags=[4], return_df=True)["lb_pvalue"].iloc[0])
          for c in r.columns}
    Qm, dfree, pm = multivariate_portmanteau(res.resid, 4)
    print(f"\n Ljung-Box(4) min p across equations = {min(lb.values()):.3f}")
    print(f" multivariate portmanteau(4): Q={Qm:.1f} df={dfree} p={pm:.4f}")
    roots = companion_roots(res, len(names))
    print(" companion roots (largest 8):", np.round(roots[:8], 4))

    # Granger causality (short-run block only; ECM channel unrestricted)
    gc = {}
    for c in ["LGDP", "UNRATE"]:
        for x in [pol, "LWALCL", "LM2"]:
            t = res.test_granger_causality(caused=c, causing=x)
            w = granger_wald(names, c, x)
            gc[f"{x}->{c}"] = (round(float(t.test_statistic), 3),
                               round(float(t.pvalue), 4),
                               round(w["ml"][1], 4), round(w["df"][2], 4))
            print(f" GC {x:6}->{c:6}: stat={t.test_statistic:7.3f} "
                  f"p={t.pvalue:.4f}  [Wald chi2 p={w['ml'][1]:.4f}, "
                  f"exact F p={w['df'][2]:.4f}]")

    orth = res.irf(H).orth_irfs
    fev_g, fev_u = fevd_at(orth, names, "LGDP"), fevd_at(orth, names, "UNRATE")
    print("\n FEVD(8) of LGDP  :", fev_g)
    print(" FEVD(8) of UNRATE:", fev_u)

    irfk = {}
    for s_, r_ in [(pol, "LGDP"), (pol, "UNRATE"), ("LM2", "LGDP"),
                   ("LM2", "UNRATE"), ("LWALCL", "LGDP"), ("LWALCL", "UNRATE")]:
        y = orth[:, names.index(r_), names.index(s_)]
        pk = int(np.argmax(np.abs(y)))
        irfk[f"{s_}->{r_}"] = {"h4": round(float(y[4]), 4), "h8": round(float(y[8]), 4),
                               "h12": round(float(y[12]), 4),
                               "peak": round(float(y[pk]), 4), "peak_h": pk}
        print(f" IRF {s_:6}->{r_:6} h4={y[4]:+.4f} h8={y[8]:+.4f} peak={y[pk]:+.4f} @h{pk}")
    np.save(f"orth_{pol}.npy", orth)
    return res, names, dict(trace=trace, beta=[round(b, 4) for b in bn],
                            longrun={n: round(-b, 4) for n, b in zip(names[1:], bn[1:])},
                            alpha=alpha,
                            lb_min=round(min(lb.values()), 3),
                            portmanteau=(round(Qm, 1), int(dfree),
                                         None if pm is None else round(pm, 4)),
                            roots=[round(x, 4) for x in roots[:8]], gc=gc,
                            fevd_lgdp=fev_g, fevd_unrate=fev_u, irf=irfk)


res_main, NAMES, OUT["main"] = estimate("SSR", "MAIN: shadow rate + four impulse dummies")
res_rob, ROBN, OUT["rob"] = estimate("FFR", "ROBUSTNESS I: nominal FFR + four impulse dummies")

M = OUT["main"]
check("Johansen trace r=0 (main)", M["trace"][0][0], 124.72, 0.02)
check("Johansen 5% cv, r=0", M["trace"][0][1], 95.75, 0.02)
check("Johansen 1% cv, r=0", M["trace"][0][2], 104.96, 0.02)
check("Johansen trace r<=1 (main)", M["trace"][1][0], 70.97, 0.02)
check("Johansen 5% cv, r<=1", M["trace"][1][1], 69.82, 0.02)
# CORRECTION (v5).  The 1% trace critical value for n - r = 5 in the
# constant case is 77.8202 (Osterwald-Lenum 1992 / MacKinnon-Haug-Michelis;
# statsmodels' ejcp1 table), NOT 76.97 as stated in Section 4.1 of the
# manuscript.  The conclusion is unaffected -- the r <= 1 trace statistic is
# 70.97, which fails to reject at 1% against either value -- but the number
# quoted in the text must be corrected to 77.82.
check("Johansen 1% cv, r<=1", M["trace"][1][2], 77.82, 0.02)
# remaining rows of Table V, previously printed but never checked
for _i, _w in [(2, 37.30), (3, 15.89), (4, 3.29), (5, 0.12)]:
    check(f"Johansen trace r<={_i} (main)", M["trace"][_i][0], _w, 0.02)
# Section 4.2 prints the whole solved relation, so check the whole thing
check("long-run M2 elasticity", M["longrun"]["LM2"], 0.726, 0.002)
check("long-run LWALCL coefficient", M["longrun"]["LWALCL"], -0.212, 0.002)
check("long-run INF coefficient", M["longrun"]["INF"], -0.066, 0.002)
check("long-run UNRATE coefficient", M["longrun"]["UNRATE"], 0.004, 0.002)
check("long-run SSR coefficient", M["longrun"]["SSR"], 0.014, 0.002)
check("alpha INF", M["alpha"][1][0], 0.3857, 0.0005)
check("alpha UNRATE", M["alpha"][2][0], -0.8779, 0.0005)
check("alpha LM2", M["alpha"][3][0], -0.0339, 0.0005)
check("alpha LWALCL", M["alpha"][4][0], -0.0431, 0.0005)
check("alpha SSR", M["alpha"][5][0], 0.2768, 0.0005)
check("t(alpha) INF", M["alpha"][1][1], 0.54, 0.02)
check("t(alpha) SSR", M["alpha"][5][1], 0.67, 0.02)
check("alpha LGDP", M["alpha"][0][0], -0.0232, 0.0002)
check("t(alpha) LGDP", M["alpha"][0][1], -4.16, 0.02)
check("t(alpha) UNRATE", M["alpha"][2][1], -4.84, 0.02)
check("t(alpha) LM2", M["alpha"][3][1], -5.52, 0.02)
check("t(alpha) LWALCL", M["alpha"][4][1], -1.75, 0.02)
check("Ljung-Box min p (main)", M["lb_min"], 0.317, 0.002)
check("portmanteau Q(4), main", M["portmanteau"][0], 91.8, 0.15)
check("portmanteau df, main", M["portmanteau"][1], 66, 0)
check("portmanteau p, main", M["portmanteau"][2], 0.020, 0.001)
check("largest non-unit root (main)", M["roots"][5], 0.992, 0.002)
check("FEVD(8) LGDP <- SSR", M["fevd_lgdp"]["SSR"], 22.6, 0.06)
# remaining Table VIII cells
check("FEVD(8) LGDP <- LGDP", M["fevd_lgdp"]["LGDP"], 66.5, 0.06)
check("FEVD(8) LGDP <- INF", M["fevd_lgdp"]["INF"], 8.5, 0.06)
check("FEVD(8) LGDP <- UNRATE", M["fevd_lgdp"]["UNRATE"], 2.0, 0.06)
check("FEVD(8) LGDP <- LM2", M["fevd_lgdp"]["LM2"], 0.3, 0.06)
check("FEVD(8) LGDP <- LWALCL", M["fevd_lgdp"]["LWALCL"], 0.0, 0.06)
check("FEVD(8) UNRATE <- LGDP", M["fevd_unrate"]["LGDP"], 22.9, 0.06)
check("FEVD(8) UNRATE <- INF", M["fevd_unrate"]["INF"], 2.4, 0.06)
check("FEVD(8) UNRATE <- UNRATE", M["fevd_unrate"]["UNRATE"], 55.6, 0.06)
# Section 5.1: the two quantity instruments jointly explain 18.8%
check("FEVD(8) UNRATE <- LM2+LWALCL",
      M["fevd_unrate"]["LM2"] + M["fevd_unrate"]["LWALCL"], 18.8, 0.12)
# Table IX, Granger row.
# NOTE (v5).  The value 0.384 quoted in Table IX comes from statsmodels'
# VECMResults.test_granger_causality, whose covariance accounts for the
# estimation of beta.  It is therefore SOFTWARE- AND VERSION-SPECIFIC: the
# third digit should not be treated as a property of the data.  The tolerance
# is widened accordingly, and the self-contained Wald test below -- which
# conditions on beta and touches no statsmodels internals -- is checked
# instead as the implementation-independent statistic.  All three variants
# (statsmodels 0.384, Wald 0.412, exact F 0.491) agree that the null is not
# rejected at any conventional level, which is the only use the paper makes
# of the number.  Recommend adding a footnote to Table IX saying so.
check("Granger SSR->LGDP p (Table IX, statsmodels)", M["gc"]["SSR->LGDP"][1], 0.384, 0.05)
check("Granger SSR->LGDP p (independent Wald)", M["gc"]["SSR->LGDP"][2], 0.4118, 0.005)
check("Granger SSR->LGDP: null not rejected at 10%",
      float(M["gc"]["SSR->LGDP"][1] > 0.10), 1.0, 0)
check("FEVD(8) UNRATE <- LM2", M["fevd_unrate"]["LM2"], 9.2, 0.06)
check("FEVD(8) UNRATE <- LWALCL", M["fevd_unrate"]["LWALCL"], 9.6, 0.06)
check("FEVD(8) UNRATE <- SSR", M["fevd_unrate"]["SSR"], 0.2, 0.06)
check("QE -> U, h4", M["irf"]["LWALCL->UNRATE"]["h4"], -0.091, 0.002)
check("QE -> U, h8", M["irf"]["LWALCL->UNRATE"]["h8"], -0.136, 0.002)
check("QE -> U, peak", M["irf"]["LWALCL->UNRATE"]["peak"], -0.149, 0.002)
check("QE -> U, peak horizon", M["irf"]["LWALCL->UNRATE"]["peak_h"], 13, 0)
check("M2 -> U, h8", M["irf"]["LM2->UNRATE"]["h8"], -0.125, 0.002)
check("M2 -> U, peak", M["irf"]["LM2->UNRATE"]["peak"], -0.132, 0.002)
check("SSR -> GDP, h8 (log pts)", M["irf"]["SSR->LGDP"]["h8"], 0.0047, 0.0005)
R = OUT["rob"]
check("Johansen trace r=0 (robustness)", R["trace"][0][0], 101.97, 0.02)
check("Ljung-Box min p (robustness)", R["lb_min"], 0.250, 0.002)
check("alpha LGDP (robustness)", R["alpha"][0][0], -0.020, 0.001)
check("t(alpha) LGDP (robustness)", R["alpha"][0][1], -4.13, 0.02)
check("long-run M2 elasticity (robustness)", R["longrun"]["LM2"], 0.844, 0.002)
check("FEVD(8) LGDP <- FFR", R["fevd_lgdp"]["FFR"], 35.0, 0.06)
check("QE -> U, h8 (robustness)", R["irf"]["LWALCL->UNRATE"]["h8"], -0.15, 0.006)
check("M2 -> U, h8 (robustness)", R["irf"]["LM2->UNRATE"]["h8"], -0.11, 0.006)
check("largest root (robustness)", R["roots"][0], 1.000, 0.002)
# statsmodels-specific (see the note above); the independent Wald test gives
# 0.0025 and the exact F test 0.0105.  All reject at the 5% level.
check("Granger FFR->LGDP p (Table IX, statsmodels)", R["gc"]["FFR->LGDP"][1], 0.003, 0.005)
check("Granger FFR->LGDP p (independent Wald)", R["gc"]["FFR->LGDP"][2], 0.0025, 0.002)
check("Granger FFR->LGDP: null rejected at 5%",
      float(R["gc"]["FFR->LGDP"][1] < 0.05), 1.0, 0)

# =============================================================================
# 5. SECTION 3.2 / 4.3: THE DUMMY SPECIFICATION
# =============================================================================
print(f"\n{'='*70}\n=== Dummy specification (Sections 3.2 and 4.3) ===\n{'='*70}")
OUT["dummies"] = {}
for tag, ex in [("no dummies", None), ("two spanning", EXS), ("four impulse", EX)]:
    r_ = fit(NAMES, exog=ex)
    rr = pd.DataFrame(r_.resid)
    lbm = min(float(acorr_ljungbox(rr[c], lags=[4], return_df=True)["lb_pvalue"].iloc[0])
              for c in rr.columns)
    Qm, _, pm = multivariate_portmanteau(r_.resid, 4)
    h8 = irf(r_, NAMES, "LWALCL", "UNRATE", 8)
    h12 = irf(r_, NAMES, "LWALCL", "UNRATE", 12)
    OUT["dummies"][tag] = dict(h8=round(float(h8), 4), h12=round(float(h12), 4),
                               lb_min=round(lbm, 3), Q4=round(Qm, 1),
                               Q4_p=None if pm is None else round(pm, 4))
    print(f" {tag:13} QE->U h8={h8:+.4f} h12={h12:+.4f} LB min p={lbm:.3f} "
          f"Q(4)={Qm:.1f} p={pm:.4f}")
check("no-dummy QE->U h8", OUT["dummies"]["no dummies"]["h8"], 0.07, 0.006)
check("no-dummy QE->U h12", OUT["dummies"]["no dummies"]["h12"], 0.08, 0.006)
check("spanning QE->U h8", OUT["dummies"]["two spanning"]["h8"], 0.23, 0.006)
check("spanning portmanteau Q(4)", OUT["dummies"]["two spanning"]["Q4"], 181.0, 0.15)
check("spanning Ljung-Box min p", OUT["dummies"]["two spanning"]["lb_min"], 0.000, 0.0006)

# NOTE: deleting the quarters breaks the time index; the rows are re-indexed so
# that the remaining observations are treated as consecutive, which is what the
# "drop the quarters" check in Section 4.7 means.
drop = df.loc[~df.index.isin(pd.to_datetime(CRISIS))][NAMES].reset_index(drop=True)
r_ = VECM(drop, k_ar_diff=K_AR_DIFF, coint_rank=RANK, deterministic="ci").fit()
h8_del = float(irf(r_, NAMES, "LWALCL", "UNRATE", 8))
print(f" crisis quarters deleted outright: QE->U h8={h8_del:+.4f}")
OUT["dummies"]["deleted"] = round(h8_del, 4)
check("crisis quarters deleted, QE->U h8", h8_del, -0.51, 0.006)

# =============================================================================
# 6. TABLE X: THE RECURSIVE ORDERING
# =============================================================================
print(f"\n{'='*70}\n=== Table X: recursive ordering ===\n{'='*70}")
ORDERS = [["LGDP", "INF", "UNRATE", "LM2", "LWALCL", "SSR"],
          ["LGDP", "INF", "UNRATE", "LWALCL", "LM2", "SSR"],
          ["LGDP", "INF", "UNRATE", "SSR", "LWALCL", "LM2"],
          ["LGDP", "UNRATE", "INF", "LM2", "LWALCL", "SSR"],
          ["LWALCL", "LM2", "SSR", "LGDP", "INF", "UNRATE"]]
TABX = [(-0.136, -0.125), (-0.126, -0.135), (-0.128, -0.132),
        (-0.136, -0.125), (-0.155, -0.155)]
OUT["orderings"] = []
for i, o in enumerate(ORDERS):
    r_ = fit(o)
    qe = float(irf(r_, o, "LWALCL", "UNRATE", 8))
    m2_ = float(irf(r_, o, "LM2", "UNRATE", 8))
    OUT["orderings"].append({"order": o, "QE": round(qe, 4), "M2": round(m2_, 4)})
    print(f" {', '.join(o):48} QE {qe:+.4f}  M2 {m2_:+.4f}")
    check(f"Table X row {i+1}, QE", qe, TABX[i][0], 0.002)
    check(f"Table X row {i+1}, M2", m2_, TABX[i][1], 0.002)

# =============================================================================
# 7. SECTION 4.7: THE LAG ORDER
# =============================================================================
print(f"\n{'='*70}\n=== Section 4.7: lag order ===\n{'='*70}")
LAGX = {1: (-0.12, -0.15), 2: (-0.09, -0.14), 3: (-0.06, -0.11), 4: (0.01, -0.02)}
OUT["lags"] = {}
for k in [1, 2, 3, 4]:
    r_ = fit(NAMES, k=k)
    rr = pd.DataFrame(r_.resid)
    lbm = min(float(acorr_ljungbox(rr[c], lags=[4], return_df=True)["lb_pvalue"].iloc[0])
              for c in rr.columns)
    h4, h8 = float(irf(r_, NAMES, "LWALCL", "UNRATE", 4)), float(irf(r_, NAMES, "LWALCL", "UNRATE", 8))
    npar = 6 * k + 4 + 1
    Qk, dfk, pk = multivariate_portmanteau(r_.resid, 4, k_ar_diff=k)
    OUT["lags"][k] = dict(h4=round(h4, 4), h8=round(h8, 4), lb_min=round(lbm, 3),
                          nobs=int(r_.nobs), npar_per_eq=npar,
                          Q4=round(Qk, 1), Q4_df=int(dfk),
                          Q4_p=None if pk is None else round(pk, 4))
    ptxt = "undefined" if pk is None else f"{pk:.4f}"
    print(f" k={k}: h4={h4:+.4f} h8={h8:+.4f}  LB min p={lbm:.3f}  "
          f"nobs={r_.nobs} params/eq={npar}  Q(4)={Qk:.1f} df={dfk} p={ptxt}")
    check(f"k={k} QE->U h4", h4, LAGX[k][0], 0.006)
    check(f"k={k} QE->U h8", h8, LAGX[k][1], 0.006)
check("k=2 nobs", OUT["lags"][2]["nobs"], 89, 0)
check("k=2 params per equation", OUT["lags"][2]["npar_per_eq"], 17, 0)
check("k=4 nobs", OUT["lags"][4]["nobs"], 87, 0)
check("k=4 params per equation", OUT["lags"][4]["npar_per_eq"], 29, 0)
check("k=4 portmanteau undefined", float(OUT["lags"][4]["Q4_p"] is None), 1.0, 0)

# =============================================================================
# 7b. NEW IN v5: THE COINTEGRATING RANK
#
# Section 4.1 records that the second Johansen trace statistic (70.97) exceeds
# its 5% critical value (69.82) only marginally, and Limitation 3 concedes that
# imposing r = 1 "is a judgement rather than a certainty".  That judgement was
# never tested.  Because r <= 1 is the one null the trace test rejects only
# marginally, a referee will ask whether the central result survives r = 2.
# It does: the eight-quarter response of unemployment to a balance-sheet shock
# is essentially unchanged, and it stays negative at r = 3.
# =============================================================================
print(f"\n{'='*70}\n=== Section 4.x (new): cointegrating rank ===\n{'='*70}")
RANKX = {1: -0.136, 2: -0.135, 3: -0.121}
OUT["rank"] = {}
for _r in [1, 2, 3]:
    _m = VECM(df[NAMES], k_ar_diff=K_AR_DIFF, coint_rank=_r,
              deterministic="ci", exog=EX).fit()
    _o = _m.irf(H).orth_irfs
    _qe4 = float(_o[4, NAMES.index("UNRATE"), NAMES.index("LWALCL")])
    _qe8 = float(_o[8, NAMES.index("UNRATE"), NAMES.index("LWALCL")])
    _m28 = float(_o[8, NAMES.index("UNRATE"), NAMES.index("LM2")])
    _rr = pd.DataFrame(_m.resid)
    _lbm = min(float(acorr_ljungbox(_rr[c], lags=[4], return_df=True)["lb_pvalue"].iloc[0])
               for c in _rr.columns)
    OUT["rank"][_r] = dict(qe_h4=round(_qe4, 4), qe_h8=round(_qe8, 4),
                           m2_h8=round(_m28, 4), lb_min=round(_lbm, 3))
    print(f" rank={_r}: QE->U h4={_qe4:+.4f} h8={_qe8:+.4f}  M2->U h8={_m28:+.4f}"
          f"  LB min p={_lbm:.3f}")
    # tolerance is wide: these are new numbers, checked to guard against
    # regressions rather than against the published manuscript
    check(f"rank={_r} QE->U h8", _qe8, RANKX[_r], 0.01)
    check(f"rank={_r} QE->U h8 is negative", float(_qe8 < 0), 1.0, 0)

# =============================================================================
# 8. BOOTSTRAP  (Table VII)
# =============================================================================
print(f"\n{'='*70}\n=== Bootstrap ({NBOOT} replications, seed {SEED}) ===\n{'='*70}")
Y = df[NAMES].values
Kd, p_lev = len(NAMES), K_AR_DIFF + 1
A = [np.asarray(a) for a in res_main.var_rep]          # VAR(K_AR_DIFF+1) matrices
const = (res_main.alpha @ res_main.det_coef_coint).ravel()   # restricted constant
# `exog_coefs` is not exposed by every statsmodels release; fall back to
# recovering the dummy coefficients directly (the VECM is linear in them once
# beta is fixed, so a single OLS on the model regressors reproduces them).
Phi = getattr(res_main, "exog_coefs", None)
if Phi is None:
    _dY = np.diff(Y, axis=0)
    _dY0 = _dY[K_AR_DIFF:]
    _n = len(_dY0)
    _ect = (np.hstack([Y[K_AR_DIFF:-1], np.ones((_n, 1))])
            @ np.vstack([res_main.beta, res_main.det_coef_coint]))
    _Z = np.hstack([_ect]
                   + [_dY[K_AR_DIFF - i:len(_dY) - i] for i in range(1, K_AR_DIFF + 1)]
                   + [EX[K_AR_DIFF + 1:]])
    _b, *_ = np.linalg.lstsq(_Z, _dY0, rcond=None)
    Phi = _b[-EX.shape[1]:, :].T
    print(" (exog_coefs unavailable; dummy coefficients recovered by OLS)")
Phi = np.asarray(Phi)
if Phi.shape != (Kd, EX.shape[1]):
    Phi = Phi.T if Phi.T.shape == (Kd, EX.shape[1]) else Phi.reshape(Kd, EX.shape[1])
E = np.asarray(res_main.resid)
Ec = E - E.mean(0)
T = len(Y)


def simulate(eps):
    ys = np.zeros_like(Y)
    ys[:p_lev] = Y[:p_lev]
    for t in range(p_lev, T):
        v = const.copy()
        for i in range(p_lev):
            v = v + A[i] @ ys[t - 1 - i]
        ys[t] = v + Phi @ EX[t] + eps[t - p_lev]
    return ys


def run_bootstrap(wild):
    rng = np.random.default_rng(SEED)
    out = np.full((NBOOT, H + 1, Kd, Kd), np.nan)
    for b in range(NBOOT):
        if wild:            # Rademacher multiplier, one per period, all equations
            eps = Ec * rng.choice([-1.0, 1.0], size=(len(Ec), 1))
        else:
            eps = Ec[rng.integers(0, len(Ec), len(Ec))]
        eps = eps[:T - p_lev]
        try:
            ys = pd.DataFrame(simulate(eps), index=df.index, columns=NAMES)
            rb = VECM(ys, k_ar_diff=K_AR_DIFF, coint_rank=RANK,
                      deterministic="ci", exog=EX).fit()
            out[b] = rb.irf(H).orth_irfs
        except Exception:
            continue
    return out


PAIRS = [("LWALCL", "UNRATE", [4, 8]), ("LM2", "UNRATE", [4, 8]),
         ("LWALCL", "LGDP", [8]), ("LM2", "LGDP", [8]),
         ("SSR", "LGDP", [8]), ("SSR", "UNRATE", [8])]
orth_main = res_main.irf(H).orth_irfs
BANDS = {}
for wild in [False, True]:
    o = run_bootstrap(wild)
    lab = "wild" if wild else "iid "
    key = "wild" if wild else "iid"
    BANDS[key] = o
    OUT[f"boot_{key}"] = {}
    for s_, r_, hs in PAIRS:
        for h in hs:
            d_ = o[:, h, NAMES.index(r_), NAMES.index(s_)]
            d_ = d_[~np.isnan(d_)]
            q = np.percentile(d_, [2.5, 5, 95, 97.5])
            pt = orth_main[h, NAMES.index(r_), NAMES.index(s_)]
            OUT[f"boot_{key}"][f"{s_}->{r_}@{h}"] = dict(
                point=round(float(pt), 4), ci90=[round(q[1], 4), round(q[2], 4)],
                ci95=[round(q[0], 4), round(q[3], 4)], reps=int(len(d_)))
            print(f" {lab} {s_:6}->{r_:6} h{h}: pt={pt:+.4f} "
                  f"90%[{q[1]:+.4f},{q[2]:+.4f}] 95%[{q[0]:+.4f},{q[3]:+.4f}]")
    # last horizon at which the 90% band excludes zero -- now for BOTH bootstraps
    for s_ in ["LWALCL", "LM2"]:
        ub = [np.nanpercentile(o[:, h, NAMES.index("UNRATE"), NAMES.index(s_)], 95)
              for h in range(H + 1)]
        hh = [h for h in range(1, H + 1) if ub[h] < 0]
        run = 0
        for h in range(1, H + 1):
            if ub[h] < 0:
                run = h
            else:
                break
        print(f"   {lab} {s_}: 90% band excludes zero from h=1 through h="
              f"{run if run else 'none'} (any horizon: {max(hh) if hh else 'none'})")
        OUT[f"boot_{key}"][f"{s_}_excl_zero_through"] = run

# heteroskedasticity of the balance-sheet residual (Section 4.3)
rw = pd.Series(E[:, NAMES.index("LWALCL")], index=df.index[p_lev:])
sd_pre, sd_post = rw[:"2008-07-01"].std(), rw["2008-10-01":].std()
print("\n balance-sheet residual sd: pre-2008Q4 %.4f, post %.4f, ratio %.2f" %
      (sd_pre, sd_post, sd_post / sd_pre))
check("balance-sheet residual sd ratio", sd_post / sd_pre, 2.09, 0.15)

b = OUT["boot_iid"]
check("Table VII: QE->U h4 point", b["LWALCL->UNRATE@4"]["point"], -0.091, 0.002)
check("Table VII: QE->U h8 point", b["LWALCL->UNRATE@8"]["point"], -0.136, 0.002)
check("Table VII: M2->U h8 point", b["LM2->UNRATE@8"]["point"], -0.125, 0.002)
check("Table VII: QE->U h8 90% upper < 0", b["LWALCL->UNRATE@8"]["ci90"][1], -0.010, 0.010)
check("Table VII: QE->U h8 95% upper > 0", b["LWALCL->UNRATE@8"]["ci95"][1], 0.007, 0.010)
check("Section 4.3: M2 band excludes zero through h=20",
      b["LM2_excl_zero_through"], 20, 0)
# Section 4.3 quotes the 95% band for the shadow-rate -> output response;
# tolerance is loose because these are bootstrap tail quantiles
check("Table VII: SSR->GDP h8 point", b["SSR->LGDP@8"]["point"], 0.0047, 0.0005)
check("Section 4.3: SSR->GDP h8 95% lower > 0", b["SSR->LGDP@8"]["ci95"][0], 0.0010, 0.0010)
check("Section 4.3: SSR->GDP h8 95% upper", b["SSR->LGDP@8"]["ci95"][1], 0.0070, 0.0010)
check("Table VII: SSR->U h8 point", b["SSR->UNRATE@8"]["point"], 0.008, 0.002)
check("Table VII: QE->GDP h8 point", b["LWALCL->LGDP@8"]["point"], -0.0001, 0.0005)
check("Table VII: M2->GDP h8 point", b["LM2->LGDP@8"]["point"], 0.0005, 0.0005)
check("Section 4.3: QE band excludes zero through h>=14",
      1.0 if b["LWALCL_excl_zero_through"] >= 14 else 0.0, 1.0, 0)
w = OUT["boot_wild"]
check("wild: QE->U h8 90% band", w["LWALCL->UNRATE@8"]["ci90"][1], 0.010, 0.010)
check("wild: M2->U h8 90% upper < 0", w["LM2->UNRATE@8"]["ci90"][1], -0.013, 0.010)

# =============================================================================
# 9. FIGURES 1-4  (absent from v2)
# =============================================================================
GREY = dict(color="0.85", zorder=0)
CQ = [(pd.Timestamp(q), pd.Timestamp(q) + pd.offsets.QuarterBegin(startingMonth=1))
      for q in CRISIS]


def shade(ax):
    for a, b_ in CQ:
        ax.axvspan(a, b_, **GREY)


# Fig. 1 -- nominal versus shadow rate
fig, ax = plt.subplots(figsize=(9, 4))
shade(ax)
ax.plot(df.index, df.FFR, lw=1.4, color="0.35", label="Effective federal funds rate")
ax.plot(df.index, df.SSR, lw=1.6, color="black", label="Wu-Xia shadow rate (spliced)")
ax.fill_between(df.index, df.SSR, df.FFR, where=df.SSR < df.FFR,
                color="0.6", alpha=.35, label="Stimulus visible only to the shadow rate")
ax.axhline(0, color="black", lw=.6)
ax.set_ylabel("Per cent")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("fig1_shadow_vs_nominal.png", dpi=300)
plt.close(fig)

# Fig. 2 -- growth, unemployment, inflation
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
g = df.LGDP.diff() * 100
shade(axes[0])
axes[0].bar(df.index, g, width=70, color="0.35")
axes[0].axhline(0, color="black", lw=.6)
axes[0].set_title("Real GDP growth, q/q (%)", fontsize=9)
shade(axes[1])
axes[1].plot(df.index, df.UNRATE, color="black", lw=1.4, label="Unemployment (%)")
axes[1].plot(df.index, df.INF, color="0.5", lw=1.4, label="Core PCE inflation (%)")
axes[1].legend(frameon=False, fontsize=8)
axes[1].set_title("Unemployment and inflation", fontsize=9)
fig.tight_layout()
fig.savefig("fig2_growth_unemployment_inflation.png", dpi=300)
plt.close(fig)

# Fig. 3 -- balance sheet and M2
fig, ax = plt.subplots(figsize=(9, 3.8))
shade(ax)
ax.plot(df.index, df.WALCL / 1e6, color="black", lw=1.5, label="Fed total assets (tn $)")
ax2 = ax.twinx()
ax2.plot(df.index, df.M2SL / 1e3, color="0.5", lw=1.5, label="M2 (tn $)")
ax.set_ylabel("Fed total assets (tn $)")
ax2.set_ylabel("M2 (tn $)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("fig3_balance_sheet_m2.png", dpi=300)
plt.close(fig)

# Fig. 4 -- orthogonalised IRFs with bootstrap bands
PANELS = [("SSR", "LGDP"), ("SSR", "UNRATE"), ("LM2", "LGDP"),
          ("LM2", "UNRATE"), ("LWALCL", "LGDP"), ("LWALCL", "UNRATE")]
LBL = {"SSR": "Shadow rate", "LM2": "M2", "LWALCL": "Balance sheet",
       "LGDP": "Real GDP (log pts)", "UNRATE": "Unemployment (pp)"}
o = BANDS["iid"]
fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=True)
for ax, (s_, r_) in zip(axes.ravel(), PANELS):
    si, ri = NAMES.index(s_), NAMES.index(r_)
    pt = orth_main[:, ri, si]
    lo95, lo90, hi90, hi95 = [np.nanpercentile(o[:, :, ri, si], q, axis=0)
                             for q in (2.5, 5, 95, 97.5)]
    hs = np.arange(H + 1)
    ax.fill_between(hs, lo95, hi95, color="0.85")
    ax.fill_between(hs, lo90, hi90, color="0.65")
    ax.plot(hs, pt, color="black", lw=1.6)
    ax.axhline(0, color="black", lw=.6)
    ax.set_title(f"{LBL[s_]} shock -> {LBL[r_]}", fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("Quarters")
fig.tight_layout()
fig.savefig("fig4_impulse_responses.png", dpi=300)
plt.close(fig)
print("\nFigures written: fig1..fig4 (.png, 300 dpi)")

# =============================================================================
# 10. VERIFICATION AGAINST THE MANUSCRIPT
# =============================================================================
print(f"\n{'='*70}\n=== Checks against the numbers quoted in the paper ===\n{'='*70}")
bad = [c for c in CHECKS if not c[4]]
for label, got, want, tol, ok in CHECKS:
    g = "None" if got is None else f"{float(got):.4f}"
    print(f" [{'PASS' if ok else 'FAIL'}] {label:44} got {g:>10}  want {want:>9} (+-{tol})")
print(f"\n {len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed.")
if bad:
    print(" FAILED:", ", ".join(c[0] for c in bad))
OUT["checks"] = [{"label": c[0], "got": None if c[1] is None else float(c[1]),
                  "expected": c[2], "tol": c[3], "pass": c[4]} for c in CHECKS]

json.dump(OUT, open("results_final.json", "w"), indent=1, default=str)
print("\nSAVED results_final.json, data_final.csv, orth_SSR.npy, orth_FFR.npy, fig1-4.png")

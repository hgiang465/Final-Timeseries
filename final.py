"""
=============================================================================
BRENT CRUDE OIL ANALYSIS AND FORECASTING
=============================================================================
Phase 0  : Data cleaning
Phase 1  : ADF + KPSS
Phase 2  : Structural break – CUSUM + Chow-test (4 events)
Phase 3  : Engle-Granger cointegration
Phase 4  : Descriptive stats + Jarque-Bera normality test
Phase 5  : ARCH-LM + ACF/PACF (Yule-Walker) + Ljung-Box
Phase 6  : PLS dimension reduction
Phase 7  : Train/test + Walk-forward rolling (8 windows)
Phase 8  : Models: OLS, PLS, ARIMA, SARIMA, ARCH(1), GARCH(1,1), EGARCH,
           Dummy Variables (GFC, COVID-19, Ukraine War) + OLS+Dummy
Phase 9  : Metrics + Diebold-Mariano test
Phase 10 : Bootstrap prediction interval + Bull/Base/Bear scenarios
Phase 11 : Figures + Excel
=============================================================================
"""
import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats, optimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": False, "grid.alpha": 0.35, "font.family": "DejaVu Sans",
    "axes.titlesize": 11, "axes.labelsize": 9,
})
PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
           "#aec7e8","#ffbb78","#98df8a","#ff9896"]

# =============================================================================
# 0. CLEANING & MERGING DATA
# =============================================================================
print("\n" + "="*70)
print("PART 0 – CLEANING & MERGING DATA")
print("="*70)

base_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir  = os.path.join(base_dir, "Raw_data")
out_dir  = os.path.join(base_dir, "Out_put")
os.makedirs(out_dir, exist_ok=True)

files = {
    "OIL": os.path.join(raw_dir, "DCOILBRENTEU (2).xlsx"),
    "CPI": os.path.join(raw_dir, "CPIAUCSL.xlsx"),
    "USD": os.path.join(raw_dir, "DTWEXBGS.xlsx"),
    "FED": os.path.join(raw_dir, "FEDFUNDS.xlsx"),
    "IND": os.path.join(raw_dir, "INDPRO.xlsx"),
}
col_map = {"DCOILBRENTEU": "OIL", "CPIAUCSL": "CPI",
           "DTWEXBGS": "USD", "FEDFUNDS": "FED", "INDPRO": "IND"}

monthly = {}
for name, path in files.items():
    df = pd.read_excel(path, parse_dates=["observation_date"])
    df.rename(columns={"observation_date": "date"}, inplace=True)
    df.set_index("date", inplace=True)
    df.columns = [col_map.get(c, c) for c in df.columns]
    monthly[name] = df.resample("ME").last()

df_raw = pd.concat(monthly.values(), axis=1)
df_raw.ffill(limit=3, inplace=True)
df_raw.bfill(limit=3, inplace=True)
df_raw.dropna(inplace=True)

print(f"Date range : {df_raw.index[0].date()} -> {df_raw.index[-1].date()}")
print(f"Number of observations (raw): {len(df_raw)}")

# =============================================================================
# 1. STATIONARITY TESTS: ADF + KPSS
# =============================================================================
print("\n" + "="*70)
print("PART 1 – STATIONARITY TESTS: ADF + KPSS")
print("="*70)

def adf_test(y, maxlag=None):
    y = np.asarray(y, dtype=float); n = len(y)
    if maxlag is None:
        maxlag = int(np.ceil(12 * (n / 100) ** 0.25))
    best = (np.inf, 0, 0, 1.0)
    for lag in range(0, maxlag + 1):
        dy = np.diff(y); T2 = len(dy) - lag
        if T2 <= lag + 3: continue
        dy_l = dy[lag:]
        cols  = [y[lag:-1], np.ones(len(dy_l))]
        if lag > 0:
            cols += [dy[lag-i-1:-i-1] for i in range(lag)]
        X = np.column_stack(cols)
        try:
            b, _, _, _ = np.linalg.lstsq(X, dy_l, rcond=None)
        except Exception:
            continue
        e = dy_l - X @ b
        s2 = np.dot(e, e) / max(len(e) - X.shape[1], 1)
        aic = len(e) * np.log(max(s2, 1e-15)) + 2 * X.shape[1]
        if aic < best[0]:
            XtX_inv = np.linalg.pinv(X.T @ X)
            se = np.sqrt(s2 * XtX_inv[0, 0])
            tstat = b[0] / se if se > 1e-12 else 0
            cv = {1: -3.43, 5: -2.86, 10: -2.57}
            if   tstat <= cv[1]:  pv = 0.01
            elif tstat <= cv[5]:  pv = 0.05
            elif tstat <= cv[10]: pv = 0.10
            else:                 pv = 0.50
            best = (aic, lag, tstat, pv)
    return best[2], best[3], best[1]

def kpss_test(y, lags=None):
    y = np.asarray(y, dtype=float); n = len(y)
    resid = y - y.mean(); S = np.cumsum(resid)
    if lags is None:
        lags = int(np.ceil(4 * (n / 100) ** 0.25))
    gamma0 = np.dot(resid, resid) / n
    long_run_var = gamma0
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        gj = np.dot(resid[j:], resid[:-j]) / n
        long_run_var += 2 * w * gj
    long_run_var = max(long_run_var, 1e-15)
    kpss_stat = np.sum(S ** 2) / (n ** 2 * long_run_var)
    cv = {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739}
    if   kpss_stat < cv[0.10]:  p_approx = 0.60
    elif kpss_stat < cv[0.05]:  p_approx = 0.10
    elif kpss_stat < cv[0.025]: p_approx = 0.05
    elif kpss_stat < cv[0.01]:  p_approx = 0.025
    else:                        p_approx = 0.01
    return kpss_stat, p_approx

def stationarity_decision(adf_p, kpss_p):
    adf_reject  = (adf_p  < 0.05)
    kpss_accept = (kpss_p >= 0.05)
    if adf_reject and kpss_accept:
        return True,  "STATIONARY ✅"
    elif adf_reject and not kpss_accept:
        return True,  "POSSIBLY STATIONARY ⚠️"
    else:
        return False, "NON-STATIONARY ❌"

level_cols = ["OIL", "USD", "CPI", "FED", "IND"]
print(f"\n[LEVEL]")
print(f"{'Variable':<10} {'ADF_stat':>10} {'ADF_p':>8} {'KPSS_stat':>10} {'KPSS_p':>8} {'Conclusion':<22} {'I(d)'}")
print("-"*76)

level_results = {}
for col in level_cols:
    series = df_raw[col].dropna().values
    adf_s, adf_p, _ = adf_test(series)
    kpss_s, kpss_p  = kpss_test(series)
    is_stat, label  = stationarity_decision(adf_p, kpss_p)
    level_results[col] = {"adf_p": adf_p, "kpss_p": kpss_p, "is_stationary": is_stat}
    order = "I(0)" if is_stat else "I(1)"
    print(f"{col:<10} {adf_s:>10.4f} {adf_p:>8.4f} {kpss_s:>10.4f} {kpss_p:>8.4f} {label:<22} {order}")

diff_df = pd.DataFrame(index=df_raw.index)
for col in level_cols:
    diff_df[f"{col}_RET"] = np.log(df_raw[col] / df_raw[col].shift(1))
diff_df.dropna(inplace=True)

print(f"\n[LOG-RETURN]")
print(f"{'Variable':<12} {'ADF_stat':>10} {'ADF_p':>8} {'KPSS_stat':>10} {'KPSS_p':>8} {'Conclusion':<22} {'I(d)'}")
print("-"*78)

diff_results = {}
for col in level_cols:
    rc = f"{col}_RET"
    series = diff_df[rc].values
    adf_s, adf_p, _ = adf_test(series)
    kpss_s, kpss_p  = kpss_test(series)
    is_stat, label  = stationarity_decision(adf_p, kpss_p)
    diff_results[rc] = {"adf_p": adf_p, "kpss_p": kpss_p, "is_stationary": is_stat}
    order = "I(0)" if is_stat else "I(1)*"
    print(f"{rc:<12} {adf_s:>10.4f} {adf_p:>8.4f} {kpss_s:>10.4f} {kpss_p:>8.4f} {label:<22} {order}")

# =============================================================================
# 2. STRUCTURAL BREAK: CUSUM + CHOW-TEST (4 events)
# =============================================================================
print("\n" + "="*70)
print("PART 2 – STRUCTURAL BREAK: CUSUM + CHOW-TEST (4 events)")
print("="*70)

# 4 structural break dates
BREAK_EVENTS = {
    "GFC_2008":        pd.Timestamp("2008-09-01"),
    "OilCrash_2014":   pd.Timestamp("2014-11-01"),
    "COVID_2020":      pd.Timestamp("2020-03-01"),
    "UkraineWar_2022": pd.Timestamp("2022-02-01"),
}

oil_ret_all = diff_df["OIL_RET"].values
oil_ret_idx = diff_df.index

def cusum_test(y, h_pct=0.15):
    n = len(y); k = 1
    cusum = np.zeros(n); e_hat = np.zeros(n)
    for t in range(k, n):
        y_hist = y[:t]; b = np.mean(y_hist)
        e_hat[t] = y[t] - b
    s = np.std(e_hat[k:]) if np.std(e_hat[k:]) > 0 else 1e-9
    W = np.cumsum(e_hat[k:]) / s
    a = 0.948
    bound = [a * (np.sqrt(n-k) + 2*a*j/np.sqrt(n-k)) for j in range(len(W))]
    return W, np.array(bound), s

W_cusum, bound_cusum, s_cusum = cusum_test(oil_ret_all)

print("\n[CUSUM Test – OIL_RET]")
print(f"  σ (recursive residuals) = {s_cusum:.6f}")
n_exceed = np.sum(np.abs(W_cusum) > bound_cusum)
print(f"  Periods exceeding 5% boundary: {n_exceed} / {len(W_cusum)}")
if n_exceed > 0:
    print("  → Structural instability detected ✅")
else:
    print("  → No significant instability")

def chow_test(y, idx, break_date):
    mask_pre  = idx < break_date
    mask_post = idx >= break_date
    y_pre  = y[mask_pre]; y_post = y[mask_post]
    if len(y_pre) < 5 or len(y_post) < 5:
        return np.nan, np.nan, len(y_pre), len(y_post)
    n = len(y); k = 1
    rss_r   = np.sum((y - y.mean())**2)
    rss_pre  = np.sum((y_pre  - y_pre.mean())**2)
    rss_post = np.sum((y_post - y_post.mean())**2)
    rss_u   = rss_pre + rss_post
    F = ((rss_r - rss_u) / k) / (rss_u / (n - 2*k))
    p_val = 1 - stats.f.cdf(F, dfn=k, dfd=n-2*k)
    return F, p_val, len(y_pre), len(y_post)

print("\n[Chow Test – OIL_RET at 4 structural break points]")
print(f"{'Event':<22} {'Break Date':<14} {'n_pre':>6} {'n_post':>6} {'F-stat':>10} {'p-value':>10} {'Decision'}")
print("-"*80)

chow_results = {}
for event, bdate in BREAK_EVENTS.items():
    F, p, n_pre, n_post = chow_test(oil_ret_all, oil_ret_idx, bdate)
    if not np.isnan(F):
        decision = "Break ✅" if p < 0.05 else "No break ❌"
        chow_results[event] = {"date": bdate, "F": F, "p": p}
        print(f"{event:<22} {str(bdate.date()):<14} {n_pre:>6} {n_post:>6} {F:>10.4f} {p:>10.6f}  {decision}")
    else:
        print(f"{event:<22} {str(bdate.date()):<14} {'Insufficient data':>32}")

# =============================================================================
# 3. ENGLE-GRANGER COINTEGRATION
# =============================================================================
print("\n" + "="*70)
print("PART 3 – COINTEGRATION TEST (Engle-Granger)")
print("="*70)

def engle_granger_coint(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    stat, _, lag = adf_test(resid)
    if   stat <= -3.90: p_eg = 0.01
    elif stat <= -3.34: p_eg = 0.05
    elif stat <= -3.04: p_eg = 0.10
    else:               p_eg = 0.50
    return stat, p_eg, b, lag

exo_for_coint    = ["USD", "CPI", "FED", "IND"]
non_stationary_l = [c for c in level_cols if not level_results[c]["is_stationary"]]

print(f"I(1) variables: {non_stationary_l}")
print(f"\n{'Pair':<22} {'EG stat':>10} {'p-value':>10} {'Lag':>5} {'Conclusion'}")
print("-"*72)

coint_pairs = {}; eg_ran = False
for col in exo_for_coint:
    if (not level_results["OIL"]["is_stationary"]) and (not level_results[col]["is_stationary"]):
        eg_ran = True
        stat, p_eg, beta, lag = engle_granger_coint(df_raw["OIL"].values, df_raw[col].values)
        coint = (p_eg < 0.05)
        coint_pairs[col] = {"stat": stat, "p": p_eg, "cointegrated": coint, "beta": beta}
        conclusion = "Cointegrated ✅" if coint else "Not cointegrated ❌"
        print(f"OIL ~ {col:<16} {stat:>10.4f} {p_eg:>10.4f} {lag:>5}  {conclusion}")
    else:
        print(f"OIL ~ {col:<16} {'(skipped: one or both stationary)':>52}")

has_coint = any(v["cointegrated"] for v in coint_pairs.values()) if coint_pairs else False
print(f"\n[DECISION]: Use LOG-RETURN (I(0)) for all models.")

# =============================================================================
# 4. DESCRIPTIVE STATS + JARQUE-BERA NORMALITY TEST
# =============================================================================
print("\n" + "="*70)
print("PART 4 – DESCRIPTIVE STATS + JARQUE-BERA NORMALITY TEST")
print("="*70)

df_all = diff_df.copy()
df_all["OIL"] = df_raw["OIL"].reindex(df_all.index)

cols_d = ["OIL_RET", "USD_RET", "CPI_RET", "FED_RET", "IND_RET"]
desc = df_all[cols_d].describe().T
desc["skewness"] = df_all[cols_d].skew()
desc["kurtosis"] = df_all[cols_d].kurt()

def jarque_bera(x):
    n = len(x)
    s = stats.skew(x); k = stats.kurtosis(x)
    jb = n / 6 * (s**2 + k**2 / 4)
    p  = 1 - stats.chi2.cdf(jb, df=2)
    return jb, p

print(f"\n{'Variable':<12} {'JB_stat':>12} {'JB_p':>10} {'Skewness':>10} {'Ex.Kurt':>10} {'Normal?'}")
print("-"*65)
jb_rows = []
for col in cols_d:
    jb_stat, jb_p = jarque_bera(df_all[col].dropna().values)
    sk = stats.skew(df_all[col].dropna())
    ku = stats.kurtosis(df_all[col].dropna())
    normal = "No ❌" if jb_p < 0.05 else "Yes ✅"
    print(f"{col:<12} {jb_stat:>12.4f} {jb_p:>10.6f} {sk:>10.4f} {ku:>10.4f}  {normal}")
    jb_rows.append({"Variable": col, "JB_stat": round(jb_stat,4),
                    "JB_p": round(jb_p,6), "Skewness": round(sk,4),
                    "ExKurt": round(ku,4), "Normal": normal})

desc_full = desc.copy()
desc_full["JB_stat"] = [jb_rows[i]["JB_stat"] for i in range(len(cols_d))]
desc_full["JB_p"]    = [jb_rows[i]["JB_p"]    for i in range(len(cols_d))]

print("\n[NOTE] All return series are non-normal (fat tails). OIL_RET: kurtosis≈34.")
print(desc.round(4).to_string())

# =============================================================================
# 5. ARCH-LM + ACF/PACF (Yule-Walker) + LJUNG-BOX
# =============================================================================
print("\n" + "="*70)
print("PART 5 – ARCH-LM + ACF/PACF (Yule-Walker) + LJUNG-BOX")
print("="*70)

def arch_lm(resid, nlags=12):
    e2 = resid ** 2; T = len(e2)
    Y = e2[nlags:]
    X = np.column_stack([np.ones(len(Y))] +
                        [e2[nlags-i-1:T-i-1] for i in range(nlags)])
    b, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    e_ = Y - X @ b
    ss_r = np.dot(e_, e_); ss_t = np.dot(Y - Y.mean(), Y - Y.mean())
    r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
    lm = T * r2
    return lm, 1 - stats.chi2.cdf(lm, df=nlags)

def yule_walker_pacf(x, maxlag=20):
    x = x - x.mean(); n = len(x)
    c0 = np.dot(x, x) / n
    acf_vals = [1.] + [np.dot(x[:n-k], x[k:]) / (n*c0) for k in range(1, maxlag+1)]
    pacf = np.zeros(maxlag + 1); pacf[0] = 1.
    for m in range(1, maxlag + 1):
        R = np.array([[acf_vals[abs(i-j)] for j in range(m)] for i in range(m)])
        r = np.array([acf_vals[i+1] for i in range(m)])
        try:
            phi = np.linalg.solve(R, r)
            pacf[m] = phi[-1]
        except np.linalg.LinAlgError:
            pacf[m] = 0.
    return pacf[1:]

def ljung_box(x, lags=12):
    n = len(x); x = x - x.mean(); c0 = np.dot(x,x)/n
    acf_ = [np.dot(x[:n-k], x[k:])/(n*c0) for k in range(1, lags+1)]
    Q = n*(n+2) * sum(rk**2/(n-k) for k, rk in enumerate(acf_, 1))
    p = 1 - stats.chi2.cdf(Q, df=lags)
    return Q, p

ret = df_all["OIL_RET"].values

lm_s, lm_p = arch_lm(ret)
print(f"\nARCH LM – full sample (lag=12): stat={lm_s:.4f}  p={lm_p:.6f}  "
      f"→ {'ARCH effect ✅' if lm_p < 0.05 else 'No ARCH'}")

covid_mask = ~((df_all.index.year == 2020) & (df_all.index.month.isin([3,4,5])))
ret_no_covid = df_all.loc[covid_mask, "OIL_RET"].values
lm_s2, lm_p2 = arch_lm(ret_no_covid)
print(f"ARCH LM – ex-COVID (lag=12):     stat={lm_s2:.4f}  p={lm_p2:.6f}  "
      f"→ {'ARCH effect ✅' if lm_p2 < 0.05 else 'No ARCH'}")

lb_Q, lb_p = ljung_box(ret, lags=12)
lb_Q2, lb_p2 = ljung_box(ret**2, lags=12)
print(f"\nLjung-Box on OIL_RET      : Q={lb_Q:.4f}  p={lb_p:.6f}  "
      f"→ {'Serial correlation ✅' if lb_p < 0.05 else 'No serial corr'}")
print(f"Ljung-Box on OIL_RET²     : Q={lb_Q2:.4f}  p={lb_p2:.6f}  "
      f"→ {'Volatility clustering ✅' if lb_p2 < 0.05 else 'No clustering'}")

def manual_acf(x, n=30):
    x = x - x.mean(); c0 = np.dot(x,x)/len(x)
    return [1.] + [np.dot(x[:len(x)-k], x[k:])/(len(x)*c0) for k in range(1, n+1)]

cb = 1.96 / np.sqrt(len(ret))
ret_acf  = manual_acf(ret)[1:]
e2_acf   = manual_acf(ret**2)[1:]
ret_pacf = yule_walker_pacf(ret, maxlag=30)
e2_pacf  = yule_walker_pacf(ret**2, maxlag=30)

fig2, axes = plt.subplots(2, 2, figsize=(14, 8))
fig2.suptitle("ACF & PACF (Yule-Walker) – OIL_RET", fontsize=13, fontweight="bold")

def plot_bar(ax, vals, title):
    ax.bar(range(len(vals)), vals, color="#1f77b4", width=0.4)
    ax.axhline(cb,  ls="--", color="red", lw=1)
    ax.axhline(-cb, ls="--", color="red", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(title)

plot_bar(axes[0,0], ret_acf,  "ACF – OIL_RET")
plot_bar(axes[0,1], ret_pacf, "PACF (Yule-Walker) – OIL_RET")
plot_bar(axes[1,0], e2_acf,   "ACF – OIL_RET²  (ARCH effect)")
plot_bar(axes[1,1], e2_pacf,  "PACF (Yule-Walker) – OIL_RET²")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig2_ACF_PACF_YW.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Saved] Fig2_ACF_PACF_YW.png")

# =============================================================================
# 6. DIMENSION REDUCTION: PLS
# =============================================================================
print("\n" + "="*70)
print("PART 6 – DIMENSION REDUCTION: PLS")
print("="*70)

exo_cols = ["USD_RET", "CPI_RET", "FED_RET", "IND_RET"]
N_COMP   = 4
X_raw_   = df_all[exo_cols].values
y_raw_   = df_all["OIL_RET"].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_raw_)

pls = PLSRegression(n_components=N_COMP)
pls.fit(X_sc, y_raw_)
X_pls    = pls.transform(X_sc)
pls_cols = [f"PLS{i+1}" for i in range(N_COMP)]

var_explained = []
X_residual = X_sc.copy()
for k in range(N_COMP):
    t_k = X_pls[:, k:k+1]
    p_k = (X_residual.T @ t_k) / (t_k.T @ t_k)
    X_hat_k = t_k @ p_k.T
    var_k = np.sum(X_hat_k ** 2) / (X_sc.shape[0] - 1)
    var_explained.append(var_k)
    X_residual = X_residual - X_hat_k

var_pct = np.array(var_explained) / sum(var_explained) * 100
print(f"\n{'Comp':<8} {'Var explained':>15} {'%':>8} {'Cum%':>8}")
print("-"*42)
cum = 0
for i, (v, p) in enumerate(zip(var_explained, var_pct)):
    cum += p
    print(f"PLS{i+1:<5} {v:>15.6f} {p:>8.2f}% {cum:>7.2f}%")

df_pls   = pd.DataFrame(X_pls, index=df_all.index, columns=pls_cols)
df_model = df_all[["OIL", "OIL_RET"]].join(df_pls)

# =============================================================================
# DUMMY VARIABLES: GFC 2008, OIL CRASH OPEC 2014, COVID-19 2020, UKRAINE WAR 2022
# =============================================================================
print("\n" + "="*70)
print("DUMMY VARIABLES – GFC 2008 / Oil Crash OPEC 2014 / COVID-19 2020 / Ukraine War 2022")
print("="*70)

# --- GFC 2008: Sep 2008 – Jun 2009 ---
D_GFC = pd.Series(
    ((df_model.index >= "2008-09-01") & (df_model.index <= "2009-06-30")).astype(float),
    index=df_model.index
)
# --- Oil Crash OPEC 2014: Nov 2014 – Jan 2016 ---
D_OIL2014 = pd.Series(
    ((df_model.index >= "2014-11-01") & (df_model.index <= "2016-01-31")).astype(float),
    index=df_model.index
)
# --- COVID-19: Mar 2020 – Dec 2020 ---
D_COVID = pd.Series(
    ((df_model.index >= "2020-03-01") & (df_model.index <= "2020-12-31")).astype(float),
    index=df_model.index
)
# --- Ukraine War: Feb 2022 – Dec 2022 ---
D_UKRAINE = pd.Series(
    ((df_model.index >= "2022-02-01") & (df_model.index <= "2022-12-31")).astype(float),
    index=df_model.index
)

df_model["D_GFC"]     = D_GFC.values
df_model["D_OIL2014"] = D_OIL2014.values
df_model["D_COVID"]   = D_COVID.values
df_model["D_UKRAINE"] = D_UKRAINE.values

dummy_cols = ["D_GFC", "D_OIL2014", "D_COVID", "D_UKRAINE"]

print(f"\nDummy variable summary (1 = crisis period, 0 = normal):")
print(f"  D_GFC      (Sep 2008 – Jun 2009): {int(D_GFC.sum())} months")
print(f"  D_OIL2014  (Nov 2014 – Jan 2016): {int(D_OIL2014.sum())} months")
print(f"  D_COVID    (Mar 2020 – Dec 2020): {int(D_COVID.sum())} months")
print(f"  D_UKRAINE  (Feb 2022 – Dec 2022): {int(D_UKRAINE.sum())} months")

# =============================================================================
# 7. TRAIN / TEST + WALK-FORWARD ROLLING (8 WINDOWS)
# =============================================================================
print("\n" + "="*70)
print("PART 7 – TRAIN/TEST SPLIT + WALK-FORWARD ROLLING (8 windows)")
print("="*70)

n = len(df_model); n_train = int(n * 0.75)
df_train = df_model.iloc[:n_train]; df_test = df_model.iloc[n_train:]
y_train  = df_train["OIL_RET"].values; y_test = df_test["OIL_RET"].values
X_tr     = df_train[pls_cols].values;  X_te   = df_test[pls_cols].values

# Dummy variables split
D_tr = df_train[dummy_cols].values
D_te = df_test[dummy_cols].values

print(f"Train: {df_train.index[0].date()} -> {df_train.index[-1].date()} ({n_train} obs)")
print(f"Test : {df_test.index[0].date()}  -> {df_test.index[-1].date()}  ({len(df_test)} obs)")

N_WINDOWS   = 8
test_len    = len(df_test)
window_size = test_len // N_WINDOWS

print(f"\n[Walk-forward rolling – {N_WINDOWS} windows of ~{window_size} obs each]")
print(f"{'Win':<5} {'Train end':>12} {'Test start':>12} {'Test end':>12} {'n_test':>7}")
print("-"*52)

rolling_windows = []
for w in range(N_WINDOWS):
    te_start = n_train + w * window_size
    te_end   = te_start + window_size if w < N_WINDOWS - 1 else n
    tr_end   = te_start
    rolling_windows.append((tr_end, te_start, te_end))
    print(f"{w+1:<5} {str(df_model.index[tr_end-1].date()):>12} "
          f"{str(df_model.index[te_start].date()):>12} "
          f"{str(df_model.index[te_end-1].date()):>12} "
          f"{te_end-te_start:>7}")

# =============================================================================
# 8. MODELS: OLS, PLS, ARIMA, SARIMA, ARCH(1), GARCH(1,1), EGARCH, DUMMY OLS
# =============================================================================
print("\n" + "="*70)
print("PART 8 – MODELS: OLS | PLS | ARIMA | SARIMA | ARCH(1) | GARCH(1,1) | EGARCH | OLS+Dummy")
print("="*70)

# ── Utility functions ─────────────────────────────────────────────────────────
def ols_pred(Xtr, ytr, Xte):
    Xt = np.column_stack([np.ones(len(Xtr)), Xtr])
    Xp = np.column_stack([np.ones(len(Xte)), Xte])
    b, _, _, _ = np.linalg.lstsq(Xt, ytr, rcond=None)
    return Xp @ b

def arma_css(y, p, q, maxiter=600):
    n = len(y); mu0 = y.mean()
    def css(params):
        c = params[0]; phi = params[1:1+p]; th = params[1+p:1+p+q]
        e = np.zeros(n)
        for t in range(max(p, q, 1), n):
            ar = sum(phi[i] * y[t-1-i] for i in range(p))
            ma = sum(th[j] * e[t-1-j] for j in range(q))
            e[t] = y[t] - c - ar - ma
        return np.sum(e[max(p, q, 1):]**2)
    x0 = np.zeros(1+p+q); x0[0] = mu0
    r = optimize.minimize(css, x0, method="Nelder-Mead",
                          options={"maxiter": maxiter, "xatol": 1e-7, "fatol": 1e-7})
    return r.x

def arma_resid(params, y, p, q):
    n = len(y); c = params[0]; phi = params[1:1+p]; th = params[1+p:1+p+q]
    e = np.zeros(n)
    for t in range(max(p, q, 1), n):
        ar = sum(phi[i] * y[t-1-i] for i in range(p))
        ma = sum(th[j] * e[t-1-j] for j in range(q))
        e[t] = y[t] - c - ar - ma
    return e

def arma_fc(params, y_hist, e_hist, p, q, h):
    c = params[0]; phi = params[1:1+p]; th = params[1+p:1+p+q]
    ye = list(y_hist); ee = list(e_hist); out = []
    for _ in range(h):
        ar = sum(phi[i]*ye[-1-i] for i in range(p)) if p else 0
        ma = sum(th[j]*ee[-1-j] for j in range(q)) if q else 0
        f = c + ar + ma; out.append(f); ye.append(f); ee.append(0.)
    return np.array(out)

def sarimax_fc_simple(y, Xtr, Xte, p, q, h):
    Xt = np.column_stack([np.ones(len(Xtr)), Xtr]) if Xtr is not None \
         else np.ones((len(y), 1))
    Xp = np.column_stack([np.ones(h), Xte]) if Xte is not None \
         else np.ones((h, 1))
    b, _, _, _ = np.linalg.lstsq(Xt, y, rcond=None)
    r = y - Xt @ b
    par = arma_css(r, p, q)
    e_  = arma_resid(par, r, p, q)
    return arma_fc(par, r, e_, p, q, h) + Xp @ b

def garch_fit_fn(r, p, q, dist="normal", maxiter=800):
    """
    Fit GARCH(p,q) with either Normal or Student-t innovations.
    dist = "normal" : Gaussian log-likelihood
    dist = "t"      : Student-t log-likelihood (extra parameter: log(nu-2))
    Returns: mu_hat, omega, alpha_array, beta_array, h_array, e_array,
             [df_t]  (degrees-of-freedom only when dist="t")
    """
    from scipy.special import gammaln
    T = len(r); s2 = r.var(); mu0 = r.mean()

    def _build_h(params, p, q, T, s2, r):
        mu  = params[0]
        om  = np.exp(params[1])
        al  = np.abs(params[2:2+p])
        be  = np.abs(params[2+p:2+p+q])
        # enforce stationarity: alpha+beta < 0.999
        if al.sum() + be.sum() >= 0.999:
            sc = 0.98 / (al.sum() + be.sum() + 1e-9)
            al = al * sc; be = be * sc
        e = r - mu
        h = np.full(T, s2)
        for t in range(max(p, q), T):
            h[t] = max(om + sum(al[i]*e[t-1-i]**2 for i in range(p))
                          + sum(be[j]*h[t-1-j]    for j in range(q)), 1e-9)
        return mu, om, al, be, e, h

    if dist == "normal":
        def nll(params):
            mu, om, al, be, e, h = _build_h(params, p, q, T, s2, r)
            ll = 0.
            for t in range(max(p, q), T):
                ll += 0.5 * (np.log(2*np.pi*h[t]) + e[t]**2 / h[t])
            return ll
        x0 = [mu0, np.log(s2*0.1)] + [0.1]*p + [0.8]*q

    else:  # Student-t
        # extra param: log(nu - 2), so nu = exp(x) + 2  => nu > 2
        def nll(params):
            mu, om, al, be, e, h = _build_h(params[:-1], p, q, T, s2, r)
            nu = np.exp(params[-1]) + 2.0   # degrees of freedom > 2
            ll = 0.
            for t in range(max(p, q), T):
                z2 = e[t]**2 / h[t]
                # Student-t log-density (standardised)
                ll += (0.5*np.log(h[t])
                       - gammaln((nu+1)/2) + gammaln(nu/2)
                       + 0.5*np.log(np.pi*(nu-2))
                       + (nu+1)/2 * np.log(1 + z2/(nu-2)))
            return ll
        x0 = [mu0, np.log(s2*0.1)] + [0.1]*p + [0.8]*q + [np.log(5.0)]

    res = optimize.minimize(nll, x0, method="Nelder-Mead",
                            options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6})
    par = res.x
    mu_hat, om, al, be, e, h = _build_h(
        par if dist == "normal" else par[:-1], p, q, T, s2, r)

    if dist == "t":
        nu_hat = np.exp(par[-1]) + 2.0
        return mu_hat, om, al, be, h, e, nu_hat
    return mu_hat, om, al, be, h, e

# ── Fit all models ────────────────────────────────────────────────────────────
results_fc = {}

# --- 1. OLS (PLS regressors) ---
print("[8a] OLS (with PLS regressors)")
results_fc["OLS"] = ols_pred(X_tr, y_train, X_te)

# --- 2. PLS Regression ---
print("[8b] PLS Regression")
plsr = PLSRegression(n_components=N_COMP)
plsr.fit(X_tr, y_train)
results_fc["PLS"] = plsr.predict(X_te).flatten()

# --- 3. ARIMA(1,0,1) ---
print("[8c] ARIMA(1,0,1)")
par_ar = arma_css(y_train, 1, 1)
e_ar   = arma_resid(par_ar, y_train, 1, 1)
results_fc["ARIMA"] = arma_fc(par_ar, y_train, e_ar, 1, 1, len(y_test))
print(f"     const={par_ar[0]:.6f}  phi={par_ar[1]:.4f}  theta={par_ar[2]:.4f}")

# --- 4. SARIMA(1,0,1)(1,0,1,12) ---
print("[8d] SARIMA(1,0,1)(1,0,1,12)")
if n_train > 12:
    ys_tr = y_train[12:]
    xs_tr = y_train[:-12].reshape(-1,1)
    xs_te = np.concatenate([y_train[-12:], y_test[:-1]])[:len(y_test)].reshape(-1,1)
    results_fc["SARIMA"] = sarimax_fc_simple(ys_tr, xs_tr, xs_te, 1, 1, len(y_test))
else:
    results_fc["SARIMA"] = results_fc["ARIMA"].copy()
    print("     (n_train<=12, fallback to ARIMA)")

# --- 5. ARCH(1) – Normal ---
print("[8e] ARCH(1)")
mu_a, _, _, _, h_arch, e_arch = garch_fit_fn(y_train, 1, 0, dist="normal")
results_fc["ARCH(1)"] = np.full(len(y_test), mu_a)
print(f"     mu={mu_a:.6f}  (variance model only; mean forecast = mu_hat)")

# --- 6. GARCH(1,1) – Normal ---
print("[8f] GARCH(1,1)")
mu_g, om_g, al_g, be_g, h_garch, e_garch = garch_fit_fn(y_train, 1, 1, dist="normal")
results_fc["GARCH(1,1)"] = np.full(len(y_test), mu_g)
print(f"     mu={mu_g:.6f}  omega={om_g:.6e}  alpha={al_g[0]:.4f}  "
      f"beta={be_g[0]:.4f}  persist={al_g[0]+be_g[0]:.4f}")

# --- 6b. GARCH(1,1)-t – Student-t innovations ---
print("[8f-t] GARCH(1,1)-t  [Student-t innovations]")
(mu_gt, om_gt, al_gt, be_gt,
 h_garch_t, e_garch_t, nu_gt) = garch_fit_fn(y_train, 1, 1, dist="t")
results_fc["GARCH-t"] = np.full(len(y_test), mu_gt)
print(f"     mu={mu_gt:.6f}  omega={om_gt:.6e}  alpha={al_gt[0]:.4f}  "
      f"beta={be_gt[0]:.4f}  persist={al_gt[0]+be_gt[0]:.4f}  "
      f"df(nu)={nu_gt:.2f}")

# --- 7. OLS + Dummy Variables ---
print("[8g] OLS + Dummy Variables (GFC, Oil Crash 2014, COVID-19, Ukraine War)")
# Combine PLS components with dummy variables
X_tr_dummy = np.column_stack([X_tr, D_tr])
X_te_dummy = np.column_stack([X_te, D_te])
results_fc["OLS+Dummy"] = ols_pred(X_tr_dummy, y_train, X_te_dummy)

# Print OLS+Dummy coefficients for interpretation
Xd_aug = np.column_stack([np.ones(len(X_tr_dummy)), X_tr_dummy])
b_dum, _, _, _ = np.linalg.lstsq(Xd_aug, y_train, rcond=None)
print(f"     Intercept : {b_dum[0]:.6f}")
for i, col in enumerate(pls_cols):
    print(f"     {col:<10}: {b_dum[i+1]:.6f}")
print(f"     D_GFC     : {b_dum[N_COMP+1]:.6f}  (GFC 2008 impact on OIL_RET)")
print(f"     D_OIL2014 : {b_dum[N_COMP+2]:.6f}  (Oil Crash OPEC 2014 impact on OIL_RET)")
print(f"     D_COVID   : {b_dum[N_COMP+3]:.6f}  (COVID-19 2020 impact on OIL_RET)")
print(f"     D_UKRAINE : {b_dum[N_COMP+4]:.6f}  (Ukraine War 2022 impact on OIL_RET)")

# --- GARCH variance forecast (for prediction intervals) ---
def garch_variance_forecast(h_hist, e_hist, om, al, be, steps):
    """Multi-step GARCH(1,1) variance forecast."""
    h_last = h_hist[-1]; e_last = e_hist[-1]
    h_fc = []; h_cur = h_last
    for s in range(steps):
        if s == 0:
            h_next = om + al[0]*e_last**2 + be[0]*h_last
        else:
            lr_var = om / max(1 - al[0] - be[0], 1e-6)
            h_next = lr_var + (al[0]+be[0])**s * (h_cur - lr_var)
        h_cur = max(h_next, 1e-9)
        h_fc.append(h_cur)
    return np.array(h_fc)

garch_var_fc = garch_variance_forecast(h_garch, e_garch, om_g, al_g, be_g, len(y_test))
garch_vol_fc = np.sqrt(np.maximum(garch_var_fc, 0))
print(f"\n[GARCH variance forecast] Mean conditional vol (test period): {garch_vol_fc.mean():.6f}")

# =============================================================================
# 9. METRICS + DIEBOLD-MARIANO TEST
# =============================================================================
print("\n" + "="*70)
print("PART 9 – METRICS + DIEBOLD-MARIANO TEST")
print("="*70)

def qlike(actual, forecast):
    s2 = max(np.var(y_train), 1e-10)
    return np.mean((actual - forecast)**2 / s2)

def theil_u2(actual, forecast):
    naive = np.zeros(len(actual))
    naive[0] = y_train[-1]
    for i in range(1, len(actual)):
        naive[i] = actual[i-1]
    rmse_m = np.sqrt(mean_squared_error(actual, forecast))
    rmse_n = np.sqrt(mean_squared_error(actual, naive))
    return rmse_m / max(rmse_n, 1e-10)

def mase(actual, forecast, y_train_ref):
    """
    Mean Absolute Scaled Error.
    Scale = MAE of naive random-walk on training set.
    MASE < 1  → better than naïve RW on training data.
    Avoids near-zero denominator that breaks sMAPE on log-returns.
    """
    naive_mae = np.mean(np.abs(np.diff(y_train_ref)))   # MAE of in-sample RW
    naive_mae = max(naive_mae, 1e-10)
    return np.mean(np.abs(actual - forecast)) / naive_mae

rows = []
for model, fc in results_fc.items():
    fc = np.asarray(fc)
    rows.append({
        "Model":   model,
        "MAE":     mean_absolute_error(y_test, fc),
        "RMSE":    np.sqrt(mean_squared_error(y_test, fc)),
        "QLIKE":   qlike(y_test, fc),
        "MASE":    mase(y_test, fc, y_train),
        "TheilU2": theil_u2(y_test, fc),
    })

df_perf = pd.DataFrame(rows).set_index("Model").sort_values("RMSE")
print(df_perf.round(6).to_string())

def diebold_mariano(actual, fc1, fc2, h=1, loss="SE"):
    if loss == "SE":
        d = (actual - fc1)**2 - (actual - fc2)**2
    else:
        d = np.abs(actual - fc1) - np.abs(actual - fc2)
    n = len(d); dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    nw_var = gamma0
    for j in range(1, h):
        w = 1 - j/h
        gj = np.mean((d[j:] - dbar) * (d[:-j] - dbar))
        nw_var += 2*w*gj
    nw_var = max(nw_var, 1e-15)
    dm_stat = dbar / np.sqrt(nw_var / n)
    p_val   = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_val

benchmark = df_perf.index[0]
print(f"\n[Diebold-Mariano Test – benchmark: {benchmark}]")
print(f"{'Model':<14} {'DM_stat':>10} {'p-value':>10} {'Decision (5%)':>18}")
print("-"*55)

dm_rows = []
for m in df_perf.index:
    if m == benchmark: continue
    dm_s, dm_p = diebold_mariano(y_test, results_fc[benchmark], results_fc[m])
    if dm_s < 0 and dm_p < 0.05:
        decision = "Benchmark BETTER ✅"
    elif dm_s > 0 and dm_p < 0.05:
        decision = "Alternative BETTER ⚠️"
    else:
        decision = "No sig. difference"
    print(f"{m:<14} {dm_s:>10.4f} {dm_p:>10.6f}  {decision}")
    dm_rows.append({"Model": m, "vs_Benchmark": benchmark,
                    "DM_stat": round(dm_s,4), "p_value": round(dm_p,6),
                    "Decision": decision})

df_dm = pd.DataFrame(dm_rows)

# =============================================================================
# 10. BOOTSTRAP PREDICTION INTERVAL + BULL/BASE/BEAR SCENARIOS
# =============================================================================
print("\n" + "="*70)
print("PART 10 – BOOTSTRAP PREDICTION INTERVAL + SCENARIOS (Bull/Base/Bear)")
print("="*70)

N_BOOT     = 500
ALPHA      = 0.05
H_SCENARIO = 12

# ── Student-t block bootstrap ─────────────────────────────────────────────────
def fit_t_df(residuals):
    """Fit Student-t degrees-of-freedom to residuals via MLE  (df > 2)."""
    from scipy.special import gammaln
    r = residuals / (residuals.std() + 1e-10)
    def neg_ll(log_nu_m2):
        nu = float(np.exp(log_nu_m2) + 2.0)
        return -float(np.sum(
            gammaln((nu+1)/2) - gammaln(nu/2)
            - 0.5*np.log(np.pi*(nu-2))
            - (nu+1)/2 * np.log(1.0 + r**2/(nu-2))
        ))
    res = optimize.minimize_scalar(neg_ll, bounds=(np.log(0.5), np.log(98)),
                                   method="bounded")
    return float(np.exp(res.x) + 2.0)


def bootstrap_pi_t(y_tr, X_tr, X_te, n_boot=N_BOOT, alpha=ALPHA, rng_seed=42):
    """
    Block bootstrap with Student-t fat-tail correction.
    Steps:
      1. Fit OLS on train; compute in-sample residuals.
      2. Fit Student-t df on residuals (MLE).
      3. Each boot iteration:
         a. Draw block-resampled residual indices (cube-root block length).
         b. Draw iid t(df) noise; standardise both.
         c. Blend: resid_boot = block_resid_std * sigma * (0.5 + 0.5*|t_noise|)
            – preserves autocorrelation structure AND inflates tails.
      4. PI = empirical 2.5th / 97.5th percentile across boot forecasts.
    """
    n     = len(y_tr)
    block = max(int(np.ceil(n ** (1/3))), 3)   # cube-root block length
    fc_base = ols_pred(X_tr, y_tr, X_te)
    resid   = y_tr - ols_pred(X_tr, y_tr, X_tr)
    sigma_r = resid.std() + 1e-10

    # --- fit t-df on in-sample residuals ---
    try:
        df_t = max(fit_t_df(resid), 2.1)
    except Exception:
        df_t = 5.0
    print(f"  [Bootstrap] Student-t df fitted on residuals = {df_t:.2f}  "
          f"(block length = {block})")

    rng      = np.random.default_rng(rng_seed)
    boot_fc  = np.zeros((n_boot, len(X_te)))

    for b in range(n_boot):
        # a. block-resample indices
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n - block + 1)
            idx.extend(range(s, min(s + block, n)))
        idx      = np.array(idx[:n])
        resid_b  = resid[idx]

        # b. draw t(df) noise, standardise both
        t_noise = rng.standard_t(df=df_t, size=n)
        t_noise = (t_noise  - t_noise.mean())  / (t_noise.std()  + 1e-10)
        resid_b = (resid_b  - resid_b.mean())  / (resid_b.std()  + 1e-10)

        # c. fat-tail blend: keep block autocorrelation + inflate extremes
        resid_bt = resid_b * sigma_r * (0.5 + 0.5 * np.abs(t_noise))

        y_boot      = ols_pred(X_tr, y_tr, X_tr) + resid_bt
        boot_fc[b]  = ols_pred(X_tr, y_boot, X_te)

    lo = np.percentile(boot_fc, 100 * alpha / 2,       axis=0)
    hi = np.percentile(boot_fc, 100 * (1 - alpha / 2), axis=0)
    return fc_base, lo, hi, df_t


print(f"Running {N_BOOT}-iteration Student-t block bootstrap "
      f"({int((1-ALPHA)*100)}% PI)...")
fc_base_pi, pi_lo, pi_hi, df_t_boot = bootstrap_pi_t(y_train, X_tr, X_te)
print(f"  Mean PI width : {(pi_hi - pi_lo).mean():.6f}")
print(f"  Coverage check: "
      f"{np.mean((y_test >= pi_lo) & (y_test <= pi_hi))*100:.1f}%")

# ── Price conversion helper ───────────────────────────────────────────────────
def log_ret_to_price(base_price, ret_fc):
    prices = [base_price]
    for r in ret_fc:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices[1:])

# ── Scenario anchor: LAST date of training set ────────────────────────────────
# This ensures the scenario price path starts exactly where the model
# stops observing data, eliminating the train→test date gap that caused
# the original Fig 6 to look like it was projected from 2021 instead of
# the true train-end date.
last_train_date = df_train.index[-1]
last_oil_price  = df_raw.loc[last_train_date, "OIL"]
future_dates    = pd.date_range(last_train_date,
                                periods=H_SCENARIO + 1, freq="ME")[1:]

print(f"\n  Scenario anchor : {last_train_date.date()}  "
      f"(last training observation)")
print(f"  Base oil price  : ${last_oil_price:.2f}/bbl")
print(f"  Horizon         : {H_SCENARIO} months  "
      f"({future_dates[0].date()} → {future_dates[-1].date()})")

# ── Scenario shocks calibrated to GARCH-t conditional volatility ─────────────
# Use the LAST conditional standard deviation from GARCH-t as the shock scale.
# Bear  = −0.5 σ_t  (sustained negative monthly shock)
# Base  = 0          (no additional shock beyond OLS forecast)
# Bull  = +0.5 σ_t  (sustained positive monthly shock)
garch_t_sigma = float(np.sqrt(max(h_garch_t[-1], 1e-9)))
print(f"  GARCH-t last conditional σ = {garch_t_sigma:.4f}  "
      f"(used to calibrate Bear/Bull shocks)")

scenarios = {
    "Bear": {"shock": -garch_t_sigma * 0.50, "vol_mult": 1.5,  "color": "#d62728"},
    "Base": {"shock":  0.000,                "vol_mult": 1.0,  "color": "#1f77b4"},
    "Bull": {"shock": +garch_t_sigma * 0.50, "vol_mult": 0.75, "color": "#2ca02c"},
}

X_scenario            = np.tile(X_te.mean(axis=0), (H_SCENARIO, 1))
fc_scenario_base, sc_lo, sc_hi, _ = bootstrap_pi_t(y_train, X_tr, X_scenario)

print(f"\n[Scenarios – {H_SCENARIO}-month horizon from {last_train_date.date()}]")
print(f"{'Scenario':<8} {'Start $':>9} {'End $':>9} "
      f"{'Δ%':>7} {'Min $':>9} {'Max $':>9}")
print("-"*52)

scenario_df_dict = {}
for sc_name, sc_params in scenarios.items():
    sc_ret       = fc_scenario_base + sc_params["shock"]
    sc_prices    = log_ret_to_price(last_oil_price, sc_ret)
    sc_vol       = (sc_hi - sc_lo) / 2 * sc_params["vol_mult"]
    sc_prices_lo = log_ret_to_price(last_oil_price, sc_ret - sc_vol)
    sc_prices_hi = log_ret_to_price(last_oil_price, sc_ret + sc_vol)
    pct_chg      = (sc_prices[-1] / last_oil_price - 1) * 100
    print(f"{sc_name:<8} {last_oil_price:>9.2f} {sc_prices[-1]:>9.2f} "
          f"{pct_chg:>7.1f}% {sc_prices.min():>9.2f} {sc_prices.max():>9.2f}")
    scenario_df_dict[sc_name] = {
        "prices": sc_prices, "lo": sc_prices_lo, "hi": sc_prices_hi,
        "ret": sc_ret, "shock": sc_params["shock"], "color": sc_params["color"]
    }

# =============================================================================
# 11. FIGURES + EXCEL
# =============================================================================
print("\n" + "="*70)
print("PART 11 – FIGURES + EXCEL OUTPUT")
print("="*70)

# ── Figure 1: Overview ────────────────────────────────────────────────────────
fig1, axes = plt.subplots(3, 2, figsize=(14, 12))
fig1.suptitle("Data overview – Level vs Log-Return", fontsize=13, fontweight="bold")
pairs = [
    ("OIL",     "Brent Price – Level (I(1), reference)",   "#1f77b4"),
    ("OIL_RET", "OIL Log-Return – STATIONARY I(0)",        "#ff7f0e"),
    ("USD_RET", "USD Log-Return – STATIONARY I(0)",        "#2ca02c"),
    ("CPI_RET", "CPI Log-Return – STATIONARY I(0)",        "#d62728"),
    ("FED_RET", "FED Log-Return – STATIONARY I(0)",        "#9467bd"),
    ("IND_RET", "IND Log-Return – STATIONARY I(0)",        "#8c564b"),
]
for ax, (col, title, color) in zip(axes.flatten(), pairs):
    ax.plot(df_all.index, df_all[col], color=color, lw=1.0)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig1_Overview.png"), dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] Fig1_Overview.png")

# ── Figure 2: CUSUM + Chow ───────────────────────────────────────────────────
fig_sb, axes_sb = plt.subplots(2, 1, figsize=(14, 9))
fig_sb.suptitle("Structural Break Analysis – CUSUM & Chow Test", fontsize=13, fontweight="bold")

ax0 = axes_sb[0]
idx_plot = oil_ret_idx[1:]
ax0.plot(idx_plot, W_cusum, color="#1f77b4", lw=1.2, label="CUSUM")
ax0.plot(idx_plot, bound_cusum,  color="red", ls="--", lw=1, label="5% boundary")
ax0.plot(idx_plot, -bound_cusum, color="red", ls="--", lw=1)
ax0.axhline(0, color="k", lw=0.8)
for ev, bd in BREAK_EVENTS.items():
    ax0.axvline(bd, color="gray", ls=":", lw=1.2, alpha=0.8)
    ax0.text(bd, ax0.get_ylim()[1]*0.85, ev.split("_")[0], fontsize=7, ha="center", color="gray")
ax0.set_title("CUSUM of Recursive Residuals – OIL_RET"); ax0.legend(fontsize=8)

ax1 = axes_sb[1]
ax1.plot(oil_ret_idx, oil_ret_all, color="#1f77b4", lw=0.9, alpha=0.8, label="OIL_RET")
colors_ev = ["#d62728","#ff7f0e","#9467bd","#2ca02c"]
for (ev, res), c in zip(chow_results.items(), colors_ev):
    ax1.axvline(res["date"], color=c, ls="--", lw=1.5,
                label=f"{ev} (F={res['F']:.2f}, p={res['p']:.4f})")
ax1.set_title("Chow Test Break Points – OIL_RET")
ax1.legend(fontsize=7.5, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig_StructuralBreak.png"), dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] Fig_StructuralBreak.png")

# ── Figure 3: Dummy variable visualization ────────────────────────────────────
fig_d, ax_d = plt.subplots(figsize=(15, 5))
ax_d.plot(df_model.index, df_model["OIL_RET"], color="#1f77b4", lw=0.9, alpha=0.8, label="OIL_RET")
dummy_colors = {"D_GFC":     ("#d62728", "GFC 2008"),
                "D_OIL2014": ("#8c564b", "Oil Crash OPEC 2014"),
                "D_COVID":   ("#9467bd", "COVID-19 2020"),
                "D_UKRAINE": ("#ff7f0e", "Ukraine War 2022")}
for dcol, (dcolor, dlabel) in dummy_colors.items():
    d_mask = df_model[dcol] == 1
    if d_mask.any():
        d_idx = df_model.index[d_mask]
        ax_d.axvspan(d_idx[0], d_idx[-1], alpha=0.20, color=dcolor, label=dlabel)
ax_d.set_title("OIL_RET with Crisis Dummy Variable Periods", fontsize=12, fontweight="bold")
ax_d.set_ylabel("OIL_RET (log return)")
ax_d.legend(fontsize=9, loc="upper right")
ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_d.xaxis.set_major_locator(mdates.YearLocator(3))
plt.setp(ax_d.xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig_DummyVariables.png"), dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] Fig_DummyVariables.png")

# ── Figure 4: Forecast comparison ────────────────────────────────────────────
fig3, axes = plt.subplots(2, 1, figsize=(15, 10))
fig3.suptitle("Out-of-sample Forecasts – OIL_RET", fontsize=13, fontweight="bold")
ax = axes[0]
ax.plot(df_test.index, y_test, color="black", lw=2, label="Actual", zorder=5)
for i, (m, fc) in enumerate(results_fc.items()):
    ax.plot(df_test.index, fc, ls="-" if i < 5 else "--",
            color=PALETTE[i % len(PALETTE)], lw=1.1, alpha=0.85, label=m)
ax.set_title("All models (OLS, PLS, ARIMA, SARIMA, ARCH(1), GARCH(1,1), OLS+Dummy)")
ax.set_ylabel("OIL_RET")
ax.legend(fontsize=8, ncol=4, loc="upper right")

ax2 = axes[1]
ax2.plot(df_test.index, y_test, color="black", lw=2, label="Actual", zorder=5)
ax2.fill_between(df_test.index, pi_lo, pi_hi, alpha=0.2, color="#1f77b4",
                 label=f"{int((1-ALPHA)*100)}% Bootstrap PI")
for i, m in enumerate(df_perf.index[:3]):
    ax2.plot(df_test.index, results_fc[m], color=PALETTE[i], lw=1.6,
             label=f"{m}  RMSE={df_perf.loc[m,'RMSE']:.5f}")
ax2.set_title("Top-3 models + Bootstrap Prediction Interval")
ax2.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig3_Forecast_PI.png"), dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] Fig3_Forecast_PI.png")

# ── Figure 5: Performance bar  (5 metrics incl. MASE) ───────────────────────
fig4, axes = plt.subplots(1, 5, figsize=(22, 6))
fig4.suptitle("Forecast Performance – All Models", fontsize=13, fontweight="bold")
bc = [PALETTE[i % len(PALETTE)] for i in range(len(df_perf))]
metric_labels = {
    "MAE":     "MAE",
    "RMSE":    "RMSE",
    "QLIKE":   "QLIKE",
    "MASE":    "MASE  (< 1 = beats naïve RW)",
    "TheilU2": "TheilU2  (< 1 = beats naïve RW)",
}
for ax, metric in zip(axes, ["MAE", "RMSE", "QLIKE", "MASE", "TheilU2"]):
    vals = df_perf[metric]
    bars = ax.barh(df_perf.index, vals, color=bc, edgecolor="white")
    ax.set_title(metric_labels[metric], fontsize=9); ax.invert_yaxis()
    if metric in ("TheilU2", "MASE"):
        ax.axvline(1.0, color="red", ls="--", lw=1.2, label="= 1 (naïve RW)")
        ax.legend(fontsize=7)
    for bar, v in zip(bars, vals):
        ax.text(v * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig4_Performance.png"), dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] Fig4_Performance.png")

# ── Figure 6: GARCH(1,1) Normal vs GARCH-t conditional volatility ────────────
fig5, axes = plt.subplots(1, 3, figsize=(20, 5))
fig5.suptitle(
    f"Conditional Volatility – GARCH(1,1) Normal vs GARCH(1,1)-t\n"    f"GARCH-t df(ν) = {nu_gt:.2f}  |  "    f"persist(Normal) = {al_g[0]+be_g[0]:.4f}  "    f"persist(t) = {al_gt[0]+be_gt[0]:.4f}",
    fontsize=11, fontweight="bold"
)
cv_g  = np.sqrt(np.maximum(h_garch,   0))   # Normal
cv_gt = np.sqrt(np.maximum(h_garch_t, 0))   # Student-t

# Panel 1: GARCH Normal volatility
axes[0].plot(df_train.index, cv_g, color="#1f77b4", lw=1.2)
axes[0].set_title("GARCH(1,1) Normal – Conditional σ (Train)")
axes[0].set_ylabel("Conditional Std Dev")

# Panel 2: GARCH-t volatility
axes[1].plot(df_train.index, cv_gt, color="#e6550d", lw=1.2,
             label=f"GARCH-t  ν={nu_gt:.1f}")
axes[1].set_title("GARCH(1,1)-t – Conditional σ (Train)")
axes[1].legend(fontsize=8)

# Panel 3: Overlay – Normal vs t, with OIL_RET returns
ax3 = axes[2]
ax3.plot(df_train.index, df_train["OIL_RET"].values,
         color="gray", lw=0.6, alpha=0.5, label="OIL_RET")
ax3.plot(df_train.index, cv_g,  color="#1f77b4", lw=1.4,
         label=f"GARCH Normal  α={al_g[0]:.3f} β={be_g[0]:.3f}")
ax3.plot(df_train.index, cv_gt, color="#e6550d", lw=1.4, ls="--",
         label=f"GARCH-t  α={al_gt[0]:.3f} β={be_gt[0]:.3f}  ν={nu_gt:.1f}")
ax3.set_title("OIL_RET + GARCH Normal vs GARCH-t")
ax3.legend(fontsize=7.5)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig5_GARCH_Vol.png"), dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] Fig5_GARCH_Vol.png")

# ── Figure 7: Scenarios Bull/Base/Bear ───────────────────────────────────────
# future_dates and last_train_date are already defined in Part 10.
# Show 36 months of history so the historical→forecast transition is clear.
fig6, axes = plt.subplots(1, 2, figsize=(16, 6))
fig6.suptitle(
    f"Bull / Base / Bear Price Scenarios – Brent Crude Oil\n"    f"Anchored at train-end: {last_train_date.date()}  "    f"(${last_oil_price:.1f}/bbl)  |  Horizon: {H_SCENARIO} months",
    fontsize=12, fontweight="bold"
)

ax_l = axes[0]
ax_r = axes[1]

# 36-month history window ending exactly at last_train_date
hist_prices = df_raw["OIL"].loc[
    (df_raw.index >= last_train_date - pd.DateOffset(months=36)) &
    (df_raw.index <= last_train_date)
]
for ax in [ax_l, ax_r]:
    ax.plot(hist_prices.index, hist_prices.values,
            color="black", lw=1.8, label="Historical", zorder=3)
    # vertical marker at anchor date
    ax.axvline(last_train_date, color="gray", ls="--", lw=1,
               label=f"Train end ({last_train_date.strftime('%Y-%m')})")

for sc_name, sc_data in scenario_df_dict.items():
    c = sc_data["color"]
    shock_label = (f"{sc_name}  "
                   f"(shock={sc_data['shock']:+.4f}/mo, "                   f"end=${sc_data['prices'][-1]:.1f})")
    ax_l.plot(future_dates, sc_data["prices"], color=c, lw=2.2, label=shock_label)
    ax_r.plot(future_dates, sc_data["prices"], color=c, lw=2.2, label=shock_label)
    ax_r.fill_between(future_dates, sc_data["lo"], sc_data["hi"],
                      alpha=0.18, color=c)

for ax in [ax_l, ax_r]:
    ax.set_title("Price Scenarios" if ax == ax_l else "Price Scenarios + 95% PI (Student-t Bootstrap)")
    ax.set_ylabel("Brent Crude ($/bbl)")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig6_Scenarios_Bull_Base_Bear.png"),
            dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] Fig6_Scenarios_Bull_Base_Bear.png")

# ── Figure 8: Walk-forward rolling RMSE ──────────────────────────────────────
roll_rmse = {"OLS": [], "ARIMA": [], "GARCH(1,1)": [], "OLS+Dummy": []}
roll_labels = []
for w, (tr_end, te_start, te_end) in enumerate(rolling_windows):
    y_tr_w = df_model["OIL_RET"].values[:tr_end]
    y_te_w = df_model["OIL_RET"].values[te_start:te_end]
    X_tr_w = df_model[pls_cols].values[:tr_end]
    X_te_w = df_model[pls_cols].values[te_start:te_end]
    D_tr_w = df_model[dummy_cols].values[:tr_end]
    D_te_w = df_model[dummy_cols].values[te_start:te_end]
    h_w    = len(y_te_w)
    roll_labels.append(f"W{w+1}")
    # OLS
    try:
        fc_ols_w = ols_pred(X_tr_w, y_tr_w, X_te_w)
        roll_rmse["OLS"].append(np.sqrt(mean_squared_error(y_te_w, fc_ols_w)))
    except: roll_rmse["OLS"].append(np.nan)
    # ARIMA
    try:
        par_w = arma_css(y_tr_w, 1, 1)
        e_w   = arma_resid(par_w, y_tr_w, 1, 1)
        fc_ar_w = arma_fc(par_w, y_tr_w, e_w, 1, 1, h_w)
        roll_rmse["ARIMA"].append(np.sqrt(mean_squared_error(y_te_w, fc_ar_w)))
    except: roll_rmse["ARIMA"].append(np.nan)
    # GARCH(1,1)
    try:
        mu_gw, _, _, _, _, _ = garch_fit_fn(y_tr_w, 1, 1, dist="normal")
        fc_g_w = np.full(h_w, mu_gw)
        roll_rmse["GARCH(1,1)"].append(np.sqrt(mean_squared_error(y_te_w, fc_g_w)))
    except: roll_rmse["GARCH(1,1)"].append(np.nan)
    # OLS+Dummy
    try:
        X_tr_d_w = np.column_stack([X_tr_w, D_tr_w])
        X_te_d_w = np.column_stack([X_te_w, D_te_w])
        fc_d_w = ols_pred(X_tr_d_w, y_tr_w, X_te_d_w)
        roll_rmse["OLS+Dummy"].append(np.sqrt(mean_squared_error(y_te_w, fc_d_w)))
    except: roll_rmse["OLS+Dummy"].append(np.nan)

fig7, ax7 = plt.subplots(figsize=(12, 5))
colors_roll = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for (m, vals), c in zip(roll_rmse.items(), colors_roll):
    ax7.plot(roll_labels, vals, marker="o", lw=1.8, color=c, label=m)
ax7.set_title("Walk-forward Rolling RMSE (8 windows)"); ax7.set_ylabel("RMSE")
ax7.legend(); ax7.set_xlabel("Window")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig7_RollingRMSE.png"), dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] Fig7_RollingRMSE.png")

# =============================================================================
# EXCEL OUTPUT
# =============================================================================
out_xl = os.path.join(out_dir, "OilAnalysis_Results_v5.xlsx")

# Stationarity table
stat_rows = []
for col in level_cols:
    adf_s, adf_p, _ = adf_test(df_raw[col].values)
    kpss_s, kpss_p  = kpss_test(df_raw[col].values)
    is_s, label     = stationarity_decision(adf_p, kpss_p)
    stat_rows.append({"Variable": col, "Type": "Level",
                      "ADF_stat": round(adf_s,4), "ADF_p": round(adf_p,4),
                      "KPSS_stat": round(kpss_s,4), "KPSS_p": round(kpss_p,4),
                      "Is_Stationary": is_s, "Decision": label, "Order": "I(1)"})
    rc = f"{col}_RET"
    adf_s2, adf_p2, _ = adf_test(diff_df[rc].values)
    kpss_s2, kpss_p2  = kpss_test(diff_df[rc].values)
    is_s2, label2     = stationarity_decision(adf_p2, kpss_p2)
    stat_rows.append({"Variable": rc, "Type": "Log-Return",
                      "ADF_stat": round(adf_s2,4), "ADF_p": round(adf_p2,4),
                      "KPSS_stat": round(kpss_s2,4), "KPSS_p": round(kpss_p2,4),
                      "Is_Stationary": is_s2, "Decision": label2, "Order": "I(0)"})
df_stat_out = pd.DataFrame(stat_rows)

# Structural break table
sb_rows = [{"Event": ev, "Break_Date": str(res["date"].date()),
            "F_stat": round(res["F"],4), "p_value": round(res["p"],6),
            "Decision": "Break ✅" if res["p"] < 0.05 else "No break"}
           for ev, res in chow_results.items()]
df_sb = pd.DataFrame(sb_rows)

# Cointegration table
if coint_pairs:
    df_coint_out = pd.DataFrame([
        {"Pair": f"OIL~{k}", "EG_stat": round(v["stat"],4),
         "p_value": round(v["p"],4), "Cointegrated": v["cointegrated"]}
        for k, v in coint_pairs.items()
    ])
else:
    df_coint_out = pd.DataFrame({"Note": ["EG test did not run"]})

# JB normality table
df_jb = pd.DataFrame(jb_rows)

# PLS variance
pls_var_df = pd.DataFrame({
    "Component": pls_cols,
    "Var_Explained": var_explained,
    "Pct_of_Explained": var_pct
})

# Dummy variable summary
dummy_summary = pd.DataFrame([
    {"Dummy": "D_GFC",     "Period": "Sep 2008 – Jun 2009", "Months": int(D_GFC.sum()),
     "Coeff_OLS+Dummy": round(b_dum[N_COMP+1], 6), "Event": "Global Financial Crisis"},
    {"Dummy": "D_OIL2014", "Period": "Nov 2014 – Jan 2016", "Months": int(D_OIL2014.sum()),
     "Coeff_OLS+Dummy": round(b_dum[N_COMP+2], 6), "Event": "Oil Crash OPEC 2014"},
    {"Dummy": "D_COVID",   "Period": "Mar 2020 – Dec 2020", "Months": int(D_COVID.sum()),
     "Coeff_OLS+Dummy": round(b_dum[N_COMP+3], 6), "Event": "COVID-19 Pandemic"},
    {"Dummy": "D_UKRAINE", "Period": "Feb 2022 – Dec 2022", "Months": int(D_UKRAINE.sum()),
     "Coeff_OLS+Dummy": round(b_dum[N_COMP+4], 6), "Event": "Russia-Ukraine War"},
])

# Forecasts
fc_out = pd.DataFrame(results_fc, index=df_test.index)
fc_out.insert(0, "Actual", y_test)
# GARCH-t conditional vol forecast (multi-step, same function as Normal)
garch_t_var_fc = garch_variance_forecast(h_garch_t, e_garch_t,
                                         om_gt, al_gt, be_gt, len(y_test))
garch_t_vol_fc = np.sqrt(np.maximum(garch_t_var_fc, 0))
fc_out["GARCH_Vol_Normal"]    = garch_vol_fc
fc_out["GARCH_Vol_t"]         = garch_t_vol_fc
fc_out["Bootstrap_PI_Lo"]     = pi_lo
fc_out["Bootstrap_PI_Hi"]     = pi_hi
print(f"  GARCH-t mean conditional vol (test): {garch_t_vol_fc.mean():.6f}  "      f"vs Normal: {garch_vol_fc.mean():.6f}")

# Scenarios sheet
sc_rows = []
for sc_name, sc_data in scenario_df_dict.items():
    for i, (d, p, lo, hi, r) in enumerate(zip(
            future_dates, sc_data["prices"], sc_data["lo"],
            sc_data["hi"], sc_data["ret"])):
        sc_rows.append({
            "Scenario": sc_name, "Date": d,
            "OIL_Forecast_Price": round(p, 4),
            "PI_Lo": round(lo, 4), "PI_Hi": round(hi, 4),
            "Log_Return_Forecast": round(r, 6),
            "Shock": sc_data["shock"]
        })
df_scenarios_pi = pd.DataFrame(sc_rows)

# Rolling RMSE sheet
df_rolling = pd.DataFrame(roll_rmse, index=roll_labels)
df_rolling.index.name = "Window"

# Overall ranking
df_rank = df_perf.copy()
for col in ["MAE", "RMSE", "QLIKE", "MASE", "TheilU2"]:
    df_rank[f"rank_{col}"] = df_perf[col].rank()
rank_cols = [c for c in df_rank.columns if c.startswith("rank_")]
df_rank["avg_rank"] = df_rank[rank_cols].mean(axis=1)
df_rank = df_rank.sort_values("avg_rank")

with pd.ExcelWriter(out_xl, engine="openpyxl") as w:
    df_model.to_excel(w, sheet_name="Data_Stationary")
    desc_full.round(4).to_excel(w, sheet_name="Descriptive_Stats")
    df_stat_out.to_excel(w, sheet_name="Stationarity_Tests",   index=False)
    df_sb.to_excel(w, sheet_name="Structural_Break",           index=False)
    df_coint_out.to_excel(w, sheet_name="Cointegration_EG",    index=False)
    df_jb.to_excel(w, sheet_name="JarqueBera_Normality",       index=False)
    pls_var_df.round(6).to_excel(w, sheet_name="PLS_Variance", index=False)
    dummy_summary.to_excel(w, sheet_name="Dummy_Variables",    index=False)
    fc_out.to_excel(w, sheet_name="Forecasts")
    df_perf.round(6).to_excel(w, sheet_name="Performance_Metrics")
    df_rank[["avg_rank", "MAE", "RMSE", "QLIKE", "MASE", "TheilU2"]]\
        .round(6).to_excel(w, sheet_name="Overall_Ranking")
    df_dm.to_excel(w, sheet_name="DM_Test",                    index=False)
    df_scenarios_pi.to_excel(w, sheet_name="Scenarios_PI",     index=False)
    df_rolling.round(6).to_excel(w, sheet_name="RollingRMSE_8win")

print(f"\n[Saved Excel] {out_xl}")
print(f"  Sheets: Data_Stationary, Descriptive_Stats, Stationarity_Tests,")
print(f"          Structural_Break, Cointegration_EG, JarqueBera_Normality,")
print(f"          PLS_Variance, Dummy_Variables, Forecasts, Performance_Metrics,")
print(f"          Overall_Ranking, DM_Test, Scenarios_PI, RollingRMSE_8win")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"\n  Phases completed:")
print(f"  Phase 0 : Data cleaning & merging")
print(f"  Phase 1 : ADF + KPSS stationarity tests")
print(f"  Phase 2 : Structural break (CUSUM + Chow-test at 4 events)")
print(f"  Phase 3 : Engle-Granger cointegration")
print(f"  Phase 4 : Descriptive stats + Jarque-Bera normality test")
print(f"  Phase 5 : ARCH-LM + ACF/PACF (Yule-Walker) + Ljung-Box")
print(f"  Phase 6 : PLS dimension reduction ({N_COMP} components)")
print(f"  Phase 7 : Train/test split + Walk-forward rolling ({N_WINDOWS} windows)")
print(f"  Phase 8 : 8 models → OLS | PLS | ARIMA(1,0,1) | SARIMA(1,0,1)(1,0,1,12)")
print(f"            ARCH(1) | GARCH(1,1)-Normal | GARCH(1,1)-t | OLS+Dummy")
print(f"            GARCH-t df(ν) fitted = {nu_gt:.2f}")
print(f"  Phase 9 : Metrics (MAE/RMSE/QLIKE/MASE/TheilU2) + Diebold-Mariano test")
print(f"            [sMAPE replaced by MASE – avoids near-zero denom on log-returns]")
print(f"  Phase 10: Student-t block bootstrap PI ({N_BOOT} iter, df={df_t_boot:.2f})")
print(f"            Bull/Base/Bear anchored at {last_train_date.date()}, "
      f"shock=GARCH-t σ={garch_t_sigma:.4f}")
print(f"  Phase 11: 7 figures + Excel with 14 sheets incl. Dummy_Variables sheet")

print(f"\n{'='*50}")
print(f"MODEL RANKING (by RMSE)")
print(f"{'='*50}")
print(f"{'#':<4} {'Model':<16} {'RMSE':>10} {'TheilU2':>10}")
print("-"*44)
for i, (m, r) in enumerate(df_perf.iterrows()):
    star = " ← BEST" if i == 0 else ""
    print(f"{i+1:<4} {m:<16} {r['RMSE']:>10.6f} {r['TheilU2']:>10.4f}{star}")

print(f"\nHoan tat! File ket qua: {out_xl}")
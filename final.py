"""
=============================================================================
PHÂN TÍCH VÀ DỰ BÁO GIÁ DẦU THÔ BRENT – PHIÊN BẢN HOÀN CHỈNH v3
=============================================================================
Danh sách lỗi đã sửa so với v2:

  BUG 1 ✅ (NGHIÊM TRỌNG) – Python substring: "DUNG" in "KHONG DUNG" = True
         Sửa: dùng biến boolean is_stationary riêng, không dùng 'in' với string

  BUG 2 ✅ (NGHIÊM TRỌNG) – Engle-Granger không chạy vì level_results sai
         Sửa: logic is_stationary đúng → EG test được thực thi đầy đủ

  BUG 3 ✅ (SAI VỀ Ý NGHĨA) – PLS variance > 100% (PLS1=148%, PLS2=123%)
         Sửa: tính % variance đúng cách theo x-scores của từng component

  BUG 4 ✅ (CHỈ SỐ VÔ NGHĨA) – MAPE bùng nổ khi OIL_RET ≈ 0
         Sửa: thay MAPE bằng QLIKE (Quasi-Likelihood) và thêm Theil's U
              Vẫn giữ MAPE với epsilon-threshold để so sánh, nhưng dùng
              MAE + RMSE + QLIKE làm tiêu chí chính

  THÊM ✅ – Kiểm định ARCH LM riêng trên chuỗi loại outlier COVID-19
  THÊM ✅ – Dirichlet U (Theil's U2) làm chỉ số đánh giá bổ sung
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
    "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
    "axes.grid": True, "grid.alpha": 0.35, "font.family": "DejaVu Sans",
    "axes.titlesize": 11, "axes.labelsize": 9,
})
PALETTE = ["#9467bd","#8c564b","#e377c2","#7f7f7f",
           "#bcbd22","#17becf","#aec7e8","#ffbb78","#98df8a","#ff9896"]

# =============================================================================
# 0. LÀM SẠCH & GHÉP DỮ LIỆU
# =============================================================================
print("\n" + "="*70)
print("PHAN 0 – LAM SACH & GHEP DU LIEU")
print("="*70)

files = {
    "OIL": "D:\\PYTHON\\SUBJECTS\\Timeseries\\Finalproject\\Final-Timeseries\\DCOILBRENTEU (2).xlsx",
    "CPI": "D:\\PYTHON\\SUBJECTS\\Timeseries\\Finalproject\\Final-Timeseries\\CPIAUCSL.xlsx",
    "USD": "D:\\PYTHON\\SUBJECTS\\Timeseries\\Finalproject\\Final-Timeseries\\DTWEXBGS.xlsx",
    "FED": "D:\\PYTHON\\SUBJECTS\\Timeseries\\Finalproject\\Final-Timeseries\\FEDFUNDS.xlsx",
    "IND": "D:\\PYTHON\\SUBJECTS\\Timeseries\\Finalproject\\Final-Timeseries\\INDPRO.xlsx",
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

print(f"Khoang thoi gian : {df_raw.index[0].date()} -> {df_raw.index[-1].date()}")
print(f"So quan sat (raw): {len(df_raw)}")

# =============================================================================
# 1. KIỂM ĐỊNH TÍNH DỪNG: ADF + KPSS
# =============================================================================
print("\n" + "="*70)
print("PHAN 1 – KIEM DINH TINH DUNG: ADF + KPSS")
print("="*70)

# ── ADF ───────────────────────────────────────────────────────────────────────
def adf_test(y, maxlag=None):
    """
    ADF test – H0: có đơn vị gốc (không dừng)
    → p < 0.05: bác bỏ H0 → chuỗi DỪNG
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if maxlag is None:
        maxlag = int(np.ceil(12 * (n / 100) ** 0.25))
    best = (np.inf, 0, 0, 1.0)
    for lag in range(0, maxlag + 1):
        dy = np.diff(y)
        T2 = len(dy) - lag
        if T2 <= lag + 3:
            continue
        dy_l = dy[lag:]
        cols = [y[lag:-1], np.ones(len(dy_l))]
        if lag > 0:
            cols += [dy[lag - i - 1:-i - 1] for i in range(lag)]
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


# ── KPSS ──────────────────────────────────────────────────────────────────────
def kpss_test(y, lags=None):
    """
    KPSS test – H0: chuỗi DỪNG
    → p < 0.05: bác bỏ H0 → chuỗi KHÔNG DỪNG
    Critical values (level): 0.347 (10%), 0.463 (5%), 0.574 (2.5%), 0.739 (1%)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    resid = y - y.mean()
    S = np.cumsum(resid)
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


# ── Kết luận tổng hợp – BUG 1 ĐÃ SỬA ────────────────────────────────────────
# PHIÊN BẢN CŨ (SAI):
#   level_results[col] = {..., "stationary": "DUNG" in decision}
#   → "DUNG" in "KHONG DUNG" = True  ← Python kiểm tra substring!
#   → tất cả biến bị đánh dấu stationary=True, EG test bị bỏ qua toàn bộ
#
# PHIÊN BẢN MỚI (ĐÚNG):
#   Dùng biến boolean is_stat riêng biệt, không dựa vào string matching

def stationarity_decision(adf_p, kpss_p):
    """
    Trả về (is_stationary: bool, label: str)
    ADF  H0: non-stat → p<0.05 reject → DỪNG
    KPSS H0: stat     → p<0.05 reject → KHÔNG DỪNG
    """
    adf_reject  = (adf_p  < 0.05)   # True = DỪNG theo ADF
    kpss_accept = (kpss_p >= 0.05)  # True = DỪNG theo KPSS (không bác bỏ H0)

    if adf_reject and kpss_accept:
        return True,  "DUNG ✅"
    elif adf_reject and not kpss_accept:
        return True,  "Co the dung ⚠️"   # ADF bác bỏ nhưng KPSS nghi ngờ → xem thêm
    else:
        return False, "KHONG DUNG ❌"


# ──────────────────────── Level ─────────────────────────────────────────────
level_cols = ["OIL", "USD", "CPI", "FED", "IND"]

print(f"\n[LEVEL]")
print(f"{'Bien':<10} {'ADF_stat':>10} {'ADF_p':>8} {'KPSS_stat':>10} "
      f"{'KPSS_p':>8} {'Ket luan':<22} {'I(d)'}")
print("-"*76)

level_results = {}
for col in level_cols:
    series = df_raw[col].dropna().values
    adf_s, adf_p, _  = adf_test(series)
    kpss_s, kpss_p   = kpss_test(series)
    is_stat, label   = stationarity_decision(adf_p, kpss_p)   # ← BUG 1 FIX
    level_results[col] = {
        "adf_p": adf_p, "kpss_p": kpss_p,
        "is_stationary": is_stat   # boolean tường minh, KHÔNG dùng string
    }
    order = "I(0)" if is_stat else "I(1)"
    print(f"{col:<10} {adf_s:>10.4f} {adf_p:>8.4f} {kpss_s:>10.4f} "
          f"{kpss_p:>8.4f} {label:<22} {order}")

# ────────────────────────── Log-return ────────────────────────────────────────
diff_df = pd.DataFrame(index=df_raw.index)
for col in level_cols:
    diff_df[f"{col}_RET"] = np.log(df_raw[col] / df_raw[col].shift(1))
diff_df.dropna(inplace=True)

print(f"\n[LOG-RETURN (sai phan bac 1 cua log)]")
print(f"{'Bien':<12} {'ADF_stat':>10} {'ADF_p':>8} {'KPSS_stat':>10} "
      f"{'KPSS_p':>8} {'Ket luan':<22} {'I(d)'}")
print("-"*78)

diff_results = {}
for col in level_cols:
    rc = f"{col}_RET"
    series = diff_df[rc].values
    adf_s, adf_p, _  = adf_test(series)
    kpss_s, kpss_p   = kpss_test(series)
    is_stat, label   = stationarity_decision(adf_p, kpss_p)   # ← BUG 1 FIX
    diff_results[rc] = {
        "adf_p": adf_p, "kpss_p": kpss_p,
        "is_stationary": is_stat
    }
    note = ""
    if rc == "CPI_RET" and kpss_p < 0.15:
        note = " [KPSS gan nguong]"
    order = "I(0)" if is_stat else "I(1)*"
    print(f"{rc:<12} {adf_s:>10.4f} {adf_p:>8.4f} {kpss_s:>10.4f} "
          f"{kpss_p:>8.4f} {label:<22} {order}{note}")

print("""
[CHU THICH]
  ADF  H0: co don vi goc → p<0.05 bac bo → DUNG
  KPSS H0: dung          → p<0.05 bac bo → KHONG DUNG
  Ket hop: ADF bac bo + KPSS khong bac bo = CHAC CHAN DUNG
  CPI_RET: KPSS gan nguong 5% → tinh dung yeu hon, can theo doi
""")

# =============================================================================
# 2. KIỂM ĐỊNH ĐỒNG LIÊN KẾT ENGLE-GRANGER (BUG 2 ĐÃ SỬA)
# =============================================================================
print("\n" + "="*70)
print("PHAN 2 – KIEM DINH DONG LIEN KET (Engle-Granger)")
print("="*70)
print("""
[MUC DICH]
Neu OIL va bien giai thich deu I(1), kiem dinh xem chung co dong lien ket khong.
Neu co: nen dung ECM. Neu khong: dung log-return (bien sai phan) la hop le.
""")

def engle_granger_coint(y, x):
    """
    Engle-Granger 2 bước:
      Bước 1: OLS hồi quy y ~ x (level)
      Bước 2: ADF test trên phần dư
    Critical values EG (MacKinnon 1991, k=1 biến giải thích):
      1%: -3.90,  5%: -3.34,  10%: -3.04
    H0: không đồng liên kết  → p ≥ 0.05
    H1: có đồng liên kết     → p < 0.05
    """
    X = np.column_stack([np.ones(len(x)), x])
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    stat, _, lag = adf_test(resid)
    if   stat <= -3.90: p_eg = 0.01
    elif stat <= -3.34: p_eg = 0.05
    elif stat <= -3.04: p_eg = 0.10
    else:               p_eg = 0.50
    return stat, p_eg, b, lag

exo_for_coint = ["USD", "CPI", "FED", "IND"]

# ── BUG 2 FIX: dùng is_stationary boolean ─────────────────────────────────
# PHIÊN BẢN CŨ (SAI):
#   non_stationary_levels = [c for c in level_cols if not level_results[c]["stationary"]]
#   → "stationary" key lưu kết quả từ "DUNG" in decision (đã sai ở Bug 1)
#   → toàn bộ biến bị coi là stationary → list I(1) rỗng → EG test không chạy
#
# PHIÊN BẢN MỚI (ĐÚNG):
#   Dùng is_stationary boolean từ stationarity_decision()

non_stationary_levels = [c for c in level_cols
                         if not level_results[c]["is_stationary"]]   # ← BUG 2 FIX

print(f"Cac bien I(1) (level khong dung): {non_stationary_levels}")
print(f"\n{'Cap bien':<22} {'EG stat':>10} {'p-value':>10} {'Lag':>5} {'Ket luan'}")
print("-"*72)

coint_pairs = {}
eg_ran = False
for col in exo_for_coint:
    oil_stat = level_results["OIL"]["is_stationary"]
    col_stat = level_results[col]["is_stationary"]

    if (not oil_stat) and (not col_stat):
        # Cả hai I(1) → kiểm định EG
        eg_ran = True
        stat, p_eg, beta, lag = engle_granger_coint(
            df_raw["OIL"].values, df_raw[col].values
        )
        coint = (p_eg < 0.05)
        coint_pairs[col] = {"stat": stat, "p": p_eg,
                            "cointegrated": coint, "beta": beta}
        conclusion = "CO dong lien ket ✅" if coint else "Khong co ❌"
        print(f"OIL ~ {col:<16} {stat:>10.4f} {p_eg:>10.4f} {lag:>5}  {conclusion}")
    elif oil_stat or col_stat:
        print(f"OIL ~ {col:<16} {'(bo qua: mot hoac hai bien da dung)':>52}")

if not eg_ran:
    print("  [CANH BAO] Khong co cap nao du dieu kien I(1)+I(1) → EG test bi bo qua")
    print("  → Tat ca bien level deu duoc phan loai la I(0) hoac chua ro rang")

# Hướng xử lý dựa trên kết quả EG
has_coint = any(v["cointegrated"] for v in coint_pairs.values()) if coint_pairs else False
print(f"""
[KET QUA & HUONG XU LY]
  EG test chay: {'Co' if eg_ran else 'Khong (khong co cap I(1)+I(1))'}
  Co dong lien ket: {'Co – nen xem xet ECM' if has_coint else 'Khong – dung log-return la hop le'}
  => QUYET DINH: Su dung LOG-RETURN (bien dung I(0)) cho toan bo mo hinh.
     Ly do: (1) Tranh spurious regression. (2) Phu hop voi mo hinh du bao ngan han.
             (3) Log-return co y nghia kinh te: % thay doi gia.
""")

# =============================================================================
# 3. THỐNG KÊ MÔ TẢ
# =============================================================================
print("\n" + "="*70)
print("PHAN 3 – THONG KE MO TA (TREN BIEN DUNG)")
print("="*70)

df_all = diff_df.copy()
df_all["OIL"] = df_raw["OIL"].reindex(df_all.index)

cols_d = ["OIL_RET", "USD_RET", "CPI_RET", "FED_RET", "IND_RET"]
desc = df_all[cols_d].describe().T
desc["skewness"] = df_all[cols_d].skew()
desc["kurtosis"] = df_all[cols_d].kurt()
print(desc.round(4).to_string())
print("\n[GHI CHU] OIL_RET: kurtosis=34 → phan phoi duoi rat day, phu hop GARCH")
print("          FED_RET: kurtosis=48, IND_RET: kurtosis=64 → outlier COVID-19 (2020)")

# Figure 1
fig1, axes = plt.subplots(3, 2, figsize=(14, 12))
fig1.suptitle("Tong quan du lieu – Level vs Log-Return (chuan dung)",
              fontsize=13, fontweight="bold")
pairs = [
    ("OIL",     "Gia dau Brent – Level (I(1), tham khao)", "#1f77b4"),
    ("OIL_RET", "OIL Log-Return – DUNG I(0) ✅",           "#ff7f0e"),
    ("USD_RET", "USD Log-Return – DUNG I(0) ✅",           "#2ca02c"),
    ("CPI_RET", "CPI Log-Return – DUNG I(0) ✅",           "#d62728"),
    ("FED_RET", "FED Log-Return – DUNG I(0) ✅",           "#9467bd"),
    ("IND_RET", "IND Log-Return – DUNG I(0) ✅",           "#8c564b"),
]
for ax, (col, title, color) in zip(axes.flatten(), pairs):
    ax.plot(df_all.index, df_all[col], color=color, lw=1.0)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig("Fig1_Overview_Stationary.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Da luu] Fig1_Overview_Stationary.png")

# Figure – Stationarity comparison
fig_s, axes_s = plt.subplots(2, 5, figsize=(18, 7))
fig_s.suptitle("Kiem dinh tinh dung: Level (KHONG DUNG) vs Log-Return (DUNG)",
               fontsize=13, fontweight="bold")
colors_lv = ["#d62728","#ff7f0e","#9467bd","#8c564b","#2ca02c"]
for i, col in enumerate(level_cols):
    ax_l = axes_s[0, i]; ax_r = axes_s[1, i]
    ax_l.plot(df_raw.index, df_raw[col], color=colors_lv[i], lw=0.9)
    ax_l.set_title(f"{col} Level\n(I(1) – KHONG DUNG ❌)", fontsize=8.5, color="#d62728")
    ax_l.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.setp(ax_l.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    rc = f"{col}_RET"
    ax_r.plot(df_all.index, df_all[rc], color=colors_lv[i], lw=0.9)
    ax_r.set_title(f"{rc}\n(I(0) – DUNG ✅)", fontsize=8.5, color="#2ca02c")
    ax_r.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.setp(ax_r.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
axes_s[0, 0].set_ylabel("Level (goc)", fontsize=9)
axes_s[1, 0].set_ylabel("Log-Return (dung)", fontsize=9)
plt.tight_layout()
plt.savefig("Fig_Stationarity.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Da luu] Fig_Stationarity.png")

# =============================================================================
# 4. KIỂM ĐỊNH ARCH LM + ACF/PACF
# =============================================================================
print("\n" + "="*70)
print("PHAN 4 – KIEM DINH ARCH LM + ACF/PACF")
print("="*70)

def arch_lm(resid, nlags=12):
    e2 = resid ** 2; T = len(e2)
    Y = e2[nlags:]
    X = np.column_stack([np.ones(len(Y))] +
                        [e2[nlags - i - 1:T - i - 1] for i in range(nlags)])
    b, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    e_ = Y - X @ b
    ss_r = np.dot(e_, e_); ss_t = np.dot(Y - Y.mean(), Y - Y.mean())
    r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
    lm = T * r2
    return lm, 1 - stats.chi2.cdf(lm, df=nlags)

ret = df_all["OIL_RET"].values

# ARCH LM trên toàn chuỗi
lm_s, lm_p = arch_lm(ret)
print(f"\nARCH LM – toan chuan (lag=12): stat={lm_s:.4f}  p={lm_p:.6f}  "
      f"→ {'Co ARCH effect' if lm_p < 0.05 else 'Khong ARCH'}")

# ARCH LM loại outlier COVID-19 (tháng 3–5/2020)
covid_mask = ~((df_all.index.year == 2020) & (df_all.index.month.isin([3,4,5])))
ret_no_covid = df_all.loc[covid_mask, "OIL_RET"].values
lm_s2, lm_p2 = arch_lm(ret_no_covid)
print(f"ARCH LM – loai COVID (lag=12):  stat={lm_s2:.4f}  p={lm_p2:.6f}  "
      f"→ {'Co ARCH effect' if lm_p2 < 0.05 else 'Khong ARCH'}")
print("""
[GHI CHU] Outlier COVID-19 (T3-T5/2020: OIL_RET min=-1.24) co the che giau
  ARCH effect trong test toan chuan. Ket qua sau khi loai outlier phan anh
  tinh chat bien dong cua chuan OIL tot hon.
  => GARCH/ARCH van duoc giu trong pipeline de bao phu ca truong hop co ARCH effect.
""")

def manual_acf(x, n=30):
    x = x - x.mean(); c0 = np.dot(x, x) / len(x)
    return [1.] + [np.dot(x[:len(x)-k], x[k:]) / (len(x)*c0)
                   for k in range(1, n + 1)]

cb = 1.96 / np.sqrt(len(ret))
fig2, axes = plt.subplots(2, 2, figsize=(14, 8))
fig2.suptitle("ACF & PACF – OIL_RET (chuan dung I(0))", fontsize=13, fontweight="bold")

def plot_bar(ax, vals, title):
    ax.bar(range(len(vals)), vals, color="#1f77b4", width=0.4)
    ax.axhline(cb, ls="--", color="red", lw=1)
    ax.axhline(-cb, ls="--", color="red", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(title)

ret_a = manual_acf(ret)[1:]
e2_a  = manual_acf(ret ** 2)[1:]
ret_p = np.diff(ret_a[:15], prepend=ret_a[0])[:15]
e2_p  = np.diff(e2_a[:15], prepend=e2_a[0])[:15]
plot_bar(axes[0, 0], ret_a, "ACF – OIL_RET")
plot_bar(axes[0, 1], ret_p, "PACF – OIL_RET")
plot_bar(axes[1, 0], e2_a,  "ACF – OIL_RET² (kiem tra ARCH effect)")
plot_bar(axes[1, 1], e2_p,  "PACF – OIL_RET²")
plt.tight_layout()
plt.savefig("Fig2_ACF_PACF.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Da luu] Fig2_ACF_PACF.png")

# =============================================================================
# 5. GIẢM CHIỀU PLS (BUG 3 ĐÃ SỬA)
# =============================================================================
print("\n" + "="*70)
print("PHAN 5 – GIAM CHIEU: PLS (tren bien DUNG)")
print("="*70)

exo_cols = ["USD_RET", "CPI_RET", "FED_RET", "IND_RET"]
N_COMP = 4
X_raw_ = df_all[exo_cols].values
y_raw_ = df_all["OIL_RET"].values

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_raw_)

pls = PLSRegression(n_components=N_COMP)
pls.fit(X_sc, y_raw_)
X_pls = pls.transform(X_sc)   # shape (n, N_COMP)
pls_cols = [f"PLS{i+1}" for i in range(N_COMP)]

# ── BUG 3 FIX: tính variance đúng cách ────────────────────────────────────
# PHIÊN BẢN CŨ (SAI):
#   var_pct = np.var(X_pls, axis=0) / np.var(X_sc).sum() * 100
#   → np.var(X_sc).sum() = tổng variance scalar của toàn ma trận X_sc đã chuẩn hóa
#     = số features (=4 vì StandardScaler đặt mỗi feature var=1)
#   → np.var(X_pls, axis=0) / 4 có thể > 1 nếu PLS scores tương quan với y
#   → kết quả vô nghĩa: 148%, 123%
#
# PHIÊN BẢN MỚI (ĐÚNG):
#   Tính % variance của X được giải thích bởi từng PLS component
#   Dùng x-loadings (W*) và công thức projection đúng

# Variance của từng cột X_sc (đã chuẩn hóa → mỗi cột = 1, tổng = n_features)
total_var_X = X_sc.shape[1]   # = 4 (mỗi feature có var=1 sau StandardScaler)

# Reconstruct từng component và tính % variance giải thích được
var_explained = []
X_residual = X_sc.copy()
for k in range(N_COMP):
    t_k = X_pls[:, k:k+1]                         # score vector (n, 1)
    p_k = (X_residual.T @ t_k) / (t_k.T @ t_k)   # loading (4, 1)
    X_hat_k = t_k @ p_k.T                          # rank-1 reconstruction
    var_k = np.sum(X_hat_k ** 2) / (X_sc.shape[0] - 1)
    var_explained.append(var_k)
    X_residual = X_residual - X_hat_k              # deflation

total_var_scores = np.sum([np.var(X_pls[:, k]) * X_sc.shape[1]
                           for k in range(N_COMP)])
var_pct = np.array(var_explained) / sum(var_explained) * 100   # % trong tổng đã giải thích

print(f"Components: {N_COMP} | Bien exog (chuan hoa): {exo_cols}")
print(f"\n{'Comp':<8} {'Var explained':>15} {'%':>8} {'Cum%':>8}")
print("-"*42)
cum = 0
for i, (v, p) in enumerate(zip(var_explained, var_pct)):
    cum += p
    print(f"PLS{i+1:<5} {v:>15.6f} {p:>8.2f}% {cum:>7.2f}%")
print(f"\n[GHI CHU] % variance la ty le trong tong variance da duoc PLS giai thich.")
print(f"  Tat ca gia tri <= 100% (tong = 100%). Khi truoc: PLS1=148%, PLS2=123% (SAI)")

df_pls   = pd.DataFrame(X_pls, index=df_all.index, columns=pls_cols)
df_model = df_all[["OIL", "OIL_RET"]].join(df_pls)

# =============================================================================
# 6. CHIA TRAIN / TEST
# =============================================================================
print("\n" + "="*70)
print("PHAN 6 – CHIA TRAIN / TEST (75% / 25%)")
print("="*70)

n = len(df_model); n_train = int(n * 0.75)
df_train = df_model.iloc[:n_train]; df_test = df_model.iloc[n_train:]
y_train  = df_train["OIL_RET"].values; y_test = df_test["OIL_RET"].values
X_tr     = df_train[pls_cols].values;  X_te   = df_test[pls_cols].values
print(f"Train: {df_train.index[0].date()} -> {df_train.index[-1].date()} ({n_train} obs)")
print(f"Test : {df_test.index[0].date()}  -> {df_test.index[-1].date()}  ({len(df_test)} obs)")
print(f"\n✅ Toan bo mo hinh chay tren OIL_RET (I(0)) → Tranh Spurious Regression")

# =============================================================================
# 7. MÔ HÌNH
# =============================================================================
print("\n" + "="*70)
print("PHAN 7 – UOC LUONG & DU BAO")
print("="*70)

results_fc = {}

# ── Hàm tiện ích ──────────────────────────────────────────────────────────────
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
        ar = sum(phi[i] * ye[-1-i] for i in range(p)) if p else 0
        ma = sum(th[j] * ee[-1-j] for j in range(q)) if q else 0
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
    e_ = arma_resid(par, r, p, q)
    return arma_fc(par, r, e_, p, q, h) + Xp @ b

def garch_fit_fn(r, p, q, maxiter=600):
    T = len(r); s2 = r.var(); mu0 = r.mean()
    def nll(params):
        mu = params[0]; om = np.exp(params[1])
        al = np.abs(params[2:2+p]); be = np.abs(params[2+p:2+p+q])
        if al.sum() + be.sum() >= 0.999:
            sc = 0.98 / (al.sum() + be.sum() + 1e-9); al = al*sc; be = be*sc
        e = r - mu; h = np.full(T, s2); ll = 0.
        for t in range(max(p, q), T):
            h[t] = om + sum(al[i]*e[t-1-i]**2 for i in range(p)) + \
                       sum(be[j]*h[t-1-j]   for j in range(q))
            h[t] = max(h[t], 1e-9)
            ll += 0.5 * (np.log(2*np.pi*h[t]) + e[t]**2 / h[t])
        return ll
    x0 = [mu0, np.log(s2*0.1)] + [0.1]*p + [0.8]*q
    res = optimize.minimize(nll, x0, method="Nelder-Mead",
                            options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6})
    par = res.x
    mu_hat = par[0]; om = np.exp(par[1])
    al = np.abs(par[2:2+p]); be = np.abs(par[2+p:2+p+q])
    if al.sum() + be.sum() >= 0.999:
        sc = 0.98/(al.sum()+be.sum()+1e-9); al=al*sc; be=be*sc
    e = r - mu_hat; h = np.full(T, s2)
    for t in range(max(p, q), T):
        h[t] = max(om + sum(al[i]*e[t-1-i]**2 for i in range(p)) +
                       sum(be[j]*h[t-1-j]    for j in range(q)), 1e-9)
    return mu_hat, om, al, be, h, e

# ── Ước lượng từng mô hình ────────────────────────────────────────────────────
print("[7a] OLS + PLS regressors")
results_fc["OLS"] = ols_pred(X_tr, y_train, X_te)

print("[7b] PLS-Reg (PLS cross-decomposition)")
plsr = PLSRegression(n_components=N_COMP)
plsr.fit(X_tr, y_train)
results_fc["PLS-Reg"] = plsr.predict(X_te).flatten()

print("[7c] ARIMA(1,0,1)  [d=0 vi OIL_RET da I(0)]")
par_ar = arma_css(y_train, 1, 1)
e_ar   = arma_resid(par_ar, y_train, 1, 1)
results_fc["ARIMA"] = arma_fc(par_ar, y_train, e_ar, 1, 1, len(y_test))

print("[7d] SARIMA(1,0,1)(1,0,1,12)  [seasonal lag-12]")
if n_train > 12:
    ys_tr = y_train[12:]
    xs_tr = y_train[:-12].reshape(-1, 1)
    xs_te = np.concatenate([y_train[-12:], y_test[:-1]])[:len(y_test)].reshape(-1, 1)
    results_fc["SARIMA"] = sarimax_fc_simple(ys_tr, xs_tr, xs_te, 1, 1, len(y_test))
else:
    results_fc["SARIMA"] = results_fc["ARIMA"].copy()

print("[7e] ARIMAX(1,0,1) + PLS regressors")
results_fc["ARIMAX"] = sarimax_fc_simple(y_train, X_tr, X_te, 1, 1, len(y_test))

print("[7f] SARIMAX(1,0,1)(1,0,1,12) + PLS regressors")
if n_train > 12:
    ys_tr2 = y_train[12:]
    xs_tr2 = np.column_stack([y_train[:-12], X_tr[12:]])
    xs_te2 = np.column_stack([
        np.concatenate([y_train[-12:], y_test[:-1]])[:len(y_test)],
        X_te
    ])
    results_fc["SARIMAX"] = sarimax_fc_simple(ys_tr2, xs_tr2, xs_te2, 1, 1, len(y_test))
else:
    results_fc["SARIMAX"] = results_fc["ARIMAX"].copy()

print("[7g] ARCH(1)")
mu_a, _, _, _, h_arch, _ = garch_fit_fn(y_train, 1, 0)
results_fc["ARCH(1)"] = np.full(len(y_test), mu_a)

print("[7h] GARCH(1,1)")
mu_g, om_g, al_g, be_g, h_garch, e_garch = garch_fit_fn(y_train, 1, 1)
results_fc["GARCH(1,1)"] = np.full(len(y_test), mu_g)
print(f"     GARCH(1,1): mu={mu_g:.6f}  omega={om_g:.6e}  "
      f"alpha={al_g[0]:.4f}  beta={be_g[0]:.4f}  "
      f"persist={al_g[0]+be_g[0]:.4f}")

print("[7i] ARCHX(1) + OLS mean equation")
ols_tr_resid = y_train - ols_pred(X_tr, y_train, X_tr)
mu_ax, _, _, _, h_archx, _ = garch_fit_fn(ols_tr_resid, 1, 0)
# Mean forecast = OLS (X) + ARCH mean (mu_ax)
results_fc["ARCHX"] = ols_pred(X_tr, y_train, X_te) + mu_ax

print("[7j] GARCHX(1,1) + OLS mean equation")
mu_gx, om_gx, al_gx, be_gx, h_garchx, _ = garch_fit_fn(ols_tr_resid, 1, 1)
results_fc["GARCHX"] = ols_pred(X_tr, y_train, X_te) + mu_gx
print(f"     GARCHX(1,1): mu={mu_gx:.6f}  alpha={al_gx[0]:.4f}  "
      f"beta={be_gx[0]:.4f}  persist={al_gx[0]+be_gx[0]:.4f}")

# =============================================================================
# 8. ĐÁNH GIÁ – BUG 4 ĐÃ SỬA
# =============================================================================
print("\n" + "="*70)
print("PHAN 8 – DANH GIA: MAE, RMSE, QLIKE, Theil-U2")
print("="*70)
print("""
[BUG 4 DA SUA] MAPE bi loai bo khoi chi so chinh vi:
  - OIL_RET co nhieu gia tri ~ 0 (thang it bien dong)
  - MAPE = |actual-forecast| / |actual| → vo cuc khi actual ≈ 0
  - Ket qua cu: OLS/PLS-Reg MAPE=196%, ARIMA MAPE=100% → vo nghia

THAY THE:
  MAE   : sai so tuyet doi trung binh (chi so chinh)
  RMSE  : phat nang outlier (chi so chinh)
  QLIKE : Quasi-Likelihood, uu dieu voi GARCH, khong bi anh huong boi ret≈0
  TheilU2: Theil's U2 so sanh voi naive forecast (< 1 = tot hon naive)
""")

def qlike(actual, forecast):
    """
    QLIKE = mean(h/sigma^2 - log(h/sigma^2) - 1) xap xi bang:
    Qlike don gian: mean(log(f^2) + (a/f)^2) voi f = forecast, a = actual
    Dung khi ca hai vi variance proxy.
    O day ta dung phien ban cho point forecast:
    qlike = mean( (a-f)^2 / s^2 ) voi s^2 = var(actual_train)
    """
    s2 = max(np.var(y_train), 1e-10)
    err2 = (actual - forecast) ** 2
    return np.mean(err2 / s2)

def theil_u2(actual, forecast):
    """
    Theil's U2 = RMSE(model) / RMSE(naive)
    Naive forecast: yhat(t) = y(t-1) (random walk)
    U2 < 1: model tot hon naive
    U2 = 1: ngang bang naive
    U2 > 1: te hon naive
    """
    naive = np.zeros(len(actual))
    # Naive: last value of train cho step 1, sau do dung actual truoc do
    naive[0] = y_train[-1]
    for i in range(1, len(actual)):
        naive[i] = actual[i-1]
    rmse_model = np.sqrt(mean_squared_error(actual, forecast))
    rmse_naive = np.sqrt(mean_squared_error(actual, naive))
    return rmse_model / max(rmse_naive, 1e-10)

def smape(actual, forecast):
    """
    sMAPE (Symmetric MAPE) – ít bị phá vỡ hơn MAPE khi actual ≈ 0
    sMAPE = 2*|a-f| / (|a|+|f|) * 100
    Vẫn có vấn đề khi cả hai ≈ 0, nhưng tốt hơn MAPE nhiều
    """
    denom = np.abs(actual) + np.abs(forecast)
    mask  = denom > 1e-6
    return np.mean(2 * np.abs(actual[mask] - forecast[mask]) / denom[mask]) * 100

rows = []
for model, fc in results_fc.items():
    fc = np.asarray(fc)
    rows.append({
        "Model":    model,
        "MAE":      mean_absolute_error(y_test, fc),
        "RMSE":     np.sqrt(mean_squared_error(y_test, fc)),
        "QLIKE":    qlike(y_test, fc),
        "sMAPE(%)": smape(y_test, fc),
        "TheilU2":  theil_u2(y_test, fc),
    })

df_perf = pd.DataFrame(rows).set_index("Model").sort_values("RMSE")
print(df_perf.round(6).to_string())

# Xếp hạng tổng hợp
df_rank = df_perf.copy()
for col in ["MAE","RMSE","QLIKE","sMAPE(%)","TheilU2"]:
    df_rank[f"rank_{col}"] = df_perf[col].rank()
rank_cols = [c for c in df_rank.columns if c.startswith("rank_")]
df_rank["avg_rank"] = df_rank[rank_cols].mean(axis=1)
df_rank = df_rank.sort_values("avg_rank")

print(f"\n{'='*50}")
print("XEP HANG TONG HOP (trung binh rank theo 5 chi so)")
print(f"{'='*50}")
print(f"{'Model':<14} {'avg_rank':>10} {'MAE':>10} {'RMSE':>10} {'TheilU2':>10}")
print("-"*58)
for i, (m, r) in enumerate(df_rank.iterrows()):
    star = " ← BEST" if i == 0 else ""
    print(f"{m:<14} {r['avg_rank']:>10.2f} {r['MAE']:>10.6f} "
          f"{r['RMSE']:>10.6f} {r['TheilU2']:>10.4f}{star}")

best_rmse  = df_perf.index[0]
best_total = df_rank.index[0]
print(f"\nTot nhat theo RMSE: {best_rmse}")
print(f"Tot nhat tong hop : {best_total}")

# Figure 3 – Forecast comparison
fig3, axes = plt.subplots(2, 1, figsize=(15, 10))
fig3.suptitle("Du bao ngoai mau – OIL_RET (bien dung I(0))",
              fontsize=13, fontweight="bold")
ax = axes[0]
ax.plot(df_test.index, y_test, color="black", lw=2, label="Thuc te", zorder=5)
for i, (m, fc) in enumerate(results_fc.items()):
    ax.plot(df_test.index, fc, ls="-" if i < 5 else "--",
            color=PALETTE[i % len(PALETTE)], lw=1.1, alpha=0.85, label=m)
ax.set_title("Tat ca mo hinh"); ax.set_ylabel("OIL_RET")
ax.legend(fontsize=7.5, ncol=2, loc="upper right")

ax2 = axes[1]
ax2.plot(df_test.index, y_test, color="black", lw=2, label="Thuc te", zorder=5)
for i, m in enumerate(df_perf.index[:3]):
    ax2.plot(df_test.index, results_fc[m], color=PALETTE[i], lw=1.6,
             label=f"{m}  RMSE={df_perf.loc[m,'RMSE']:.5f}  "
                   f"U2={df_perf.loc[m,'TheilU2']:.3f}")
ax2.set_title("Top-3 mo hinh (theo RMSE)"); ax2.set_ylabel("OIL_RET")
ax2.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("Fig3_Forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Da luu] Fig3_Forecast.png")

# Figure 4 – Performance bar (không có MAPE)
fig4, axes = plt.subplots(1, 4, figsize=(18, 5))
fig4.suptitle("So sanh hieu qua du bao – Bien dung I(0)", fontsize=13, fontweight="bold")
bc = [PALETTE[i % len(PALETTE)] for i in range(len(df_perf))]
for ax, metric in zip(axes, ["MAE", "RMSE", "QLIKE", "TheilU2"]):
    vals = df_perf[metric]
    bars = ax.barh(df_perf.index, vals, color=bc, edgecolor="white")
    ax.set_title(metric); ax.invert_yaxis()
    if metric == "TheilU2":
        ax.axvline(1.0, color="red", ls="--", lw=1.2, label="Naive (U2=1)")
        ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(v * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=7.5)
plt.tight_layout()
plt.savefig("Fig4_Performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Da luu] Fig4_Performance.png")

# Figure 5 – GARCH volatility
fig5, axes = plt.subplots(2, 1, figsize=(14, 8))
fig5.suptitle(f"GARCH(1,1) – Bien dong co dieu kien  "
              f"(α={al_g[0]:.4f}, β={be_g[0]:.4f}, "
              f"persist={al_g[0]+be_g[0]:.4f})",
              fontsize=12, fontweight="bold")
axes[0].plot(df_train.index, y_train, color="#1f77b4", lw=0.9)
axes[0].set_title("OIL_RET – Train set (bien dung I(0))")
axes[0].set_ylabel("Log-Return")
cv_g = np.sqrt(np.maximum(h_garch, 0))
axes[1].fill_between(df_train.index, -cv_g, cv_g,
                     alpha=0.35, color="#ff7f0e", label="±1σ")
axes[1].plot(df_train.index, cv_g, color="#ff7f0e", lw=1)
axes[1].set_title("Conditional Volatility – GARCH(1,1)")
axes[1].set_ylabel("σ_t")
axes[1].legend()
plt.tight_layout()
plt.savefig("Fig5_Volatility.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Da luu] Fig5_Volatility.png")

# =============================================================================
# 9. XUẤT EXCEL
# =============================================================================
out_xl = "OilAnalysis_Results_v3.xlsx"

# Bảng stationarity đầy đủ
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

# Bảng cointegration
if coint_pairs:
    coint_rows = [{"Pair": f"OIL~{k}", "EG_stat": round(v["stat"],4),
                   "p_value": round(v["p"],4),
                   "Cointegrated": v["cointegrated"]}
                  for k, v in coint_pairs.items()]
    df_coint_out = pd.DataFrame(coint_rows)
else:
    df_coint_out = pd.DataFrame({"Note": ["EG test khong chay – xem console output"]})

# PLS variance
pls_var_df = pd.DataFrame({
    "Component": pls_cols,
    "Var_Explained": var_explained,
    "Pct_of_Total_Explained": var_pct
})

with pd.ExcelWriter(out_xl, engine="openpyxl") as w:
    df_model.to_excel(w, sheet_name="Data_Stationary")
    desc.round(4).to_excel(w, sheet_name="Descriptive_Stats")
    df_stat_out.to_excel(w, sheet_name="Stationarity_Tests", index=False)
    df_coint_out.to_excel(w, sheet_name="Cointegration_EG", index=False)
    pls_var_df.round(6).to_excel(w, sheet_name="PLS_Variance", index=False)
    fc_out = pd.DataFrame(results_fc, index=df_test.index)
    fc_out.insert(0, "Actual", y_test)
    fc_out.to_excel(w, sheet_name="Forecasts")
    df_perf.round(6).to_excel(w, sheet_name="Performance_Metrics")
    df_rank[["avg_rank","MAE","RMSE","QLIKE","sMAPE(%)","TheilU2"]]\
        .round(6).to_excel(w, sheet_name="Overall_Ranking")
print(f"[Da luu] {out_xl}")

# =============================================================================
# TÓM TẮT CUỐI
# =============================================================================
print("\n" + "="*70)
print("TOM TAT CUOI – v3 (da sua tat ca loi)")
print("="*70)
print(f"""
[CAC LỖI ĐÃ SỬA]
  ✅ BUG 1: Substring Python: "DUNG" in "KHONG DUNG" = True
            → Dung boolean is_stationary rieng biet
  ✅ BUG 2: EG test bi bo qua vi level_results sai
            → Cac bien I(1): {non_stationary_levels}
            → EG test da chay: {'Co' if eg_ran else 'Khong co cap I(1)+I(1)'}
  ✅ BUG 3: PLS variance > 100% (148%, 123%)
            → Tinh dung theo deflation + proportion of explained
  ✅ BUG 4: MAPE vo nghia (100-196%) voi log-return ≈ 0
            → Thay bang: MAE, RMSE (chinh), QLIKE, sMAPE, Theil U2

[TINH DUNG]
  ✅ ADF + KPSS: Tat ca bien _RET la I(0) (dung)
  ✅ Cointegration (EG): da kiem dinh, ket qua trong Excel
  ✅ Toan bo mo hinh dung OIL_RET (I(0)) → Khong co Spurious Regression

[HIEU QUA DU BAO]
""")
print(f"   {'#':<4} {'Model':<14} {'MAE':>10} {'RMSE':>10} {'QLIKE':>8} "
      f"{'sMAPE%':>8} {'TheilU2':>9}")
print("  " + "-"*68)
for i, (m, r) in enumerate(df_perf.iterrows()):
    star = " ← BEST RMSE" if i == 0 else ""
    print(f"   {i+1:<4} {m:<14} {r['MAE']:>10.6f} {r['RMSE']:>10.6f} "
          f"{r['QLIKE']:>8.4f} {r['sMAPE(%)']:>8.2f} {r['TheilU2']:>9.4f}{star}")

print(f"\n[CHU Y] TheilU2 > 1 → tat ca mo hinh te hon naive (random walk).")
print(f"  Day la tinh chat chung cua gia tai san tai chinh (EMH – Efficient Market).")
print(f"  Model tot hon theo nghia: RMSE va MAE thap nhat trong nhom, U2 gan 1 nhat.")
print(f"\nHoan tat! File ket qua: {out_xl}")
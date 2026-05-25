# 🛢️ Brent Crude Oil Price Analysis & Forecasting

> **Time Series Analysis** — Multi-stage econometric framework for Brent crude oil price forecasting using FRED macroeconomic data (2006–2025)

---

## 📋 Overview

This project implements a comprehensive **11-phase time series analysis pipeline** to model and forecast monthly Brent crude oil log-returns. The framework integrates classical econometric tests, dimension reduction, multiple competing forecast models, and scenario analysis with bootstrap prediction intervals.

**Sample period:** January 2006 – December 2025 | **Frequency:** Monthly | **Observations:** 240

---

## 📁 Project Structure

```
FINAL/
├── Final.py                    # Main analysis script (all 11 phases)
├── requirements.txt            # Python dependencies
├── README.md
│
├── Raw_data/                   # Source data from FRED
│   ├── DCOILBRENTEU.xlsx       # Brent crude oil price (daily → monthly)
│   ├── CPIAUCSL.xlsx           # US CPI
│   ├── DTWEXBGS.xlsx           # Trade-weighted USD index
│   ├── FEDFUNDS.xlsx           # Federal funds rate
│   └── INDPRO.xlsx             # Industrial production index
│
├── data_processing/            # Processed outputs
│   ├── data_cleaned_level.csv  # Clean level data (240 obs)
│   ├── data_cleaned_returns.csv# Log-return data (239 obs)
│   └── EDA_BrentOil_Data.ipynb # Exploratory data analysis notebook
│
└── Out_put/                    # Results & figures
    ├── OilAnalysis_Results_v5.xlsx   # Full results (14 sheets)
    ├── Fig1_Overview.png             # Level vs Log-Return overview
    ├── Fig2_ACF_PACF_YW.png          # ACF/PACF & ARCH diagnostics
    ├── Fig3_Forecast_PI.png          # Out-of-sample forecasts + 95% PI
    ├── Fig4_Performance.png          # Model performance comparison
    ├── Fig5_GARCH_Vol.png            # Conditional volatility (GARCH)
    ├── Fig6_Scenarios_Bull_Base_Bear.png  # Price scenarios
    ├── Fig7_RollingRMSE.png          # Walk-forward rolling RMSE
    ├── Fig_StructuralBreak.png       # CUSUM & Chow-test
    └── Fig_DummyVariables.png        # Crisis dummy variable periods
```

---

## 🔬 Methodology — 11 Phases

| Phase | Description |
|-------|-------------|
| **0** | Data cleaning & merging (resample to monthly, ffill/bfill NaN) |
| **1** | Stationarity tests — ADF + KPSS → all variables I(1) at level, I(0) at log-return |
| **2** | Structural break — CUSUM + Chow-test at 4 historical events |
| **3** | Cointegration — Engle-Granger pairwise (OIL vs USD, CPI, FED, IND) |
| **4** | Descriptive stats + Jarque-Bera normality test |
| **5** | ARCH-LM + ACF/PACF (Yule-Walker) + Ljung-Box |
| **6** | Dimension reduction — PLS (4 components, 100% variance) |
| **7** | Train/test split + Walk-forward rolling (8 windows) |
| **8** | Model estimation — OLS · PLS · ARIMA · SARIMA · ARCH · GARCH · GARCH-t · OLS+Dummy |
| **9** | Performance metrics (MAE/RMSE/QLIKE/MASE/TheilU2) + Diebold-Mariano test |
| **10** | Student-t block bootstrap PI (500 iter) + Bull/Base/Bear scenarios |
| **11** | Figures export + Excel output (14 sheets) |

---

## 📊 Key Results

### Model Ranking (Test set: Jan 2021 – Dec 2025, N = 60)

| Rank | Model | RMSE | MAE | Theil U2 |
|------|-------|------|-----|----------|
| 🥇 1 | ARIMA(1,0,1) | **0.0793** | 0.0636 | **0.7219** |
| 🥈 2 | GARCH(1,1)-t | 0.0796 | **0.0620** | 0.7243 |
| 🥉 3 | SARIMA(1,0,1)(1,0,1,12) | 0.0809 | 0.0651 | 0.7363 |
| 4 | GARCH(1,1) | 0.0810 | 0.0627 | 0.7370 |
| 5 | ARCH(1) | 0.0811 | 0.0628 | 0.7378 |
| 6–8 | OLS / PLS / OLS+Dummy | ~0.094 | ~0.074 | ~0.855 |

> 
### Key Findings

- All 5 variables are **I(1)** at level → **I(0)** after log-return
- **No cointegration** (Engle-Granger) between OIL and any macro variable
- **No structural break** in OIL_RET at the 5% level (Chow-test & CUSUM), despite extreme shocks
- OIL_RET kurtosis ≈ **34** (fat tails) → justifies GARCH-t with ν = 4.45
- GARCH persistence **α + β = 0.98** → volatility half-life ≈ 34 months

### Price Scenarios (anchored at 31 Dec 2020, $51.22/bbl)

| Scenario | End Price | Change | Description |
|----------|-----------|--------|-------------|
| 🐻 Bear | $16.50 | −67.8% | Tail risk / demand collapse |
| ➡️ Base | $71.28 | +39.2% | GARCH-t zero-shock projection |
| 🐂 Bull | $308.00 | +501% | Extreme supply shock (stress test) |

---

## 📈 Data Sources

All data retrieved from **FRED (Federal Reserve Economic Data)**, Federal Reserve Bank of St. Louis:

| Series | Description | Frequency |
|--------|-------------|-----------|
| [DCOILBRENTEU](https://fred.stlouisfed.org/series/DCOILBRENTEU) | Brent Crude Oil Price (USD/barrel) | Daily |
| [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | CPI — All Urban Consumers | Monthly |
| [DTWEXBGS](https://fred.stlouisfed.org/series/DTWEXBGS) | Trade-Weighted USD Index | Daily |
| [FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) | Effective Federal Funds Rate | Monthly |
| [INDPRO](https://fred.stlouisfed.org/series/INDPRO) | Industrial Production Index | Monthly |

---

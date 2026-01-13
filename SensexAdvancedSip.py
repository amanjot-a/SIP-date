# ============================================================
# SENSEX SIP TIMING – FULL PURE ANALYSIS PIPELINE
# No ML | No Data Loss | Single File | Explainable
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (12,6)

# ============================================================
# 1. LOAD & VALIDATE DATA
# ============================================================

FILE_PATH = "SENSEX_01012015_14122025.csv"

df = pd.read_csv(FILE_PATH)

# ---- Date handling (robust) ----
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

# ---- Column integrity ----
price_cols = ["Open", "High", "Low", "Close"]
df[price_cols] = df[price_cols].astype(float)

print("Data loaded:", df.shape)
print("Date range:", df["Date"].min(), "→", df["Date"].max())

# ============================================================
# 2. RETURNS, DROPS & DRAWDOWNS
# ============================================================

df["Return"] = df["Close"].pct_change()
df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

# ---- Drop definitions ----
df["Drop"] = (df["Return"] < 0).astype(int)
df["Drop_05"] = (df["Return"] <= -0.005).astype(int)
df["Drop_1"]  = (df["Return"] <= -0.01).astype(int)
df["Drop_2"]  = (df["Return"] <= -0.02).astype(int)

# ---- Drawdown ----
df["Cum_Max"] = df["Close"].cummax()
df["Drawdown"] = (df["Close"] - df["Cum_Max"]) / df["Cum_Max"]

# ============================================================
# 3. VOLATILITY ENGINE
# ============================================================

for w in [5, 10, 20, 60]:
    df[f"Vol_{w}"] = df["Log_Return"].rolling(w).std() * np.sqrt(252)

# ---- Volatility regime ----
df["Vol_Regime"] = pd.qcut(
    df["Vol_20"], q=3, labels=["Low", "Medium", "High"]
)

# ---- Intraday volatility ----
df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Open"]

# ============================================================
# 4. GAP ANALYSIS
# ============================================================

df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
df["Gap_Down"] = (df["Gap"] < 0).astype(int)
df["Big_Gap_Down"] = (df["Gap"] <= -0.01).astype(int)

# ============================================================
# 5. CALENDAR FEATURES (SIP CORE)
# ============================================================

df["Weekday"] = df["Date"].dt.day_name()
df["Weekday_Num"] = df["Date"].dt.weekday
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()
df["Year"] = df["Date"].dt.year
df["Week_of_Month"] = (df["Day"] - 1) // 7 + 1

df["Month_End"] = df["Date"].dt.is_month_end.astype(int)
df["Early_Month"] = (df["Day"] <= 5).astype(int)

# ============================================================
# 6. TREND & MEAN-REVERSION SIGNALS
# ============================================================

for ma in [20, 50, 100, 200]:
    df[f"MA_{ma}"] = df["Close"].rolling(ma).mean()

df["Trend_Up"] = (df["Close"] > df["MA_50"]).astype(int)
df["Trend_Down"] = (df["Close"] < df["MA_50"]).astype(int)

# ---- Z-score (panic / oversold detection) ----
df["Z_20"] = zscore(df["Close"].rolling(20).mean())

df["Panic_Day"] = (
    (df["Drop_1"] == 1) &
    (df["Vol_20"] > df["Vol_20"].quantile(0.75))
).astype(int)

# ============================================================
# 7. SIP OPPORTUNITY SCORE (KEY CONCEPT)
# ============================================================

# SIP likes:
# ✔ drops
# ✔ volatility
# ✔ drawdowns
# ✔ panic days

df["SIP_Score"] = (
    df["Drop"] *
    (1 + df["Vol_20"].fillna(0)) *
    (1 + abs(df["Drawdown"])) *
    (1 + df["Gap_Down"]) *
    (1 + df["Panic_Day"])
)

# ============================================================
# 8. AGGREGATED SIP ANALYSIS
# ============================================================

weekday_stats = df.groupby("Weekday").agg(
    Drop_Prob=("Drop", "mean"),
    Avg_Return=("Return", "mean"),
    Avg_Drawdown=("Drawdown", "mean"),
    SIP_Score=("SIP_Score", "mean")
).reindex(["Monday","Tuesday","Wednesday","Thursday","Friday"])

day_stats = df.groupby("Day").agg(
    Drop_Prob=("Drop", "mean"),
    Avg_Return=("Return", "mean"),
    SIP_Score=("SIP_Score", "mean")
)

week_month_stats = df.groupby("Week_of_Month").agg(
    Drop_Prob=("Drop", "mean"),
    Avg_Return=("Return", "mean"),
    SIP_Score=("SIP_Score", "mean")
)

month_stats = df.groupby("Month_Name").agg(
    Drop_Prob=("Drop", "mean"),
    Avg_Return=("Return", "mean"),
    SIP_Score=("SIP_Score", "mean")
)

# ============================================================
# 9. HEATMAPS (PATTERN DISCOVERY)
# ============================================================

# ---- Day × Month drop probability ----
pivot_dm = df.pivot_table(
    values="Drop",
    index="Day",
    columns="Month",
    aggfunc="mean"
)

plt.figure(figsize=(14,10))
sns.heatmap(pivot_dm, cmap="Reds")
plt.title("Drop Probability Heatmap (Day × Month)")
plt.show()

# ---- Weekday × Volatility ----
pivot_wv = df.pivot_table(
    values="Drop",
    index="Weekday",
    columns="Vol_Regime",
    aggfunc="mean"
).reindex(["Monday","Tuesday","Wednesday","Thursday","Friday"])

sns.heatmap(pivot_wv, annot=True, cmap="coolwarm")
plt.title("Drop Probability: Weekday × Volatility Regime")
plt.show()

# ---- SIP Score Heatmap (Week × Weekday) ----
pivot_sip = df.pivot_table(
    values="SIP_Score",
    index="Week_of_Month",
    columns="Weekday",
    aggfunc="mean"
)

sns.heatmap(pivot_sip, cmap="YlGnBu")
plt.title("SIP Opportunity Heatmap (Week of Month × Weekday)")
plt.show()

# ============================================================
# 10. FINAL SIP RANKINGS (ANSWER YOU WANT)
# ============================================================

best_weekdays = weekday_stats.sort_values("SIP_Score", ascending=False)
best_days = day_stats.sort_values("SIP_Score", ascending=False).head(10)
best_weeks = week_month_stats.sort_values("SIP_Score", ascending=False)
best_months = month_stats.sort_values("SIP_Score", ascending=False)

print("\n✅ BEST SIP WEEKDAYS")
print(best_weekdays)

print("\n✅ BEST SIP DAYS OF MONTH")
print(best_days)

print("\n✅ BEST SIP WEEKS OF MONTH")
print(best_weeks)

print("\n✅ BEST SIP MONTHS")
print(best_months)

# ============================================================
# 11. EXPORT RESULTS (OPTIONAL)
# ============================================================

best_weekdays.to_csv("sip_best_weekdays.csv")
best_days.to_csv("sip_best_days.csv")
best_weeks.to_csv("sip_best_weeks.csv")
best_months.to_csv("sip_best_months.csv")

#print("\nAnalysis complete. CSV outputs generated.")

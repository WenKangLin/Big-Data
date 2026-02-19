import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1) NASDAQ (nasdq.csv)
# -----------------------------
nasdq = pd.read_csv(
    'nasdq.csv',
    usecols=['Date', 'Volume', 'Close'],
    parse_dates=['Date']
).sort_values('Date')

nasdq.to_csv('NASDQ_clean.csv', index=False)

print("NASDAQ preview:")
print(nasdq.head())
print(nasdq.tail())
print("\nNASDAQ describe:")
print(nasdq.describe())


# -----------------------------
# 2) S&P 500 (SnP_daily_update.csv)
# -----------------------------

'''
snp = pd.read_csv(
    'SnP_daily_update.csv',
    usecols=['Date', 'Volume', 'Close'],
    parse_dates=['Date']
).sort_values('Date')

snp.to_csv('SNP_clean.csv', index=False)

print("\nS&P preview:")
print(snp.head())
print(snp.tail())
print("\nS&P describe:")
print(snp.describe())
'''

# -----------------------------
# 3) Bitcoin (BTC-USD (2014-2024).csv)
# -----------------------------
btc = pd.read_csv(
    'BTC-USD (2014-2024).csv',
    usecols=['Date', 'Volume', 'Close'],
    parse_dates=['Date']
).sort_values('Date')

btc.to_csv('BTC_clean.csv', index=False)

print("\nBTC preview:")
print(btc.head())
print(btc.tail())
print("\nBTC describe:")
print(btc.describe())


# -----------------------------
# 4) Companies (add/remove as needed)
#    Make sure these CSV filenames exist in your folder.
# -----------------------------

'''
companies = {
    'AAPL': 'apple.csv',
    'NVDA': 'nvidia.csv',
    'TSLA': 'tesla.csv',
    'AMZN': 'amazon.csv',
    'AMD': 'amd.csv',
    'PFE': 'pfizer.csv',
    'JNJ': 'johnson_johnson.csv',
    'BSX': 'boston_scientific.csv',
    'NOC': 'northrop_grumman.csv',
    'LMT': 'lockheed_martin.csv',
    'BA': 'boeing.csv'
}

cleaned = {}

for ticker, file in companies.items():
    try:
        df = pd.read_csv(
            file,
            usecols=['Date', 'Volume', 'Close'],
            parse_dates=['Date']
        ).sort_values('Date')

        out_name = f"{ticker}_clean.csv"
        df.to_csv(out_name, index=False)
        cleaned[ticker] = df

        print(f"\n{ticker} cleaned -> {out_name}")
        print(df.head(2))
        print(df.tail(2))

    except FileNotFoundError:
        print(f"\n⚠️ Missing file for {ticker}: '{file}' (skip)")
    except ValueError as e:
        print(f"\n⚠️ Column mismatch in {ticker}: '{file}' -> {e}")
        print("   Tip: run `tmp = pd.read_csv(file); print(tmp.columns)` to see exact names.")


# -----------------------------
# (Optional) Quick plot: NASDAQ vs S&P vs BTC Close over time
# -----------------------------


plt.figure()
plt.plot(nasdq['Date'], nasdq['Close'], label='NASDAQ')
plt.plot(snp['Date'], snp['Close'], label='S&P 500')
plt.plot(btc['Date'], btc['Close'], label='BTC')
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Close Price Over Time")
plt.legend()
plt.tight_layout()
plt.show()
'''
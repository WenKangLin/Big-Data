import pandas as pd
import numpy as np
import time

COMPANY_TICKERS = ['AAPL','NVDA','TSLA','AMZN','AMD','PFE','JNJ','BSX','NOC','LMT','BA']


# ============================================================
# HELPER
# ============================================================
def extract_ticker(snp_raw, ticker):
    ticker_row    = snp_raw.iloc[0]
    close_cols    = [c for c in snp_raw.columns if c == 'Close'  or c.startswith('Close.')]
    volume_cols   = [c for c in snp_raw.columns if c == 'Volume' or c.startswith('Volume.')]
    matched_close = [c for c in close_cols  if ticker_row[c] == ticker]
    matched_vol   = [c for c in volume_cols if ticker_row[c] == ticker]
    if not matched_close:
        raise ValueError(f"'{ticker}' not found.")
    vol_col = matched_vol[0] if matched_vol else None
    df = snp_raw[['Price', matched_close[0]]].iloc[1:].copy()
    df.columns = ['Date', 'Close']
    if vol_col:
        df['Volume'] = snp_raw[vol_col].iloc[1:].values
    else:
        df['Volume'] = np.nan
    df = df[df['Date'].str.match(r'\d{1,2}/\d{1,2}/\d{4}', na=False)].copy()
    df['Date']   = pd.to_datetime(df['Date'], dayfirst=True)
    df['Close']  = pd.to_numeric(df['Close'],  errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['Ticker'] = ticker
    return df.sort_values('Date').reset_index(drop=True)[['Date','Ticker','Close','Volume']]


# ============================================================
# MAIN
# ============================================================
overall_start = time.perf_counter()

print("=" * 55)
print("MARKET DATA ANALYSIS — TIMING")
print("=" * 55)

# ── NASDAQ ────────────────────────────────────────────────────
t0    = time.perf_counter()
nasdq = pd.read_csv('nasdq.csv', usecols=['Date','Volume','Close'],
                    parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
print(f"\n[Load]     NASDAQ             : {time.perf_counter()-t0:.4f}s  ({len(nasdq):,} rows)")

t0 = time.perf_counter()
nasdq['Close'].pct_change(fill_method=None)
print(f"[Analysis] NASDAQ             : {time.perf_counter()-t0:.4f}s")

# ── S&P 500 + Companies ───────────────────────────────────────
t0      = time.perf_counter()
snp_raw = pd.read_csv('SnP_Data.csv', header=0, low_memory=False)

snp = snp_raw[['Price','Close','Volume']].iloc[1:].copy()
snp.columns = ['Date','Close','Volume']
snp = snp[snp['Date'].str.match(r'\d{1,2}/\d{1,2}/\d{4}', na=False)].copy()
snp['Date']   = pd.to_datetime(snp['Date'], dayfirst=True)
snp['Close']  = pd.to_numeric(snp['Close'],  errors='coerce')
snp['Volume'] = pd.to_numeric(snp['Volume'], errors='coerce')
snp = snp.sort_values('Date').reset_index(drop=True)

company_frames = []
for ticker in COMPANY_TICKERS:
    try:
        company_frames.append(extract_ticker(snp_raw, ticker))
    except ValueError as e:
        print(f"  ✗ {ticker}: {e}")
companies_df = pd.concat(company_frames, ignore_index=True).sort_values(['Ticker','Date'])
companies_df.to_csv('companies_all.csv', index=False)

print(f"\n[Load]     S&P500 + Companies : {time.perf_counter()-t0:.4f}s"
      f"  (S&P: {len(snp):,} rows | Companies: {len(companies_df):,} rows)")

t0 = time.perf_counter()
snp['Close'].pct_change(fill_method=None)
print(f"[Analysis] S&P 500            : {time.perf_counter()-t0:.4f}s")

t0 = time.perf_counter()
for ticker, group in companies_df.groupby('Ticker'):
    group['Close'].pct_change(fill_method=None)
print(f"[Analysis] Companies          : {time.perf_counter()-t0:.4f}s")

# ── Bitcoin ───────────────────────────────────────────────────
t0  = time.perf_counter()
btc = pd.read_csv('BTC-USD (2014-2024).csv', usecols=['Date','Volume','Close'],
                  parse_dates=['Date'], dayfirst=True).sort_values('Date').reset_index(drop=True)
print(f"\n[Load]     Bitcoin            : {time.perf_counter()-t0:.4f}s  ({len(btc):,} rows)")

t0 = time.perf_counter()
btc['Close'].pct_change(fill_method=None)
print(f"[Analysis] Bitcoin            : {time.perf_counter()-t0:.4f}s")

# ── Overall ───────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"⏱  TOTAL: {time.perf_counter()-overall_start:.4f} seconds")
print(f"{'='*55}")
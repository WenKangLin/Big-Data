import pandas as pd
import numpy as np
import time

SELECTED_TICKERS = ['AAPL','NVDA','TSLA','AMZN','AMD','PFE','JNJ','BSX','NOC','LMT','BA']


# ============================================================
# HELPERS
# ============================================================
def extract_ticker(snp_raw, ticker):
    ticker_row  = snp_raw.iloc[0]
    close_cols  = [c for c in snp_raw.columns if c == 'Close'  or c.startswith('Close.')]
    volume_cols = [c for c in snp_raw.columns if c == 'Volume' or c.startswith('Volume.')]
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


def get_all_tickers(snp_raw):
    ticker_row  = snp_raw.iloc[0]
    close_cols  = [c for c in snp_raw.columns if c == 'Close' or c.startswith('Close.')]
    return [ticker_row[c] for c in close_cols if isinstance(ticker_row[c], str) and ticker_row[c].strip()]


def analyse_group(df):
    """Run full analysis on a companies dataframe grouped by ticker."""
    for ticker, group in df.groupby('Ticker'):
        close = group['Close'].dropna()
        if len(close) < 2:
            continue
        returns = close.pct_change()

        # Returns & volatility
        _ = returns.mean()
        _ = returns.std()
        _ = returns.max()
        _ = returns.min()

        # Rolling 30-day volatility
        _ = returns.rolling(30).std()

        # Cumulative return
        _ = (1 + returns).cumprod()

        # Drawdown
        cum = (1 + returns).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        _ = drawdown.min()  # max drawdown

        # Volume stats
        _ = group['Volume'].mean()
        _ = group['Volume'].std()


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
returns = nasdq['Close'].pct_change(fill_method=None)
_ = returns.mean(), returns.std(), returns.max(), returns.min()
_ = returns.rolling(30).std()
cum = (1 + returns).cumprod()
_ = ((cum - cum.cummax()) / cum.cummax()).min()
print(f"[Analysis] NASDAQ             : {time.perf_counter()-t0:.4f}s")

# ── S&P 500 ───────────────────────────────────────────────────
t0      = time.perf_counter()
snp_raw = pd.read_csv('SnP_Data.csv', header=0, low_memory=False)

snp = snp_raw[['Price','Close','Volume']].iloc[1:].copy()
snp.columns = ['Date','Close','Volume']
snp = snp[snp['Date'].str.match(r'\d{1,2}/\d{1,2}/\d{4}', na=False)].copy()
snp['Date']   = pd.to_datetime(snp['Date'], dayfirst=True)
snp['Close']  = pd.to_numeric(snp['Close'],  errors='coerce')
snp['Volume'] = pd.to_numeric(snp['Volume'], errors='coerce')
snp = snp.sort_values('Date').reset_index(drop=True)
print(f"\n[Load]     S&P 500            : {time.perf_counter()-t0:.4f}s  ({len(snp):,} rows)")

t0 = time.perf_counter()
returns = snp['Close'].pct_change(fill_method=None)
_ = returns.mean(), returns.std(), returns.max(), returns.min()
_ = returns.rolling(30).std()
cum = (1 + returns).cumprod()
_ = ((cum - cum.cummax()) / cum.cummax()).min()
print(f"[Analysis] S&P 500            : {time.perf_counter()-t0:.4f}s")

# ── Discover all tickers ──────────────────────────────────────
t0          = time.perf_counter()
all_tickers = get_all_tickers(snp_raw)
remaining_tickers = [t for t in all_tickers if t not in SELECTED_TICKERS]

# ── Selected 11 companies ─────────────────────────────────────
selected_frames = []
for ticker in SELECTED_TICKERS:
    try:
        selected_frames.append(extract_ticker(snp_raw, ticker))
    except ValueError as e:
        print(f"  ✗ {ticker}: {e}")
selected_df = pd.concat(selected_frames, ignore_index=True).sort_values(['Ticker','Date'])
selected_df.to_csv('companies_selected.csv', index=False)
print(f"\n[Load]     Selected Companies : {time.perf_counter()-t0:.4f}s  ({len(selected_df):,} rows, {len(SELECTED_TICKERS)} tickers)")

t0 = time.perf_counter()
analyse_group(selected_df)
print(f"[Analysis] Selected Companies : {time.perf_counter()-t0:.4f}s")

# ── Remaining ~489 companies ──────────────────────────────────
t0 = time.perf_counter()
remaining_frames = []
for ticker in remaining_tickers:
    try:
        remaining_frames.append(extract_ticker(snp_raw, ticker))
    except ValueError:
        pass
remaining_df = pd.concat(remaining_frames, ignore_index=True).sort_values(['Ticker','Date'])
remaining_df.to_csv('companies_remaining.csv', index=False)
print(f"\n[Load]     Remaining Companies: {time.perf_counter()-t0:.4f}s  ({len(remaining_df):,} rows, {len(remaining_tickers)} tickers)")

t0 = time.perf_counter()
analyse_group(remaining_df)
print(f"[Analysis] Remaining Companies: {time.perf_counter()-t0:.4f}s")

# ── Bitcoin ───────────────────────────────────────────────────
t0  = time.perf_counter()
btc = pd.read_csv('BTC-USD (2014-2024).csv', usecols=['Date','Volume','Close'],
                  parse_dates=['Date'], dayfirst=True).sort_values('Date').reset_index(drop=True)
print(f"\n[Load]     Bitcoin            : {time.perf_counter()-t0:.4f}s  ({len(btc):,} rows)")

t0 = time.perf_counter()
returns = btc['Close'].pct_change(fill_method=None)
_ = returns.mean(), returns.std(), returns.max(), returns.min()
_ = returns.rolling(30).std()
cum = (1 + returns).cumprod()
_ = ((cum - cum.cummax()) / cum.cummax()).min()
print(f"[Analysis] Bitcoin            : {time.perf_counter()-t0:.4f}s")

# ── Overall ───────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"⏱  TOTAL: {time.perf_counter()-overall_start:.4f} seconds")
print(f"{'='*55}")
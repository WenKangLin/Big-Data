import pandas as pd
import numpy as np
import time
import tracemalloc
import os

DATA_DIR = 'data'

SNP_FILE    = 'SP500_USA.csv'
NASDAQ_FILE = 'nasdq.csv'
BTC_FILE    = 'BTC-USD (2014-2024).csv'

GLOBAL_FILES = {
    'BIST100_Turkey'  : 'BIST100_Turkey.csv',
    'Bovespa_Brazil'  : 'Bovespa_Brazil.csv',
    'DAX40_Germany'   : 'DAX40_Germany.csv',
    'FTSE100_UK'      : 'FTSE100_UK.csv',
    'IDX_Indonesia'   : 'IDX_Indonesia.csv',
    'NIFTY50_India'   : 'NIFTY50_India.csv',
    'Nikkei225_Japan' : 'Nikkei225_Japan.csv',
    'SaudiArabia'     : 'Tadawul_SaudiArabia.csv',
    'SSE_China'       : 'SSE_China.csv'
}



# ---------- Helpers ----------

def load_long_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, usecols=['Ticker','Date','Close','Volume'], parse_dates=['Date'])
    df['Close']  = pd.to_numeric(df['Close'],  errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna(subset=['Close']).sort_values(['Ticker','Date']).reset_index(drop=True)
    return df[['Date','Ticker','Close','Volume']]


def load_index_csv(filename, label):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, usecols=['Date','Close','Volume'], parse_dates=['Date'])
    df['Close']  = pd.to_numeric(df['Close'],  errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['Ticker'] = label
    df = df.dropna(subset=['Close']).sort_values('Date').reset_index(drop=True)
    return df[['Date','Ticker','Close','Volume']]


def analyse_group(df):
    for ticker, group in df.groupby('Ticker'):
        close = group['Close'].dropna()
        if len(close) < 2:
            continue
        returns = close.pct_change()
        _ = returns.mean(), returns.std(), returns.max(), returns.min()
        _ = returns.rolling(30).std()
        cum = (1 + returns).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        _ = drawdown.min()
        _ = group['Volume'].mean(), group['Volume'].std()


def run_section(label, frames_dict):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    timings = {'Section': label}

    # ---------- Load ----------
    t0 = time.perf_counter()
    tracemalloc.start()
    frames = []
    for source_label, df in frames_dict.items():
        frames.append(df)
        print(f"  {source_label:<20}: {len(df):,} rows  ({df['Ticker'].nunique()} tickers)")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings['Load Time (s)']         = round(time.perf_counter()-t0, 4)
    timings['Load Memory Peak (MB)'] = round(peak / (1024**2), 2)
    print(f"  Time        : {timings['Load Time (s)']:.4f}s")
    print(f"  Memory peak : {timings['Load Memory Peak (MB)']:.1f} MB")

    # ---------- Clean ----------
    t0 = time.perf_counter()
    tracemalloc.start()
    df = pd.concat(frames, ignore_index=True)
    mem = df.memory_usage(deep=True).sum() / (1024**2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings['Clean Time (s)']         = round(time.perf_counter()-t0, 4)
    timings['Clean Memory Peak (MB)'] = round(peak / (1024**2), 2)
    timings['DataFrame Size (MB)']    = round(mem, 2)
    timings['Total Rows']             = len(df)
    print(f"\n[Clean]")
    print(f"  Total rows  : {len(df):,}")
    print(f"  Time        : {timings['Clean Time (s)']:.4f}s")
    print(f"  Memory peak : {timings['Clean Memory Peak (MB)']:.1f} MB  (DataFrame ~{mem:.1f} MB)")

    # ---------- Analyse ----------
    t0 = time.perf_counter()
    tracemalloc.start()
    analyse_group(df)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings['Analysis Time (s)']         = round(time.perf_counter()-t0, 4)
    timings['Analysis Memory Peak (MB)'] = round(peak / (1024**2), 2)
    print(f"\n[Analysis]")
    print(f"  Tickers     : {df['Ticker'].nunique()}")
    print(f"  Time        : {timings['Analysis Time (s)']:.4f}s")
    print(f"  Memory peak : {timings['Analysis Memory Peak (MB)']:.1f} MB")

    return timings


# ---------- Main ----------
overall_start = time.perf_counter()
print("MARKET DATA ANALYSIS — TIMING")

# Load all datasets from data/ folder
sp500  = load_long_csv(SNP_FILE)
nasdq  = load_index_csv(NASDAQ_FILE, 'NASDAQ')
btc    = load_index_csv(BTC_FILE,    'BTC')
global_dfs = {label: load_long_csv(fname) for label, fname in GLOBAL_FILES.items()}

# ---------- Benchmark (S&P 500 only) ----------
benchmark_timings = run_section(
    "Benchmark  (S&P 500 only)",
    {'SP500_USA': sp500}
)

# ---------- Full Dataset (all files) ----------
full_frames = {'SP500_USA': sp500, 'NASDAQ': nasdq, 'BTC': btc}
full_frames.update(global_dfs)

full_timings = run_section(
    "Full Dataset  (All Markets + NASDAQ + BTC)",
    full_frames
)

total_time = round(time.perf_counter() - overall_start, 4)
print(f"\n{'=' * 55}")
print(f"⏱  TOTAL: {total_time:.4f} seconds")
print(f"{'=' * 55}")

# ---------- Save ----------
timings_df = pd.DataFrame([benchmark_timings, full_timings])
timings_df.to_csv('timings.csv', index=False)
print("\n[Saved]")

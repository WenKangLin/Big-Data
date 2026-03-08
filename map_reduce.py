import os
os.environ['JAVA_HOME'] = '/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home'
os.environ['PATH'] = os.environ['JAVA_HOME'] + '/bin:' + os.environ['PATH']

from pyspark import SparkContext
import pandas as pd
import numpy as np
import time
import tracemalloc

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

sc = SparkContext(appName="MarketDataAnalysis")
sc.setLogLevel("ERROR")


# ---------- Helpers ----------

def load_long_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, usecols=['Ticker','Date','Close','Volume'], parse_dates=['Date'], dayfirst=True)
    df['Close']  = pd.to_numeric(df['Close'],  errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna(subset=['Close']).sort_values(['Ticker','Date']).reset_index(drop=True)
    return list(df[['Ticker','Date','Close','Volume']].itertuples(index=False, name=None))


def load_index_csv(filename, label):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, usecols=['Date','Close','Volume'], parse_dates=['Date'], dayfirst=True)
    df['Close']  = pd.to_numeric(df['Close'],  errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna(subset=['Close']).sort_values('Date').reset_index(drop=True)
    return [(label, str(row.Date), float(row.Close), float(row.Volume) if row.Volume else None)
            for row in df.itertuples(index=False)]


# ---------- Map ----------
def parse_row(record):
    ticker, date, close, volume = record
    try:
        return (str(ticker), (str(date), float(close), float(volume) if volume else None))
    except (ValueError, TypeError):
        return None


# ---------- Reduce ----------
def compute_metrics(ticker_records):
    ticker, records = ticker_records
    records = sorted(records, key=lambda x: x[0])
    closes  = np.array([r[1] for r in records], dtype=float)
    volumes = np.array([r[2] for r in records if r[2] is not None], dtype=float)

    if len(closes) < 2:
        return (ticker, None)

    returns     = np.diff(closes) / closes[:-1]
    cum         = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum)
    drawdown    = (cum - rolling_max) / rolling_max
    roll_vol    = np.array([returns[max(0, i-29):i+1].std() for i in range(len(returns))])

    return (ticker, {
        'Ticker':          ticker,
        'Mean Return':     float(returns.mean()),
        'Volatility':      float(returns.std()),
        'Max Return':      float(returns.max()),
        'Min Return':      float(returns.min()),
        'Max Drawdown':    float(drawdown.min()),
        'Avg Roll Vol 30': float(roll_vol.mean()),
        'Avg Volume':      float(volumes.mean()) if len(volumes) > 0 else None,
        'Std Volume':      float(volumes.std())  if len(volumes) > 0 else None,
    })


def run_section(label, records_dict):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    timings = {'Section': label}

    # ---------- Load / Map ----------
    t0 = time.perf_counter()
    tracemalloc.start()

    all_records = []
    for source_label, recs in records_dict.items():
        all_records += recs
        tickers = len(set(r[0] for r in recs))
        print(f"  {source_label:<20}: {len(recs):,} records  ({tickers} tickers)")

    rdd    = sc.parallelize(all_records)
    mapped = rdd.map(parse_row).filter(lambda x: x is not None)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings['Load Time (s)']         = round(time.perf_counter()-t0, 4)
    timings['Load Memory Peak (MB)'] = round(peak / (1024**2), 2)
    print(f"  Time        : {timings['Load Time (s)']:.4f}s")
    print(f"  Memory peak : {timings['Load Memory Peak (MB)']:.1f} MB")

    # ---------- Clean (Shuffle / GroupByKey) ----------
    t0 = time.perf_counter()
    tracemalloc.start()
    grouped = mapped.groupByKey().mapValues(list)
    grouped.cache()
    num_tickers = grouped.count()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings['Clean Time (s)']         = round(time.perf_counter()-t0, 4)
    timings['Clean Memory Peak (MB)'] = round(peak / (1024**2), 2)
    timings['Total Rows']             = len(all_records)
    print(f"\n[Clean]")
    print(f"  Total rows  : {len(all_records):,}")
    print(f"  Tickers     : {num_tickers}")
    print(f"  Time        : {timings['Clean Time (s)']:.4f}s")
    print(f"  Memory peak : {timings['Clean Memory Peak (MB)']:.1f} MB")

    # ---------- Analyse (Reduce) ----------
    t0 = time.perf_counter()
    tracemalloc.start()
    results = grouped.map(compute_metrics).filter(lambda x: x[1] is not None).collect()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings['Analysis Time (s)']         = round(time.perf_counter()-t0, 4)
    timings['Analysis Memory Peak (MB)'] = round(peak / (1024**2), 2)
    print(f"\n[Analysis]")
    print(f"  Tickers processed : {len(results)}")
    print(f"  Time              : {timings['Analysis Time (s)']:.4f}s")
    print(f"  Memory peak       : {timings['Analysis Memory Peak (MB)']:.1f} MB")

    grouped.unpersist()
    return timings


# ---------- Main ----------
overall_start = time.perf_counter()
print("MARKET DATA ANALYSIS — MapReduce (Spark Core)")

# Load all datasets from data/ folder
sp500      = load_long_csv(SNP_FILE)
nasdq      = load_index_csv(NASDAQ_FILE, 'NASDAQ')
btc        = load_index_csv(BTC_FILE,    'BTC')
global_recs = {label: load_long_csv(fname) for label, fname in GLOBAL_FILES.items()}

# ---------- Benchmark (S&P 500 only) ----------
benchmark_timings = run_section(
    "Benchmark  (S&P 500 only)",
    {'SP500_USA': sp500}
)

# ---------- Full Dataset (all files) ----------
full_records = {'SP500_USA': sp500, 'NASDAQ': nasdq, 'BTC': btc}
full_records.update(global_recs)

full_timings = run_section(
    "Full Dataset  (All Markets + NASDAQ + BTC)",
    full_records
)

total_time = round(time.perf_counter() - overall_start, 4)
print(f"\n{'=' * 55}")
print(f"⏱  TOTAL: {total_time:.4f} seconds")
print(f"{'=' * 55}")

# ---------- Save ----------
timings_df = pd.DataFrame([benchmark_timings, full_timings])
timings_df.to_csv('mapreduce_timings.csv', index=False)
print("\n[Saved]")

sc.stop()

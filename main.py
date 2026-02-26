import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

ALL_EVENTS = {
    # ── Geopolitical ──────────────────────────────────────────
    'Arab Spring Fallout':          {'date': '2011-08-05', 'color': '#FF4500', 'category': 'Geopolitical', 'major': True},
    'Libyan Civil War':             {'date': '2011-02-17', 'color': '#FF6347', 'category': 'Geopolitical', 'major': False},
    'Syrian Civil War Begins':      {'date': '2011-03-15', 'color': '#FF7F50', 'category': 'Geopolitical', 'major': False},
    'Syrian Civil War Peak (Russia)':{'date': '2015-09-30','color': '#DC143C', 'category': 'Geopolitical', 'major': True},
    'Crimea Annexation':            {'date': '2014-03-18', 'color': '#8B0000', 'category': 'Geopolitical', 'major': True},
    'Ukraine War Begins (2014)':    {'date': '2014-04-07', 'color': '#B22222', 'category': 'Geopolitical', 'major': True},
    'ISIS Peak / Caliphate':        {'date': '2014-06-29', 'color': '#CD5C5C', 'category': 'Geopolitical', 'major': False},
    'Brexit Vote':                  {'date': '2016-06-23', 'color': '#FF8C00', 'category': 'Geopolitical', 'major': True},
    'North Korea Missile Tests':    {'date': '2017-09-03', 'color': '#FFA500', 'category': 'Geopolitical', 'major': False},
    'US Kills Soleimani':           {'date': '2020-01-03', 'color': '#FF4500', 'category': 'Geopolitical', 'major': False},
    'Taliban Takeover':             {'date': '2021-08-15', 'color': '#CD853F', 'category': 'Geopolitical', 'major': True},
    'Russia Invades Ukraine':       {'date': '2022-02-24', 'color': '#8B0000', 'category': 'Geopolitical', 'major': True},
    'Palestine War (Oct 7)':        {'date': '2023-10-07', 'color': '#A0522D', 'category': 'Geopolitical', 'major': True},

    # ── Political ─────────────────────────────────────────────
    'Obama Re-elected':             {'date': '2012-11-06', 'color': '#4682B4', 'category': 'Political', 'major': False},
    'Trump Elected (2016)':         {'date': '2016-11-08', 'color': '#4169E1', 'category': 'Political', 'major': True},
    'Trump Inaugurated':            {'date': '2017-01-20', 'color': '#6495ED', 'category': 'Political', 'major': False},
    'US Govt Shutdown 2018':        {'date': '2018-01-20', 'color': '#5F9EA0', 'category': 'Political', 'major': False},
    'Trump Impeachment #1':         {'date': '2019-12-18', 'color': '#87CEEB', 'category': 'Political', 'major': False},
    'Biden Elected (2020)':         {'date': '2020-11-07', 'color': '#1E90FF', 'category': 'Political', 'major': True},
    'Jan 6 Capitol Riot':           {'date': '2021-01-06', 'color': '#00BFFF', 'category': 'Political', 'major': True},
    'Trump Elected (2024)':         {'date': '2024-11-05', 'color': '#4169E1', 'category': 'Political', 'major': True},

    # ── Economic / Financial ──────────────────────────────────
    'US Debt Ceiling Crisis':       {'date': '2011-08-02', 'color': '#2E8B57', 'category': 'Economic', 'major': True},
    'Eurozone Debt Crisis Peak':    {'date': '2011-11-09', 'color': '#3CB371', 'category': 'Economic', 'major': True},
    'China Market Crash':           {'date': '2015-06-12', 'color': '#20B2AA', 'category': 'Economic', 'major': True},
    'Oil Price Crash':              {'date': '2016-01-20', 'color': '#66CDAA', 'category': 'Economic', 'major': True},
    'US-China Trade War':           {'date': '2018-03-22', 'color': '#7B68EE', 'category': 'Economic', 'major': True},
    'Global Tariff Wars':           {'date': '2018-07-06', 'color': '#6A5ACD', 'category': 'Economic', 'major': True},
    'Repo Market Crisis':           {'date': '2019-09-17', 'color': '#00CED1', 'category': 'Economic', 'major': False},
    'COVID-19 Crash':               {'date': '2020-03-11', 'color': '#006400', 'category': 'Economic', 'major': True},
    'COVID Stimulus / Recovery':    {'date': '2020-04-09', 'color': '#228B22', 'category': 'Economic', 'major': False},
    'Inflation Surge Begins':       {'date': '2021-05-01', 'color': '#32CD32', 'category': 'Economic', 'major': True},
    'RAM / Chip Shortage Peak':     {'date': '2021-09-01', 'color': '#00CED1', 'category': 'Economic', 'major': True},
    'Fed Rate Hikes Begin':         {'date': '2022-03-16', 'color': '#008000', 'category': 'Economic', 'major': True},
    'SVB Collapse':                 {'date': '2023-03-10', 'color': '#7CFC00', 'category': 'Economic', 'major': True},
    'US Credit Downgrade (Fitch)':  {'date': '2023-08-01', 'color': '#ADFF2F', 'category': 'Economic', 'major': False},

    # ── Tech / AI ─────────────────────────────────────────────
    'iPhone 6 Launch':              {'date': '2014-09-09', 'color': '#9370DB', 'category': 'Tech/AI', 'major': False},
    'NVIDIA GPU AI Boom Begins':    {'date': '2016-09-13', 'color': '#8A2BE2', 'category': 'Tech/AI', 'major': False},
    'GPT-3 Released':               {'date': '2020-06-11', 'color': '#9400D3', 'category': 'Tech/AI', 'major': False},
    'ChatGPT Launch':               {'date': '2022-11-30', 'color': '#8B008B', 'category': 'Tech/AI', 'major': True},
    'AI Boom (GPT-4 / NVIDIA surge)':{'date': '2023-03-14','color': '#DA70D6', 'category': 'Tech/AI', 'major': True},
    'Meta AI / Llama Release':      {'date': '2023-07-18', 'color': '#EE82EE', 'category': 'Tech/AI', 'major': False},

    # ── Crypto ────────────────────────────────────────────────
    'Bitcoin Halving 2016':         {'date': '2016-07-09', 'color': '#FF1493', 'category': 'Crypto', 'major': False},
    'Crypto Bull Run 2017':         {'date': '2017-12-17', 'color': '#FF69B4', 'category': 'Crypto', 'major': True},
    'Crypto Crash 2018':            {'date': '2018-01-17', 'color': '#C71585', 'category': 'Crypto', 'major': True},
    'Bitcoin Halving 2020':         {'date': '2020-05-11', 'color': '#DB7093', 'category': 'Crypto', 'major': False},
    'Crypto Bull Run 2021':         {'date': '2021-11-10', 'color': '#FFB6C1', 'category': 'Crypto', 'major': True},
    'Crypto Crash (Luna)':          {'date': '2022-05-09', 'color': '#FF00FF', 'category': 'Crypto', 'major': True},
    'FTX Collapse':                 {'date': '2022-11-11', 'color': '#FF00FF', 'category': 'Crypto', 'major': True},
    'Bitcoin Halving 2024':         {'date': '2024-04-19', 'color': '#FF69B4', 'category': 'Crypto', 'major': False},

    # ── Health ────────────────────────────────────────────────
    'Ebola Outbreak':               {'date': '2014-08-08', 'color': '#8FBC8F', 'category': 'Health', 'major': False},
    'COVID-19 WHO Declaration':     {'date': '2020-03-11', 'color': '#2E8B57', 'category': 'Health', 'major': True},
    'Vaccine Rollout Begins':       {'date': '2020-12-14', 'color': '#90EE90', 'category': 'Health', 'major': False},
}

# Major events only (used for most plots to avoid clutter)
EVENTS = {k: v for k, v in ALL_EVENTS.items() if v['major']}

# Print all events for reference
print("=" * 60)
print(f"ALL EVENTS ({len(ALL_EVENTS)} total):")
print("=" * 60)
for cat in ['Geopolitical', 'Political', 'Economic', 'Tech/AI', 'Crypto', 'Health']:
    print(f"\n  [{cat}]")
    for name, ev in ALL_EVENTS.items():
        if ev['category'] == cat:
            tag = '★ MAJOR' if ev['major'] else '  minor'
            print(f"    {tag}  {ev['date']}  {name}")

print(f"\n★ Major events selected for analysis: {len(EVENTS)}")
print(f"  minor events logged but excluded from plots: {len(ALL_EVENTS) - len(EVENTS)}")
print("=" * 60)

COMPANIES = {
    'AAPL': 'Apple',
    'NVDA': 'NVIDIA',
    'TSLA': 'Tesla',
    'AMZN': 'Amazon',
    'AMD':  'AMD',
    'PFE':  'Pfizer',
    'JNJ':  'Johnson & Johnson',
    'BSX':  'Boston Scientific',
    'NOC':  'Northrop Grumman',
    'LMT':  'Lockheed Martin',
    'BA':   'Boeing',
}

WINDOW_BEFORE = 30   # days before event
WINDOW_AFTER  = 90   # days after event


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_ticker(snp_raw, ticker):
    ticker_row  = snp_raw.iloc[0]
    close_cols  = [c for c in snp_raw.columns if c == 'Close'  or c.startswith('Close.')]
    volume_cols = [c for c in snp_raw.columns if c == 'Volume' or c.startswith('Volume.')]
    matched_close = [c for c in close_cols  if ticker_row[c] == ticker]
    matched_vol   = [c for c in volume_cols if ticker_row[c] == ticker]
    if not matched_close:
        raise ValueError(f"Ticker '{ticker}' not found in S&P file.")
    close_col = matched_close[0]
    vol_col   = matched_vol[0] if matched_vol else None
    df = snp_raw[['Price', close_col]].iloc[1:].copy()
    df.columns = ['Date', 'Close']
    if vol_col:
        df['Volume'] = snp_raw[vol_col].iloc[1:].values
    else:
        df['Volume'] = np.nan
    df = df[df['Date'].str.match(r'\d{2}/\d{2}/\d{4}', na=False)].copy()
    df['Date']   = pd.to_datetime(df['Date'], dayfirst=True)
    df['Close']  = pd.to_numeric(df['Close'],  errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    return df.sort_values('Date').reset_index(drop=True)[['Date', 'Close', 'Volume']]


def normalise(series):
    first = series.dropna().iloc[0] if not series.dropna().empty else 1
    return series / first * 100


def get_event_window(df, event_date, before=WINDOW_BEFORE, after=WINDOW_AFTER):
    ed = pd.to_datetime(event_date)
    mask = (df['Date'] >= ed - pd.Timedelta(days=before)) & \
           (df['Date'] <= ed + pd.Timedelta(days=after))
    return df[mask].copy()


def calc_drawdown(df, event_date, after=WINDOW_AFTER):
    ed = pd.to_datetime(event_date)
    window = df[(df['Date'] >= ed) & (df['Date'] <= ed + pd.Timedelta(days=after))]['Close']
    if window.empty:
        return np.nan
    peak = window.iloc[0]
    trough = window.min()
    return ((trough - peak) / peak) * 100


def calc_recovery_days(df, event_date, after=WINDOW_AFTER):
    ed = pd.to_datetime(event_date)
    window = df[(df['Date'] >= ed) & (df['Date'] <= ed + pd.Timedelta(days=after))].copy()
    if window.empty:
        return np.nan
    baseline = window['Close'].iloc[0]
    recovered = window[window['Close'] >= baseline]
    if len(recovered) <= 1:
        return np.nan  # did not recover in window
    return (recovered.iloc[1]['Date'] - ed).days


# ============================================================
# 1) LOAD & CLEAN DATA
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# NASDAQ
print("Loading NASDAQ...")
nasdq = pd.read_csv(
    'nasdq.csv',
    usecols=['Date', 'Volume', 'Close'],
    parse_dates=['Date']
).sort_values('Date').reset_index(drop=True)
nasdq.to_csv('NASDQ_clean.csv', index=False)

# S&P wide file
print("Loading S&P 500 wide file...")
snp_raw = pd.read_csv('SnP_daily_update.csv', header=0, low_memory=False)

snp = snp_raw[['Price', 'Close', 'Volume']].iloc[1:].copy()
snp.columns = ['Date', 'Close', 'Volume']
snp = snp[snp['Date'].str.match(r'\d{2}/\d{2}/\d{4}', na=False)].copy()
snp['Date']   = pd.to_datetime(snp['Date'], dayfirst=True)
snp['Close']  = pd.to_numeric(snp['Close'],  errors='coerce')
snp['Volume'] = pd.to_numeric(snp['Volume'], errors='coerce')
snp = snp.sort_values('Date').reset_index(drop=True)
snp.to_csv('SNP_clean.csv', index=False)

# Companies
print("Extracting company tickers...")
cleaned_companies = {}
for ticker, name in COMPANIES.items():
    try:
        df = extract_ticker(snp_raw, ticker)
        df.to_csv(f'{ticker}_clean.csv', index=False)
        cleaned_companies[ticker] = df
        print(f"  ✓ {ticker} ({name})")
    except ValueError as e:
        print(f"  ✗ {ticker}: {e}")

# Bitcoin
print("Loading Bitcoin...")
btc = pd.read_csv(
    'BTC-USD (2014-2024).csv',
    usecols=['Date', 'Volume', 'Close'],
    parse_dates=['Date']
).sort_values('Date').reset_index(drop=True)
btc.to_csv('BTC_clean.csv', index=False)

# All assets dict for easy iteration
all_assets = {
    'NASDAQ': nasdq,
    'S&P500': snp,
    'BTC':    btc,
    **{ticker: df for ticker, df in cleaned_companies.items()}
}

print(f"\n✓ Loaded {len(all_assets)} assets total")


# ============================================================
# 2) MASTER FILE
# ============================================================
print("\nBuilding master file...")
master = nasdq.rename(columns={'Close':'NASDQ_Close','Volume':'NASDQ_Volume'})
master = master.merge(snp.rename(columns={'Close':'SNP_Close','Volume':'SNP_Volume'}), on='Date', how='outer')
master = master.merge(btc.rename(columns={'Close':'BTC_Close','Volume':'BTC_Volume'}), on='Date', how='outer')
for ticker, df in cleaned_companies.items():
    master = master.merge(
        df.rename(columns={'Close':f'{ticker}_Close','Volume':f'{ticker}_Volume'}),
        on='Date', how='outer'
    )
master = master.sort_values('Date').reset_index(drop=True)
master.to_csv('master_all.csv', index=False)
print(f"Master file: {master.shape[0]} rows x {master.shape[1]} cols -> master_all.csv")


# ============================================================
# 3) PLOT 1 — Full timeline: Indices normalised + event lines
# ============================================================
print("\nPlot 1: Full timeline with event lines...")

fig, axes = plt.subplots(2, 1, figsize=(20, 14), sharex=False)

# --- Top: Indices ---
ax = axes[0]
index_assets = {'NASDAQ': nasdq, 'S&P500': snp, 'BTC': btc}
colors_idx   = {'NASDAQ': '#1f77b4', 'S&P500': '#ff7f0e', 'BTC': '#2ca02c'}

for label, df in index_assets.items():
    d = df.dropna(subset=['Close']).copy()
    d['Norm'] = normalise(d['Close'])
    ax.plot(d['Date'], d['Norm'], label=label, color=colors_idx[label], linewidth=1.5)

# Event lines
for name, ev in EVENTS.items():
    ed = pd.to_datetime(ev['date'])
    ax.axvline(ed, color=ev['color'], alpha=0.7, linewidth=0.8, linestyle='--')
    ax.text(ed, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 300,
            name, rotation=90, fontsize=5.5, color=ev['color'],
            va='top', ha='right', alpha=0.85)

ax.set_title("NASDAQ, S&P 500 & BTC — Normalised Close with World Events", fontsize=13, fontweight='bold')
ax.set_ylabel("Normalised Close (base=100)")
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(alpha=0.3)

# --- Bottom: Companies ---
ax2 = axes[1]
co_colors = plt.cm.tab20(np.linspace(0, 1, len(cleaned_companies)))

for (ticker, df), color in zip(cleaned_companies.items(), co_colors):
    d = df.dropna(subset=['Close']).copy()
    d['Norm'] = normalise(d['Close'])
    ax2.plot(d['Date'], d['Norm'], label=ticker, color=color, linewidth=1.2)

for name, ev in EVENTS.items():
    ed = pd.to_datetime(ev['date'])
    ax2.axvline(ed, color=ev['color'], alpha=0.5, linewidth=0.7, linestyle='--')

ax2.set_title("Company Stocks — Normalised Close with World Events", fontsize=13, fontweight='bold')
ax2.set_ylabel("Normalised Close (base=100)")
ax2.set_xlabel("Date")
ax2.legend(loc='upper left', fontsize=8, ncol=2)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plot_01_timeline_events.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot_01_timeline_events.png")


# ============================================================
# 3b) PLOT 1b — NASDAQ vs S&P only (no BTC — keeps scale readable)
# ============================================================
print("Plot 1b: Indices without BTC...")

fig, axes = plt.subplots(2, 1, figsize=(20, 14))

# Top panel: NASDAQ vs S&P normalised
ax = axes[0]
for label, df, color in [('NASDAQ', nasdq, '#1f77b4'), ('S&P500', snp, '#ff7f0e')]:
    d = df.dropna(subset=['Close']).copy()
    d['Norm'] = normalise(d['Close'])
    ax.plot(d['Date'], d['Norm'], label=label, color=color, linewidth=1.8)

for name, ev in EVENTS.items():
    ed = pd.to_datetime(ev['date'])
    ax.axvline(ed, color=ev['color'], alpha=0.7, linewidth=0.8, linestyle='--')
    ax.text(ed, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 800,
            name, rotation=90, fontsize=5.5, color=ev['color'],
            va='top', ha='right', alpha=0.85)

ax.set_title("NASDAQ vs S&P 500 — Normalised Close with World Events (BTC excluded for scale)", fontsize=13, fontweight='bold')
ax.set_ylabel("Normalised Close (base=100)")
ax.legend(loc='upper left', fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(alpha=0.3)

# Bottom panel: BTC standalone for reference
ax2 = axes[1]
d = btc.dropna(subset=['Close']).copy()
d['Norm'] = normalise(d['Close'])
ax2.plot(d['Date'], d['Norm'], label='BTC', color='#2ca02c', linewidth=1.5)
ax2.fill_between(d['Date'], d['Norm'], alpha=0.12, color='#2ca02c')

for name, ev in EVENTS.items():
    ed = pd.to_datetime(ev['date'])
    ax2.axvline(ed, color=ev['color'], alpha=0.5, linewidth=0.7, linestyle='--')

ax2.set_title("BTC — Normalised Close (standalone, own scale)", fontsize=13, fontweight='bold')
ax2.set_ylabel("Normalised Close (base=100)")
ax2.set_xlabel("Date")
ax2.legend(loc='upper left', fontsize=10)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plot_01b_indices_no_btc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot_01b_indices_no_btc.png")


# ============================================================
# 4) PLOT 2 — Per-event window charts (all assets)
# ============================================================
print("Plot 2: Per-event window charts...")

# Group events by category for organised subplots
category_order = ['Geopolitical', 'Political', 'Economic', 'Tech/AI', 'Crypto']

for event_name, ev in EVENTS.items():
    ed = pd.to_datetime(ev['date'])
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle(f"Market Reaction: {event_name}  ({ev['date']})\n"
                 f"{WINDOW_BEFORE} days before → {WINDOW_AFTER} days after",
                 fontsize=13, fontweight='bold')

    # Panel 1: Indices
    ax = axes[0]
    ax.set_title("Indices (NASDAQ, S&P 500, BTC)")
    any_data = False
    for label, df, color in zip(
        ['NASDAQ','S&P500','BTC'],
        [nasdq, snp, btc],
        ['#1f77b4','#ff7f0e','#2ca02c']
    ):
        w = get_event_window(df, ev['date'])
        if w.empty:
            continue
        w['Norm'] = normalise(w['Close'])
        ax.plot(w['Date'], w['Norm'], label=label, color=color, linewidth=2)
        any_data = True
    if any_data:
        ax.axvline(ed, color='red', linewidth=1.5, linestyle='-', label='Event')
        ax.axhline(100, color='gray', linewidth=0.8, linestyle=':')
    ax.legend(fontsize=8); ax.set_ylabel("Normalised Close"); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # Panel 2: Tech companies
    ax2 = axes[1]
    ax2.set_title("Tech Companies (AAPL, NVDA, TSLA, AMZN, AMD)")
    tech_tickers = ['AAPL','NVDA','TSLA','AMZN','AMD']
    tech_colors  = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
    for t, c in zip(tech_tickers, tech_colors):
        if t not in cleaned_companies:
            continue
        w = get_event_window(cleaned_companies[t], ev['date'])
        if w.empty:
            continue
        w['Norm'] = normalise(w['Close'])
        ax2.plot(w['Date'], w['Norm'], label=t, color=c, linewidth=1.5)
    ax2.axvline(ed, color='red', linewidth=1.5, linestyle='-', label='Event')
    ax2.axhline(100, color='gray', linewidth=0.8, linestyle=':')
    ax2.legend(fontsize=8); ax2.set_ylabel("Normalised Close"); ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # Panel 3: Other companies
    ax3 = axes[2]
    ax3.set_title("Other Companies (PFE, JNJ, BSX, NOC, LMT, BA)")
    other_tickers = ['PFE','JNJ','BSX','NOC','LMT','BA']
    other_colors  = ['#a65628','#f781bf','#999999','#66c2a5','#fc8d62','#8da0cb']
    for t, c in zip(other_tickers, other_colors):
        if t not in cleaned_companies:
            continue
        w = get_event_window(cleaned_companies[t], ev['date'])
        if w.empty:
            continue
        w['Norm'] = normalise(w['Close'])
        ax3.plot(w['Date'], w['Norm'], label=t, color=c, linewidth=1.5)
    ax3.axvline(ed, color='red', linewidth=1.5, linestyle='-', label='Event')
    ax3.axhline(100, color='gray', linewidth=0.8, linestyle=':')
    ax3.legend(fontsize=8); ax3.set_ylabel("Normalised Close"); ax3.grid(alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.set_xlabel("Date")

    plt.tight_layout()
    safe_name = event_name.replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '')
    fname = f'plot_02_event_{safe_name}.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ============================================================
# 5) PLOT 3 — Drawdown table heatmap
# ============================================================
print("Plot 3: Drawdown heatmap...")

drawdown_data = {}
for asset_name, df in all_assets.items():
    row = {}
    for ev_name, ev in EVENTS.items():
        row[ev_name] = calc_drawdown(df, ev['date'])
    drawdown_data[asset_name] = row

dd_df = pd.DataFrame(drawdown_data).T  # assets as rows, events as cols
dd_df = dd_df.astype(float)

fig, ax = plt.subplots(figsize=(24, 8))
sns.heatmap(
    dd_df,
    annot=True, fmt='.1f', linewidths=0.4,
    cmap='RdYlGn', center=0,
    vmin=-60, vmax=60,
    ax=ax, annot_kws={'size': 6.5}
)
ax.set_title("Drawdown (%) After Each World Event — All Assets\n(Negative = price fell, Positive = price rose)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("World Event")
ax.set_ylabel("Asset")
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('plot_03_drawdown_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot_03_drawdown_heatmap.png")

# Save drawdown table to CSV
dd_df.to_csv('drawdown_table.csv')
print("  Saved: drawdown_table.csv")


# ============================================================
# 6) PLOT 4 — Recovery time heatmap
# ============================================================
print("Plot 4: Recovery time heatmap...")

recovery_data = {}
for asset_name, df in all_assets.items():
    row = {}
    for ev_name, ev in EVENTS.items():
        row[ev_name] = calc_recovery_days(df, ev['date'])
    recovery_data[asset_name] = row

rec_df = pd.DataFrame(recovery_data).T.astype(float)

fig, ax = plt.subplots(figsize=(24, 8))
sns.heatmap(
    rec_df,
    annot=True, fmt='.0f', linewidths=0.4,
    cmap='RdYlGn_r',
    vmin=0, vmax=90,
    ax=ax, annot_kws={'size': 6.5}
)
ax.set_title("Recovery Time (Days) After Each World Event — All Assets\n"
             "(NaN = did not recover within 90-day window, blank = event outside data range)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("World Event")
ax.set_ylabel("Asset")
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('plot_04_recovery_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot_04_recovery_heatmap.png")

rec_df.to_csv('recovery_table.csv')
print("  Saved: recovery_table.csv")


# ============================================================
# 7) PLOT 5 — Correlation heatmaps: Crisis vs Normal periods
# ============================================================
print("Plot 5: Correlation heatmaps...")

# Build a wide close-price dataframe
close_wide = pd.DataFrame({'Date': master['Date']})
for col in master.columns:
    if col.endswith('_Close'):
        label = col.replace('_Close','')
        close_wide[label] = master[col]
close_wide = close_wide.set_index('Date')

# Define crisis periods
crisis_periods = [
    ('2011-07-01', '2011-10-01'),   # US debt ceiling
    ('2016-06-01', '2016-12-31'),   # Brexit + Trump
    ('2018-03-01', '2018-12-31'),   # Trade war
    ('2020-02-01', '2020-06-30'),   # COVID crash
    ('2022-02-01', '2022-09-30'),   # Ukraine + rate hikes
    ('2023-03-01', '2023-06-30'),   # SVB + AI boom
]

crisis_mask = pd.Series(False, index=close_wide.index)
for start, end in crisis_periods:
    crisis_mask |= (close_wide.index >= start) & (close_wide.index <= end)

crisis_data = close_wide[crisis_mask].pct_change().dropna()
normal_data = close_wide[~crisis_mask].pct_change().dropna()

fig, axes = plt.subplots(1, 2, figsize=(22, 9))

for ax, data, title, cmap in zip(
    axes,
    [crisis_data, normal_data],
    ['Crisis Periods — Return Correlation', 'Normal Periods — Return Correlation'],
    ['coolwarm', 'coolwarm']
):
    if data.empty:
        ax.set_title(title + " (no data)")
        continue
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap=cmap, center=0, vmin=-1, vmax=1,
        linewidths=0.3, ax=ax, annot_kws={'size': 7},
        square=True
    )
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)

plt.suptitle("Asset Return Correlations: Crisis vs Normal Periods", fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_05_correlation_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot_05_correlation_heatmaps.png")


# ============================================================
# 8) PLOT 6 — Category event comparison (avg drawdown by sector)
# ============================================================
print("Plot 6: Event category comparison...")

# Group drawdowns by event category
cat_dd = {}
for ev_name, ev in EVENTS.items():
    cat = ev['category']
    if cat not in cat_dd:
        cat_dd[cat] = {}
    for asset in all_assets:
        if asset not in cat_dd[cat]:
            cat_dd[cat][asset] = []
        val = dd_df.loc[asset, ev_name] if asset in dd_df.index and ev_name in dd_df.columns else np.nan
        if not np.isnan(val):
            cat_dd[cat][asset].append(val)

# Average drawdown per category per asset
cat_avg = {cat: {a: np.mean(v) for a, v in assets.items() if v}
           for cat, assets in cat_dd.items()}
cat_avg_df = pd.DataFrame(cat_avg)

fig, ax = plt.subplots(figsize=(14, 7))
cat_avg_df.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='white', width=0.75)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title("Average Drawdown (%) by Event Category — All Assets", fontsize=12, fontweight='bold')
ax.set_xlabel("Asset")
ax.set_ylabel("Average Drawdown (%)")
ax.legend(title='Event Category', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('plot_06_category_drawdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot_06_category_drawdown.png")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✓ ALL DONE")
print("=" * 60)
print("\nCLEAN DATA FILES:")
print("  NASDQ_clean.csv, SNP_clean.csv, BTC_clean.csv")
print("  " + ", ".join([f"{t}_clean.csv" for t in cleaned_companies]) )
print("  master_all.csv")
print("\nANALYSIS FILES:")
print("  drawdown_table.csv")
print("  recovery_table.csv")
print("\nPLOTS:")
print("  plot_01_timeline_events.png   — Full timeline with all event lines")
print("  plot_02_event_<name>.png      — Per-event window (one per event)")
print("  plot_03_drawdown_heatmap.png  — Drawdown % heatmap")
print("  plot_04_recovery_heatmap.png  — Recovery days heatmap")
print("  plot_05_correlation_heatmaps.png — Crisis vs Normal correlations")
print("  plot_06_category_drawdown.png — Avg drawdown by event category")
print(f"\nTotal events tracked: {len(EVENTS)}")
print(f"Total assets tracked: {len(all_assets)}")


# 📊 Stock Market Index Simulator

A real-time interactive Streamlit dashboard that simulates **index construction**, **price dynamics**, and **automatic rebalancing** — built entirely with Python and Pandas.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install streamlit pandas numpy

# 2. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

Or you can visit the live link: `https://stock-sim-soham.streamlit.app`

---

## 🧠 Architecture Overview

```
stock_index_simulator/
├── app.py           ← Streamlit UI & simulation loop
├── data_engine.py   ← All data logic (Pandas-native)
└── README.md
```

### Separation of concerns
- **`data_engine.py`** contains pure Python/Pandas functions — no Streamlit imports. Fully testable in isolation.
- **`app.py`** handles all UI, session state, and the render loop.

---

## 📐 Index Construction Logic

### 1. Universe Definition
30 fictional companies across 7 sectors (Technology, Finance, Energy, Healthcare, Consumer, Industrials, Telecom, Real Estate). Each company has:
- `name`, `ticker`, `sector`
- `shares` (shares outstanding — fixed)
- `price` (dynamic, changes each tick)

### 2. Market Cap Calculation
```python
market_cap = price × shares_outstanding
```
Computed as a Pandas column:
```python
df["market_cap"] = df["price"] * df["shares"]
```

### 3. Constituent Selection
Top-N companies ranked by market cap using Pandas:
```python
sorted_df = df.sort_values("market_cap", ascending=False)
index_df  = sorted_df.head(top_n)
```

### 4. Weight Assignment
Market-cap proportional weighting:
```python
total_mcap        = index_df["market_cap"].sum()
index_df["weight"] = index_df["market_cap"] / total_mcap * 100
```

### 5. Index Value Calculation
Normalized to a base value of 1,000 at inception:
```python
index_value = 1000 × (current_total_mcap / base_total_mcap)
```
This mirrors how real-world indices (e.g., MSCI, S&P 500) are computed.

---

## 🔄 Rebalancing Logic

Rebalancing recalculates index constituents and detects changes:

```python
new_index_df  = construct_index(df, top_n)
new_tickers   = set(new_index_df["ticker"])
added         = new_tickers - current_tickers   # Companies newly qualifying
removed       = current_tickers - new_tickers   # Companies falling below cutoff
```

### When does rebalancing happen?
| Mode | Trigger |
|---|---|
| Auto | Every N ticks (configurable via sidebar slider) |
| Manual | Click **"Rebalance Now"** button |

### What happens during rebalance?
1. Rank all companies by current market cap
2. Select top N as new constituents
3. Identify added / removed names
4. Recalculate weights for new constituent set
5. Log the event with timestamp and tick number

---

## 📊 Data Handling with Pandas

| Operation | Pandas Method | Purpose |
|---|---|---|
| Price update | `df["price"] * (1 + pct_moves)` | Vectorized random walk |
| Market cap | `df["price"] * df["shares"]` | Column arithmetic |
| Sorting | `df.sort_values("market_cap")` | Rank by size |
| Filtering | `df[df["sector"] == sector]` | Sector filter |
| Grouping | `df.groupby("sector")` | Sector summary |
| Aggregation | `.agg(sum, mean, count)` | Multi-stat rollup |
| Top N | `.head(n)` | Index selection |
| Clipping | `.clip(lower=1.0)` | Price floor |

---

## 🎛️ UI Controls

| Control | Description |
|---|---|
| **Index Size (N)** | Number of companies in the index (5–20) |
| **Volatility** | Scale of price movement per tick |
| **Auto Rebalance** | Toggle automatic rebalancing on/off |
| **Rebalance Frequency** | How often (in ticks) to rebalance |
| **Sector Filter** | Show only companies from one sector |
| **Live Simulation** | Toggle real-time price updates |
| **Refresh Rate** | Seconds between simulation ticks |

---

## 📈 Visualizations

| Chart | Library | What it shows |
|---|---|---|
| Line chart | `st.line_chart` | Index value over time (last 120 ticks) |
| Bar chart | `st.bar_chart` | Total market cap by sector |
| DataFrames | `st.dataframe` | Live company data + index constituents |

---

## 💡 Key Design Decisions

- **Random walk (not random price)**: Prices evolve multiplicatively (`price × (1 + Δ)`) to avoid going negative and to mirror real log-normal returns.
- **Fixed shares outstanding**: Only prices change each tick — market cap changes come purely from price movement, just like in real markets.
- **Deque for history**: `collections.deque(maxlen=120)` gives a rolling 2-minute window without unbounded memory growth.
- **Session state**: Streamlit session state preserves data across reruns without a database.

---

## 🧪 Concepts Demonstrated

- ✅ Python + Pandas for all data processing
- ✅ Index construction via market-cap ranking
- ✅ Market-cap weighting with weight normalization
- ✅ Real-time index value computation
- ✅ Rebalancing with add/remove detection
- ✅ Sector aggregation (groupby + agg)
- ✅ Filtering, sorting, clipping with Pandas
- ✅ Interactive Streamlit dashboard
- ✅ Time-series chart and sector bar chart
- ✅ Rebalance audit log

---

## 🔮 Possible Extensions

- **Back-testing**: Load historical price data (CSV) and replay index construction over time
- **Custom weighting schemes**: Equal weight, free-float adjusted, factor-based
- **Multi-index comparison**: Run SMI-10 vs SMI-20 simultaneously
- **CSV export**: Download constituent snapshots at each rebalance
- **Factor overlays**: Add P/E or momentum scores for smart-beta style selection

---

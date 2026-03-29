"""
Page 1 — Data Preparation
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, date
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Data — FinSignal", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.stApp { background: #0d0d1a; color: #e0e0f0; }
[data-testid="stSidebar"] { background: #0a0a0f; border-right: 1px solid #1e1e2e; }
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
[data-testid="stMetric"] { background:#13131f; border:1px solid #252540; border-radius:12px; padding:16px !important; }
[data-testid="stMetricValue"] { color:#7fffb2 !important; font-family:'Syne',sans-serif !important; }
[data-testid="stMetricLabel"] { color:#8888aa !important; }
.stButton > button { background:linear-gradient(135deg,#7fffb2,#00c8ff); color:#0a0a0f; font-family:'Syne',sans-serif; font-weight:700; border:none; border-radius:8px; padding:10px 24px; }
.stSuccess { background:#0d2e1a !important; border-left:3px solid #7fffb2 !important; }
.stInfo    { background:#0d1a2e !important; border-left:3px solid #00c8ff !important; }
div[data-testid="stExpander"] { background:#13131f; border:1px solid #252540; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:1.5rem 0 0.5rem;">
  <span style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#7fffb2;">01 · Data Preparation</span><br>
  <span style="color:#8888aa;font-size:0.85rem;letter-spacing:0.08em;">DOWNLOAD · ALIGN · NORMALIZE</span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Configuration")

PRESET = {
    "RELIANCE": "RELIANCE.NS",
    "TCS":      "TCS.NS",
    "INFOSYS":  "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "WIPRO":    "WIPRO.NS",
}

selected = st.sidebar.multiselect(
    "Select Companies",
    options=list(PRESET.keys()),
    default=["RELIANCE", "TCS", "INFOSYS"],
)

custom_ticker = st.sidebar.text_input("Custom NSE ticker (e.g. TATAMOTORS.NS)", "")

col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("Start", value=date(2019, 1, 1))
end_date   = col_d2.date_input("End",   value=date.today())

normalize = st.sidebar.checkbox("Normalize prices (Min-Max)", value=True)

run_btn = st.sidebar.button("📥 Download & Prepare", use_container_width=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "stock_data" not in st.session_state:
    st.session_state.stock_data = {}

# ── Download ──────────────────────────────────────────────────────────────────
if run_btn:
    tickers = {k: PRESET[k] for k in selected}
    if custom_ticker.strip():
        name = custom_ticker.strip().replace(".NS", "").upper()
        tickers[name] = custom_ticker.strip()

    if not tickers:
        st.warning("Please select at least one company.")
        st.stop()

    progress = st.progress(0, text="Starting download…")
    stock_data = {}
    errors = []

    for i, (name, ticker) in enumerate(tickers.items()):
        progress.progress((i) / len(tickers), text=f"Downloading {name}…")
        try:
            df = yf.download(ticker, start=str(start_date), end=str(end_date), progress=False)
            if df.empty:
                errors.append(name)
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.rename(columns=str.lower, inplace=True)
            df.index = pd.to_datetime(df.index)
            df["daily_return"] = df["close"].pct_change()
            df["log_return"]   = np.log(df["close"] / df["close"].shift(1))
            df["volatility"]   = df["daily_return"].rolling(20).std() * np.sqrt(252)
            df.dropna(inplace=True)
            if normalize:
                scaler = MinMaxScaler()
                df["close_norm"] = scaler.fit_transform(df[["close"]])
            stock_data[name] = df
        except Exception as e:
            errors.append(f"{name} ({e})")

    progress.progress(1.0, text="Done!")

    if errors:
        st.error(f"Failed to download: {', '.join(errors)}")

    st.session_state.stock_data = stock_data
    if stock_data:
        st.success(f"✓ Downloaded {len(stock_data)} companies — {len(next(iter(stock_data.values())))} trading days each")

# ── Display ───────────────────────────────────────────────────────────────────
if not st.session_state.stock_data:
    st.info("👈  Configure settings in the sidebar and click **Download & Prepare** to begin.")
    st.stop()

data = st.session_state.stock_data
companies = list(data.keys())

# ── Summary metrics ───────────────────────────────────────────────────────────
st.markdown("### 📊 Dataset Overview")
mcols = st.columns(len(companies))
for col, name in zip(mcols, companies):
    df = data[name]
    with col:
        latest = df["close"].iloc[-1]
        ret    = df["daily_return"].mean() * 252 * 100
        st.metric(name, f"₹{latest:,.2f}", f"{ret:+.1f}% ann. return")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Series", "📉 Returns", "🔥 Correlation", "🗃️ Raw Data"])

with tab1:
    col_mode = st.radio("Display mode", ["Absolute Price", "Normalized (0–1)", "Indexed to 100"],
                        horizontal=True)
    fig = go.Figure()
    for name, df in data.items():
        if col_mode == "Normalized (0–1)" and "close_norm" in df.columns:
            y = df["close_norm"]
        elif col_mode == "Indexed to 100":
            y = df["close"] / df["close"].iloc[0] * 100
        else:
            y = df["close"]
        fig.add_trace(go.Scatter(x=df.index, y=y, name=name, mode="lines", line=dict(width=1.5)))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
        legend=dict(bgcolor="#13131f", bordercolor="#252540"),
        margin=dict(l=0, r=0, t=30, b=0), height=380,
        xaxis=dict(showgrid=True, gridcolor="#1e1e2e"),
        yaxis=dict(showgrid=True, gridcolor="#1e1e2e"),
        title=dict(text="Close Price History", font=dict(family="Syne", color="#7fffb2")),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    company_sel = st.selectbox("Company", companies, key="ret_sel")
    df = data[company_sel]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Daily Returns (%)", "Rolling 20-day Volatility (annualised)"),
                        vertical_spacing=0.12)

    fig.add_trace(go.Bar(x=df.index, y=df["daily_return"] * 100,
                         marker_color=np.where(df["daily_return"] >= 0, "#7fffb2", "#ff6b6b"),
                         name="Daily Return"), row=1, col=1)

    if "volatility" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["volatility"] * 100,
                                 line=dict(color="#00c8ff", width=1.5),
                                 name="Volatility"), row=2, col=1)

    fig.update_layout(template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
                      height=420, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if len(companies) < 2:
        st.info("Need at least 2 companies for correlation.")
    else:
        close_df = pd.DataFrame({n: d["close"] for n, d in data.items()}).dropna()
        corr = close_df.pct_change().dropna().corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                        zmin=-1, zmax=1, aspect="auto",
                        title="Return Correlation Matrix")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0d0d1a",
                          height=380, margin=dict(l=0, r=0, t=40, b=0),
                          title_font=dict(family="Syne", color="#7fffb2"))
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    name_sel = st.selectbox("Company", companies, key="raw_sel")
    df_show  = data[name_sel].copy()
    df_show.index = df_show.index.date
    st.dataframe(
        df_show.style.format({
            "open": "₹{:.2f}", "high": "₹{:.2f}", "low": "₹{:.2f}",
            "close": "₹{:.2f}", "volume": "{:,.0f}",
            "daily_return": "{:.4f}", "log_return": "{:.4f}",
        }),
        use_container_width=True, height=360,
    )
    st.download_button(
        "⬇️  Download CSV",
        data=data[name_sel].to_csv().encode(),
        file_name=f"{name_sel}_data.csv",
        mime="text/csv",
    )

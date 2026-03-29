"""
Stock Price Forecasting Dashboard
STFT Spectrogram + CNN — Assignment 2
"""

import streamlit as st

st.set_page_config(
    page_title="FinSignal — Stock Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0a0f;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }

/* Main background */
.stApp { background: #0d0d1a; color: #e0e0f0; }

/* Headings */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #13131f;
    border: 1px solid #252540;
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="stMetricValue"] { color: #7fffb2 !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricLabel"] { color: #8888aa !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7fffb2, #00c8ff);
    color: #0a0a0f;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Select / input boxes */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stSlider { color: #e0e0f0 !important; }

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #8888aa;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #7fffb2 !important;
    border-bottom: 2px solid #7fffb2 !important;
}

/* Info / success boxes */
.stSuccess { background: #0d2e1a !important; border-left: 3px solid #7fffb2 !important; }
.stInfo    { background: #0d1a2e !important; border-left: 3px solid #00c8ff !important; }
.stWarning { background: #2e200d !important; border-left: 3px solid #ffb347 !important; }
.stError   { background: #2e0d0d !important; border-left: 3px solid #ff6b6b !important; }

div[data-testid="stExpander"] {
    background: #13131f;
    border: 1px solid #252540;
    border-radius: 10px;
}

/* Code blocks */
code { color: #7fffb2 !important; background: #13131f !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2.5rem 0 1rem 0;">
  <div style="font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800;
              background:linear-gradient(90deg,#7fffb2,#00c8ff);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              line-height:1.1; margin-bottom:0.4rem;">
    FinSignal
  </div>
  <div style="color:#8888aa; font-size:1rem; letter-spacing:0.08em; text-transform:uppercase;">
    Pattern Recognition · Financial Time Series · STFT + CNN
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Navigation cards ──────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#8888aa;
            margin-bottom:1.2rem; letter-spacing:0.04em;">
  SELECT A MODULE FROM THE SIDEBAR →
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

cards = [
    ("01", "Data", "Download & prepare stock time series from NSE/BSE via Yahoo Finance.", "#7fffb2"),
    ("02", "Signal", "Apply FFT and STFT to generate spectrogram images.", "#00c8ff"),
    ("03", "Model", "Train a CNN regression model on spectrogram inputs.", "#ffb347"),
    ("04", "Analysis", "Evaluate predictions with MSE, RMSE, MAE, R².", "#ff6b9d"),
]

for col, (num, title, desc, color) in zip([col1, col2, col3, col4], cards):
    with col:
        st.markdown(f"""
        <div style="background:#13131f; border:1px solid #252540; border-radius:14px;
                    padding:1.4rem; min-height:160px; position:relative; overflow:hidden;">
          <div style="font-family:'Syne',sans-serif; font-size:2.5rem; font-weight:800;
                      color:{color}; opacity:0.15; position:absolute; top:8px; right:14px;">
            {num}
          </div>
          <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
                      color:{color}; margin-bottom:0.5rem;">{title}</div>
          <div style="color:#8888aa; font-size:0.82rem; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Quick-start info ──────────────────────────────────────────────────────────
with st.expander("ℹ️  How to use this dashboard"):
    st.markdown("""
**Step 1 — Data Preparation** *(sidebar → 01 Data)*
- Choose companies (RELIANCE, TCS, INFY, HDFCBANK, or custom ticker)
- Set date range and click **Download & Prepare**

**Step 2 — Signal Processing** *(sidebar → 02 Signal)*
- Choose window length and hop size
- View time-series, FFT spectrum, and STFT spectrogram

**Step 3 — Model Training** *(sidebar → 03 Model)*
- Configure CNN hyperparameters
- Train and watch live loss curves

**Step 4 — Analysis** *(sidebar → 04 Analysis)*
- Compare predictions vs actuals
- View residuals, metrics table, and feature importance
    """)

st.markdown("""
<div style="margin-top:2rem; color:#333355; font-size:0.75rem; letter-spacing:0.06em;">
  ASSIGNMENT 2 · PATTERN RECOGNITION · STFT + CNN STOCK FORECASTING
</div>
""", unsafe_allow_html=True)

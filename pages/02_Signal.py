"""
Page 2 — Signal Processing (FFT + STFT Spectrogram)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import stft
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Signal — FinSignal", page_icon="〰️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.stApp { background: #0d0d1a; color: #e0e0f0; }
[data-testid="stSidebar"] { background: #0a0a0f; border-right: 1px solid #1e1e2e; }
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
[data-testid="stMetric"] { background:#13131f; border:1px solid #252540; border-radius:12px; padding:16px !important; }
[data-testid="stMetricValue"] { color:#00c8ff !important; font-family:'Syne',sans-serif !important; }
[data-testid="stMetricLabel"] { color:#8888aa !important; }
.stButton > button { background:linear-gradient(135deg,#00c8ff,#7fffb2); color:#0a0a0f; font-family:'Syne',sans-serif; font-weight:700; border:none; border-radius:8px; padding:10px 24px; }
.stInfo { background:#0d1a2e !important; border-left:3px solid #00c8ff !important; }
div[data-testid="stExpander"] { background:#13131f; border:1px solid #252540; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:1.5rem 0 0.5rem;">
  <span style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#00c8ff;">02 · Signal Processing</span><br>
  <span style="color:#8888aa;font-size:0.85rem;letter-spacing:0.08em;">FFT · STFT · SPECTROGRAM</span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Check data ────────────────────────────────────────────────────────────────
if "stock_data" not in st.session_state or not st.session_state.stock_data:
    st.warning("⚠️  No data found. Please visit **01 Data** first and download stock data.")
    st.stop()

data      = st.session_state.stock_data
companies = list(data.keys())

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Signal Parameters")
company   = st.sidebar.selectbox("Company", companies)
signal_col = st.sidebar.selectbox("Signal", ["close", "daily_return", "log_return", "volume"])
window_len = st.sidebar.slider("Window Length L (days)", 8, 128, 64, step=8)
hop_size   = st.sidebar.slider("Hop Size H (days)", 1, window_len // 2, max(1, window_len // 8))
colormap   = st.sidebar.selectbox("Spectrogram colormap", ["Inferno", "Plasma", "Viridis", "Turbo", "Hot"])
log_scale  = st.sidebar.checkbox("Log-scale spectrogram (dB)", value=True)

df     = data[company]
signal = df[signal_col].values.astype(np.float64)

# Remove mean
signal_centered = signal - signal.mean()

# ── Metrics ───────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Signal Length",   f"{len(signal):,} days")
m2.metric("Window Length",   f"{window_len} days")
m3.metric("Hop Size",        f"{hop_size} days")
m4.metric("Overlap",         f"{window_len - hop_size} days")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Domain", "📡 Frequency Domain", "🌊 Spectrogram", "🔲 Window Comparison"])

# ── Tab 1: Time domain ────────────────────────────────────────────────────────
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=signal,
        mode="lines",
        line=dict(color="#00c8ff", width=1.2),
        name=signal_col,
    ))
    # Highlight one window
    w_start = len(signal) // 2
    w_end   = w_start + window_len
    fig.add_vrect(x0=df.index[w_start], x1=df.index[min(w_end, len(df)-1)],
                  fillcolor="#7fffb2", opacity=0.08,
                  annotation_text="sample window", annotation_position="top left",
                  annotation_font_color="#7fffb2")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
        height=350, margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"{company} — {signal_col} (time domain)", font=dict(family="Syne", color="#00c8ff")),
        xaxis=dict(showgrid=True, gridcolor="#1e1e2e"),
        yaxis=dict(showgrid=True, gridcolor="#1e1e2e"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About the sliding window"):
        st.markdown(f"""
**Window Length (L):** `{window_len}` days — each segment analysed independently.

**Hop Size (H):** `{hop_size}` days — window shifts by this amount each step.

**Overlap:** `{window_len - hop_size}` days between consecutive windows.

The STFT slides this window across the entire signal:
```
X[1 : L],  X[1+H : L+H],  X[1+2H : L+2H], …
```
        """)

# ── Tab 2: FFT ────────────────────────────────────────────────────────────────
with tab2:
    n  = len(signal_centered)
    yf = np.abs(fft(signal_centered))[:n // 2]
    xf = fftfreq(n, d=1)[:n // 2]          # cycles per day

    # Period axis
    period = np.where(xf > 0, 1 / (xf + 1e-12), np.nan)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Frequency Spectrum", "Period Spectrum (days)"))

    fig.add_trace(go.Scatter(x=xf, y=yf, mode="lines",
                              line=dict(color="#00c8ff", width=1.2), name="Magnitude"), row=1, col=1)
    fig.add_trace(go.Scatter(x=period, y=yf, mode="lines",
                              line=dict(color="#7fffb2", width=1.2), name="Period"), row=1, col=2)

    # Annotate dominant frequency
    peak_idx = np.argmax(yf[1:]) + 1
    fig.add_annotation(x=xf[peak_idx], y=yf[peak_idx],
                        text=f"Peak: {1/xf[peak_idx]:.0f}d cycle",
                        showarrow=True, arrowhead=2,
                        font=dict(color="#ffb347"), arrowcolor="#ffb347", row=1, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
        height=380, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e1e2e")
    fig.update_yaxes(showgrid=True, gridcolor="#1e1e2e")
    st.plotly_chart(fig, use_container_width=True)

    dominant_period = 1 / xf[peak_idx] if xf[peak_idx] > 0 else float("inf")
    st.info(f"**Dominant cycle:** ~{dominant_period:.0f} trading days  "
            f"≈ {dominant_period/5:.0f} weeks  ·  "
            f"Low-freq = long-term trend, high-freq = short-term noise")

# ── Tab 3: Spectrogram ────────────────────────────────────────────────────────
with tab3:
    noverlap = window_len - hop_size
    try:
        f_bins, t_frames, Zxx = stft(
            signal_centered, fs=1,
            window="hann",
            nperseg=window_len,
            noverlap=noverlap,
        )
        S = np.abs(Zxx) ** 2
        if log_scale:
            S_plot = 10 * np.log10(S + 1e-10)
            cb_label = "Power (dB)"
        else:
            S_plot = S
            cb_label = "Power"

        # Map t_frames (sample indices) back to dates
        t_dates = []
        for ti in t_frames:
            idx = min(int(ti), len(df.index) - 1)
            t_dates.append(df.index[idx])

        fig = go.Figure(go.Heatmap(
            x=t_dates,
            y=f_bins,
            z=S_plot,
            colorscale=colormap.lower(),
            colorbar=dict(title=cb_label, tickfont=dict(color="#8888aa")),
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
            height=420, margin=dict(l=0, r=0, t=40, b=0),
            title=dict(text=f"{company} — STFT Spectrogram S(t,f)",
                       font=dict(family="Syne", color="#00c8ff")),
            xaxis=dict(title="Date", showgrid=True, gridcolor="#1e1e2e"),
            yaxis=dict(title="Frequency (cycles/day)", showgrid=True, gridcolor="#1e1e2e"),
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            # Mean power vs frequency
            mean_power = S_plot.mean(axis=1)
            fig2 = go.Figure(go.Scatter(
                x=mean_power, y=f_bins, mode="lines",
                line=dict(color="#ff6b9d", width=1.5),
            ))
            fig2.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
                height=280, margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text="Mean Power vs Frequency", font=dict(family="Syne", size=13, color="#ff6b9d")),
                xaxis=dict(title=cb_label, showgrid=True, gridcolor="#1e1e2e"),
                yaxis=dict(title="Frequency", showgrid=True, gridcolor="#1e1e2e"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            # Power over time (mean over frequencies)
            mean_time = S_plot.mean(axis=0)
            fig3 = go.Figure(go.Scatter(
                x=t_dates, y=mean_time, mode="lines",
                line=dict(color="#ffb347", width=1.5),
                fill="tozeroy", fillcolor="rgba(255,179,71,0.07)",
            ))
            fig3.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
                height=280, margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text="Mean Power over Time", font=dict(family="Syne", size=13, color="#ffb347")),
                xaxis=dict(showgrid=True, gridcolor="#1e1e2e"),
                yaxis=dict(title=cb_label, showgrid=True, gridcolor="#1e1e2e"),
            )
            st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"STFT failed: {e}. Try increasing the window length or selecting a different signal.")

# ── Tab 4: Window comparison ──────────────────────────────────────────────────
with tab4:
    st.markdown("**How window length affects time vs frequency resolution:**")
    window_sizes = [16, 32, 64, 128]
    cols = st.columns(4)

    for col, wl in zip(cols, window_sizes):
        try:
            hl = max(1, wl // 8)
            _, _, Zxx_w = stft(signal_centered, fs=1, window="hann",
                                nperseg=wl, noverlap=wl - hl)
            S_w = 10 * np.log10(np.abs(Zxx_w) ** 2 + 1e-10)
            fig_w = go.Figure(go.Heatmap(z=S_w, colorscale="inferno", showscale=False))
            res = "↑ freq res" if wl >= 64 else "↑ time res"
            fig_w.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
                height=220, margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text=f"L={wl}  {res}", font=dict(family="Syne", size=12, color="#8888aa")),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
            )
            with col:
                st.plotly_chart(fig_w, use_container_width=True)
        except Exception:
            with col:
                st.write(f"L={wl} — not enough data")

    st.info(
        "**Resolution trade-off:**  "
        "Larger window → sharper frequency resolution (better trend detection)  |  "
        "Smaller window → sharper time resolution (better at catching sudden events)"
    )

# ── Save spectrogram data to session ─────────────────────────────────────────
try:
    noverlap = window_len - hop_size
    _, _, Zxx_save = stft(signal_centered, fs=1, window="hann",
                           nperseg=window_len, noverlap=noverlap)
    S_save = np.abs(Zxx_save) ** 2
    S_norm = (S_save - S_save.min()) / (S_save.max() - S_save.min() + 1e-10)
    st.session_state["spectrogram"]         = S_norm.astype(np.float32)
    st.session_state["spectrogram_company"] = company
    st.session_state["spectrogram_signal"]  = signal_col
    st.session_state["window_len"]          = window_len
    st.session_state["hop_size"]            = hop_size
    st.session_state["target_signal"]       = df["close"].values
except Exception:
    pass

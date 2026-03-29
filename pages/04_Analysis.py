"""
Page 4 — Analysis & Evaluation
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Analysis — FinSignal", page_icon="📉", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.stApp { background: #0d0d1a; color: #e0e0f0; }
[data-testid="stSidebar"] { background: #0a0a0f; border-right: 1px solid #1e1e2e; }
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
[data-testid="stMetric"] { background:#13131f; border:1px solid #252540; border-radius:12px; padding:16px !important; }
[data-testid="stMetricValue"] { color:#ff6b9d !important; font-family:'Syne',sans-serif !important; }
[data-testid="stMetricLabel"] { color:#8888aa !important; }
div[data-testid="stExpander"] { background:#13131f; border:1px solid #252540; border-radius:10px; }
.stSuccess { background:#0d2e1a !important; border-left:3px solid #7fffb2 !important; }
.stInfo    { background:#0d1a2e !important; border-left:3px solid #00c8ff !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:1.5rem 0 0.5rem;">
  <span style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#ff6b9d;">04 · Analysis</span><br>
  <span style="color:#8888aa;font-size:0.85rem;letter-spacing:0.08em;">EVALUATION · RESIDUALS · FEATURE IMPORTANCE</span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Check prerequisites ───────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.warning("⚠️  No model results found. Go to **03 Model** and train a model first.")
    st.stop()

if "stock_data" not in st.session_state or not st.session_state.stock_data:
    st.warning("⚠️  No data found. Go to **01 Data** first.")
    st.stop()

r       = st.session_state["results"]
data    = st.session_state.stock_data
company = r["company"]
preds   = r["preds"]
targets = r["targets"]
metrics = r["metrics"]
errors  = targets - preds

# ── Summary metrics ───────────────────────────────────────────────────────────
st.markdown(f"### 📊 Model Performance — {company}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("MSE",       f"{metrics['mse']:.2f}")
c2.metric("RMSE",      f"₹{metrics['rmse']:.2f}")
c3.metric("MAE",       f"₹{metrics['mae']:.2f}")
c4.metric("R²",        f"{metrics['r2']:.4f}")
c5.metric("Mean Error",f"₹{errors.mean():.2f}")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Predictions", "🔎 Residuals", "📊 Error Distribution",
    "🧩 Feature Correlation", "📄 Summary Report"
])

# ── Tab 1: Predictions ────────────────────────────────────────────────────────
with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Predicted vs Actual", "Absolute Error"),
                        row_heights=[0.7, 0.3], vertical_spacing=0.1)

    fig.add_trace(go.Scatter(y=targets, mode="lines", name="Actual",
                              line=dict(color="#7fffb2", width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(y=preds, mode="lines", name="Predicted",
                              line=dict(color="#ff6b9d", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(
        y=np.concatenate([preds, preds[::-1]]),
        x=list(range(len(preds))) + list(range(len(preds) - 1, -1, -1)),
        fill="toself", fillcolor="rgba(255,107,157,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="Prediction band", showlegend=False,
    ), row=1, col=1)

    abs_err = np.abs(errors)
    fig.add_trace(go.Bar(y=abs_err, marker_color=np.where(errors >= 0, "#7fffb2", "#ff6b6b"),
                          name="Abs Error"), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
        height=500, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(bgcolor="#13131f", bordercolor="#252540"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e1e2e")
    fig.update_yaxes(showgrid=True, gridcolor="#1e1e2e")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter
    mn, mx = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                 line=dict(color="#555577", dash="dash", width=1.5), name="Perfect"))
    fig_sc.add_trace(go.Scatter(x=targets, y=preds, mode="markers",
                                 marker=dict(color=abs_err, colorscale="RdYlGn_r",
                                             size=7, opacity=0.7,
                                             colorbar=dict(title="Abs Error ₹")),
                                 name="Samples"))
    fig_sc.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
        height=380, margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Predicted vs Actual (coloured by error)", font=dict(family="Syne", color="#ff6b9d")),
        xaxis=dict(title="Actual (₹)",    showgrid=True, gridcolor="#1e1e2e"),
        yaxis=dict(title="Predicted (₹)", showgrid=True, gridcolor="#1e1e2e"),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Tab 2: Residuals ──────────────────────────────────────────────────────────
with tab2:
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(y=errors, mode="lines+markers",
                                line=dict(color="#8888aa", width=0.8),
                                marker=dict(color=np.where(errors >= 0, "#7fffb2", "#ff6b6b"),
                                            size=4),
                                name="Residual"))
    fig_r.add_hline(y=0, line_dash="dash", line_color="#ff6b6b", line_width=1.5)
    fig_r.add_hline(y=errors.mean(), line_dash="dot", line_color="#ffb347",
                     annotation_text=f"Mean: {errors.mean():.2f}", line_width=1.2)

    fig_r.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
        height=360, margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Residuals (Actual − Predicted)", font=dict(family="Syne", color="#ff6b9d")),
        xaxis=dict(title="Sample", showgrid=True, gridcolor="#1e1e2e"),
        yaxis=dict(title="Error (₹)", showgrid=True, gridcolor="#1e1e2e"),
    )
    st.plotly_chart(fig_r, use_container_width=True)

    col_a, col_b = st.columns(2)
    col_a.metric("Max Over-prediction",  f"₹{errors.max():.2f}")
    col_b.metric("Max Under-prediction", f"₹{errors.min():.2f}")
    col_a.metric("Std of Errors",        f"₹{errors.std():.2f}")
    col_b.metric("% within ±RMSE",
                  f"{(np.abs(errors) <= metrics['rmse']).mean()*100:.1f}%")

# ── Tab 3: Error distribution ─────────────────────────────────────────────────
with tab3:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=errors, nbinsx=30,
        marker=dict(color="#ff6b9d", opacity=0.75,
                    line=dict(color="#0d0d1a", width=0.5)),
        name="Error",
    ))
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#7fffb2",
                        annotation_text="Zero error", line_width=1.5)
    fig_hist.add_vline(x=errors.mean(), line_dash="dot", line_color="#ffb347",
                        annotation_text=f"Mean {errors.mean():.2f}", line_width=1.2)

    fig_hist.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
        height=360, margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Error Distribution", font=dict(family="Syne", color="#ff6b9d")),
        xaxis=dict(title="Prediction Error (₹)", showgrid=True, gridcolor="#1e1e2e"),
        yaxis=dict(title="Count", showgrid=True, gridcolor="#1e1e2e"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Percentiles
    st.markdown("**Error Percentiles**")
    p_df = pd.DataFrame({
        "Percentile": ["10th", "25th", "50th", "75th", "90th", "95th"],
        "Error (₹)":  np.percentile(errors, [10, 25, 50, 75, 90, 95]).round(2),
        "|Error| (₹)": np.percentile(np.abs(errors), [10, 25, 50, 75, 90, 95]).round(2),
    })
    st.dataframe(p_df, use_container_width=True, hide_index=True)

# ── Tab 4: Feature correlation ────────────────────────────────────────────────
with tab4:
    if company in data:
        df = data[company]
        feat_cols = [c for c in ["open","high","low","close","volume","daily_return","log_return","volatility"] if c in df.columns]
        corr = df[feat_cols].corr()["close"].drop("close").abs().sort_values(ascending=True)

        fig_feat = go.Figure(go.Bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            marker=dict(
                color=corr.values,
                colorscale="Teal",
                line=dict(color="#0d0d1a", width=0.5),
            ),
        ))
        fig_feat.add_vline(x=0.5, line_dash="dash", line_color="#ffb347",
                            annotation_text="r = 0.5", line_width=1.2)
        fig_feat.update_layout(
            template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
            height=350, margin=dict(l=0, r=0, t=30, b=0),
            title=dict(text=f"{company} — |Correlation| with Close Price",
                       font=dict(family="Syne", color="#ff6b9d")),
            xaxis=dict(title="|Pearson r|", showgrid=True, gridcolor="#1e1e2e", range=[0, 1]),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_feat, use_container_width=True)

        # Pairwise scatter for top features
        top_feats = corr.nlargest(3).index.tolist() + ["close"]
        fig_pair = px.scatter_matrix(
            df[top_feats].dropna().sample(min(500, len(df))),
            dimensions=top_feats,
            color_continuous_scale="teal",
            title="Pairwise Feature Scatter (sample 500 points)",
        )
        fig_pair.update_layout(
            template="plotly_dark", paper_bgcolor="#0d0d1a",
            height=450, margin=dict(l=0, r=0, t=40, b=0),
            title_font=dict(family="Syne", color="#ff6b9d"),
        )
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        st.info(f"Data for {company} not available in current session.")

# ── Tab 5: Summary report ─────────────────────────────────────────────────────
with tab5:
    m = metrics
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║   ASSIGNMENT 2 — RESULTS SUMMARY REPORT                     ║
║   Pattern Recognition for Financial Time Series             ║
╚══════════════════════════════════════════════════════════════╝

  Company   : {company}
  Model     : CNN-based Spectrogram Regression
  Pipeline  : OHLCV → STFT Spectrogram → CNN → Price Prediction

  ── Test Set Metrics ──────────────────────────────────────────
  MSE          : {m['mse']:.4f}
  RMSE         : ₹{m['rmse']:.4f}
  MAE          : ₹{m['mae']:.4f}
  R²           : {m['r2']:.4f}
  Mean Error   : ₹{errors.mean():.4f}
  Std Error    : ₹{errors.std():.4f}

  ── Observations ──────────────────────────────────────────────
  • Financial time series analysed as a non-stationary signal.
  • STFT spectrograms capture time-varying frequency content.
  • Low-frequency bins → long-term price trends.
  • High-frequency bins → short-term volatility / noise.
  • CNN learns spatial patterns in 2D time-frequency images.
  • R² = {m['r2']:.4f} → {"strong" if m["r2"] > 0.7 else "moderate" if m["r2"] > 0.4 else "weak"} predictive ability on test set.

  ── References ────────────────────────────────────────────────
  [1] Y. Zhang & C. Aggarwal — "Stock Market Prediction Using
      Deep Learning," IEEE Access.
  [2] A. Tsantekidis et al. — "Deep Learning for Financial
      Time Series Forecasting."
  [3] S. Hochreiter & J. Schmidhuber — "LSTM," Neural
      Computation, 1997.
  [4] A. Borovykh et al. — "Conditional Time Series
      Forecasting with CNNs."

══════════════════════════════════════════════════════════════
"""
    st.code(report, language=None)
    st.download_button(
        "⬇️  Download Report (.txt)",
        data=report.encode(),
        file_name="assignment2_summary_report.txt",
        mime="text/plain",
    )

    # Metrics table for all companies if trained
    st.markdown("### All Companies")
    st.info("Train models for each company via **03 Model** to populate this table.")
    df_metrics = pd.DataFrame([{
        "Company": r["company"],
        "MSE": f"{m['mse']:.2f}",
        "RMSE": f"₹{m['rmse']:.2f}",
        "MAE": f"₹{m['mae']:.2f}",
        "R²": f"{m['r2']:.4f}",
    }])
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

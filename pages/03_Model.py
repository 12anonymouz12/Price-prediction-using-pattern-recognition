"""
Page 3 — CNN Model Training
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import stft
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Model — FinSignal", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.stApp { background: #0d0d1a; color: #e0e0f0; }
[data-testid="stSidebar"] { background: #0a0a0f; border-right: 1px solid #1e1e2e; }
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
[data-testid="stMetric"] { background:#13131f; border:1px solid #252540; border-radius:12px; padding:16px !important; }
[data-testid="stMetricValue"] { color:#ffb347 !important; font-family:'Syne',sans-serif !important; }
[data-testid="stMetricLabel"] { color:#8888aa !important; }
.stButton > button { background:linear-gradient(135deg,#ffb347,#ff6b9d); color:#0a0a0f; font-family:'Syne',sans-serif; font-weight:700; border:none; border-radius:8px; padding:10px 24px; }
.stSuccess { background:#0d2e1a !important; border-left:3px solid #7fffb2 !important; }
.stInfo    { background:#0d1a2e !important; border-left:3px solid #00c8ff !important; }
div[data-testid="stExpander"] { background:#13131f; border:1px solid #252540; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:1.5rem 0 0.5rem;">
  <span style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#ffb347;">03 · CNN Model</span><br>
  <span style="color:#8888aa;font-size:0.85rem;letter-spacing:0.08em;">ARCHITECTURE · TRAINING · PREDICTION</span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Check prerequisites ───────────────────────────────────────────────────────
if "stock_data" not in st.session_state or not st.session_state.stock_data:
    st.warning("⚠️  No data found. Go to **01 Data** first.")
    st.stop()

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader, random_split
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    st.error("PyTorch not installed. Run: `pip install torch`")
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Model Hyperparameters")
data      = st.session_state.stock_data
companies = list(data.keys())
company   = st.sidebar.selectbox("Company", companies)

epochs      = st.sidebar.slider("Epochs",        10, 150,  50, step=5)
batch_size  = st.sidebar.slider("Batch Size",     8,  128,  32, step=8)
lr          = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
dropout     = st.sidebar.slider("Dropout",        0.0, 0.5, 0.3, step=0.05)
pred_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 5)
window_len  = st.session_state.get("window_len", 64)
hop_size    = st.session_state.get("hop_size", 8)

train_btn = st.sidebar.button("🚀 Train Model", use_container_width=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.markdown(f"**Device:** `{DEVICE}`")

# ── Architecture diagram ──────────────────────────────────────────────────────
with st.expander("🏗️ CNN Architecture", expanded=False):
    st.markdown(f"""
```
Input  : (B, 1, freq_bins, time_frames)
         ↓
┌─────────────────────────────────────┐
│  Conv Block 1                       │
│  Conv2d(1→32, 3×3) → BN → ReLU     │
│  Conv2d(32→32, 3×3) → BN → ReLU    │
│  MaxPool2d(2×2) → Dropout2d({dropout:.1f})  │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Conv Block 2                       │
│  Conv2d(32→64, 3×3) → BN → ReLU    │
│  Conv2d(64→64, 3×3) → BN → ReLU    │
│  MaxPool2d(2×2) → Dropout2d         │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Conv Block 3                       │
│  Conv2d(64→128, 3×3) → BN → ReLU   │
│  MaxPool2d(2×2) → Dropout2d         │
└─────────────────────────────────────┘
         ↓
  AdaptiveAvgPool2d(4×4)
         ↓
  Flatten → Linear(2048→256) → ReLU → Dropout
         ↓
  Linear(256→64) → ReLU → Dropout
         ↓
  Linear(64→1) → Sigmoid
         ↓
Output : ŷ ∈ (0,1)  [denorm → price ₹]

Prediction horizon : {pred_horizon} trading days
```
    """)

# ── Helper: build spectrogram dataset ────────────────────────────────────────
def build_dataset(signal, close_prices, window_length, hop, horizon):
    X_list, y_list = [], []
    n = len(signal)
    seg_nperseg = max(4, window_length // 4)
    for start in range(0, n - window_length - horizon, hop):
        end = start + window_length
        if end + horizon - 1 >= n:
            break
        seg = signal[start:end] - signal[start:end].mean()
        _, _, Zxx = stft(seg, fs=1, window="hann",
                          nperseg=seg_nperseg,
                          noverlap=seg_nperseg - 1)
        S = np.abs(Zxx) ** 2
        S_norm = (S - S.min()) / (S.max() - S.min() + 1e-10)
        X_list.append(S_norm.astype(np.float32))
        y_list.append(close_prices[end + horizon - 1])
    return np.array(X_list), np.array(y_list)


# ── CNN Model ─────────────────────────────────────────────────────────────────
class SpectrogramCNN(nn.Module):
    def __init__(self, dp=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(dp / 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(dp / 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(dp / 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(True), nn.Dropout(dp),
            nn.Linear(256, 64), nn.ReLU(True), nn.Dropout(dp / 2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.pool(self.block3(self.block2(self.block1(x))))).squeeze(1)


# ── Training ──────────────────────────────────────────────────────────────────
if train_btn:
    df     = data[company]
    signal = df["close"].values.astype(np.float64)
    close  = df["close"].values.astype(np.float64)

    with st.spinner("Building spectrogram dataset…"):
        X, y = build_dataset(signal, close, window_len, hop_size, pred_horizon)

    if len(X) < 30:
        st.error(f"Not enough samples ({len(X)}). Reduce window length or prediction horizon, or download more data.")
        st.stop()

    st.info(f"Dataset: **{len(X)} samples** | Spectrogram shape: **{X.shape[1:]}** | Target horizon: **{pred_horizon} days**")

    # Normalise targets
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-10)

    # Split
    n        = len(X)
    n_test   = max(5, int(n * 0.15))
    n_val    = max(5, int(n * 0.15))
    n_train  = n - n_val - n_test

    X_t = torch.tensor(X[:, np.newaxis], dtype=torch.float32)
    y_t = torch.tensor(y_norm,           dtype=torch.float32)

    ds = TensorDataset(X_t, y_t)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model     = SpectrogramCNN(dp=dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    st.markdown(f"**Trainable parameters:** `{n_params:,}`")

    # ── Live training UI ──────────────────────────────────────────────────────
    prog_bar    = st.progress(0, text="Training…")
    chart_ph    = st.empty()
    metrics_ph  = st.empty()

    train_losses, val_losses = [], []
    best_val, best_state, no_improve = float("inf"), None, 0
    patience = 12

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl_loss += criterion(model(xb), yb).item() * len(xb)
        vl_loss /= len(val_loader.dataset)

        scheduler.step(vl_loss)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_loss < best_val:
            best_val   = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # Update UI every 2 epochs
        if epoch % 2 == 0 or epoch == epochs:
            prog_bar.progress(epoch / epochs, text=f"Epoch {epoch}/{epochs} — Val MSE: {vl_loss:.6f}")
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(y=train_losses, mode="lines", name="Train MSE",
                                           line=dict(color="#ffb347", width=1.5)))
            fig_live.add_trace(go.Scatter(y=val_losses,   mode="lines", name="Val MSE",
                                           line=dict(color="#ff6b9d", width=1.5, dash="dash")))
            fig_live.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
                height=280, margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text="Training Loss", font=dict(family="Syne", color="#ffb347")),
                xaxis=dict(title="Epoch", showgrid=True, gridcolor="#1e1e2e"),
                yaxis=dict(title="MSE",   showgrid=True, gridcolor="#1e1e2e"),
                legend=dict(bgcolor="#13131f"),
            )
            chart_ph.plotly_chart(fig_live, use_container_width=True)

        if no_improve >= patience:
            prog_bar.progress(1.0, text=f"Early stop at epoch {epoch}")
            break

    # ── Test evaluation ────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_preds.extend(model(xb.to(DEVICE)).cpu().numpy())
            all_targets.extend(yb.numpy())

    preds_norm   = np.array(all_preds)
    targets_norm = np.array(all_targets)
    preds   = preds_norm   * (y_max - y_min) + y_min
    targets = targets_norm * (y_max - y_min) + y_min

    mse  = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(targets, preds)
    r2   = r2_score(targets, preds)

    # Store results
    st.session_state["results"] = {
        "company": company, "preds": preds, "targets": targets,
        "metrics": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
        "history": {"train_loss": train_losses, "val_loss": val_losses},
    }

    # ── Metrics ────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Test Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSE",  f"{mse:.2f}")
    c2.metric("RMSE", f"₹{rmse:.2f}")
    c3.metric("MAE",  f"₹{mae:.2f}")
    c4.metric("R²",   f"{r2:.4f}")

    # ── Predictions chart ──────────────────────────────────────────────────────
    st.markdown("### 📈 Predictions vs Actual")
    fig_pred = make_subplots(rows=1, cols=2,
                              subplot_titles=("Time Series", "Scatter Plot"))
    fig_pred.add_trace(go.Scatter(y=targets, mode="lines", name="Actual",
                                   line=dict(color="#7fffb2", width=1.5)), row=1, col=1)
    fig_pred.add_trace(go.Scatter(y=preds, mode="lines", name="Predicted",
                                   line=dict(color="#ff6b9d", width=1.2, dash="dot")), row=1, col=1)

    mn = min(targets.min(), preds.min())
    mx = max(targets.max(), preds.max())
    fig_pred.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                   line=dict(color="#555577", dash="dash"), name="Perfect"), row=1, col=2)
    fig_pred.add_trace(go.Scatter(x=targets, y=preds, mode="markers",
                                   marker=dict(color="#ffb347", size=5, opacity=0.6),
                                   name="Predicted"), row=1, col=2)

    fig_pred.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
        height=380, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(bgcolor="#13131f", bordercolor="#252540"),
    )
    fig_pred.update_xaxes(showgrid=True, gridcolor="#1e1e2e")
    fig_pred.update_yaxes(showgrid=True, gridcolor="#1e1e2e")
    st.plotly_chart(fig_pred, use_container_width=True)

    st.success(f"✓ Training complete! Model saved to session. Go to **04 Analysis** for deeper evaluation.")

elif "results" in st.session_state:
    r = st.session_state["results"]
    st.info(f"Showing previous results for **{r['company']}**. Click **Train Model** to retrain.")
    m = r["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSE",  f"{m['mse']:.2f}")
    c2.metric("RMSE", f"₹{m['rmse']:.2f}")
    c3.metric("MAE",  f"₹{m['mae']:.2f}")
    c4.metric("R²",   f"{m['r2']:.4f}")

    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(y=r["history"]["train_loss"], name="Train MSE",
                                line=dict(color="#ffb347", width=1.5)))
    fig_h.add_trace(go.Scatter(y=r["history"]["val_loss"],   name="Val MSE",
                                line=dict(color="#ff6b9d", width=1.5, dash="dash")))
    fig_h.update_layout(template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#13131f",
                         height=280, margin=dict(l=0,r=0,t=30,b=0),
                         title=dict(text="Training History", font=dict(family="Syne", color="#ffb347")))
    st.plotly_chart(fig_h, use_container_width=True)
else:
    st.info("👈  Configure hyperparameters in the sidebar and click **Train Model** to start.")

# 📈 FinSignal — Stock Forecasting Dashboard
>  **Done by - Sahil Shaji s4 cse**

> **Assignment 2** · Interactive Streamlit dashboard for  
> Pattern Recognition in Financial Time Series using **STFT + CNN**


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)


## 🌐 Live Demo

> After deploying, your link will be:  
> https://price-prediction-using-pattern-recognition.streamlit.app

---

## 🗂️ Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Overview and navigation |
| 📊 01 Data | Download NSE/BSE stocks, align, normalize |
| 〰️ 02 Signal | FFT spectrum + interactive STFT spectrogram |
| 🧠 03 Model | Train CNN live with loss curves |
| 📉 04 Analysis | MSE/RMSE/MAE/R², residuals, feature importance |

---

## 📐 Methodology

### Signal Representation
Financial time series treated as multivariate signal:
```
X(t) = [p(t), r(t), g(t), s(t), d(t)]
```
where p = close price, r = revenue indicator, g = growth (return), s = index, d = daily change.

### STFT (Short-Time Fourier Transform)
```
STFT(t, f) = ∫ X(τ) · w(τ − t) · e^(−j2πfτ) dτ
```

Spectrogram:
```
S(t, f) = |STFT(t, f)|²
```

**Key parameters:**
| Parameter | Value |
|-----------|-------|
| Window Length (L) | 64 trading days |
| Hop Size (H) | 8 trading days |
| Overlap | 56 days |
| Sampling freq | 1 sample/day |

### CNN Architecture

```
Input: (B, 1, freq_bins, time_frames)
    ↓
Conv Block 1: Conv2d(1→32) → BN → ReLU → MaxPool → Dropout
    ↓
Conv Block 2: Conv2d(32→64) → BN → ReLU → MaxPool → Dropout
    ↓
Conv Block 3: Conv2d(64→128) → BN → ReLU → MaxPool → Dropout
    ↓
AdaptiveAvgPool2d(4×4)
    ↓
FC: Linear(2048→256) → ReLU → Dropout
    ↓
FC: Linear(256→64) → ReLU → Dropout
    ↓
Linear(64→1) → Sigmoid
    ↓
Output: ŷ ∈ (0,1)  [denormalized → price in ₹]
```

## 🔧 How It Works

```
Yahoo Finance
     ↓
OHLCV Time Series  (Task 1)
     ↓
FFT + STFT Spectrogram  (Task 2)
S(t, f) = |STFT(t, f)|²
     ↓
CNN Regression Model  (Task 3)
p̂(t + Δt) = f_θ(Sₜ)
     ↓
Evaluation: MSE, RMSE, MAE, R²  (Task 4)
```

---

## 📦 Stack

| Tool | Purpose |
|------|---------|
| `streamlit` | Web dashboard |
| `yfinance` | Stock data |
| `scipy` | FFT + STFT |
| `torch` | CNN model |
| `plotly` | Interactive charts |
| `scikit-learn` | Metrics |

# 📈 FinSignal — Stock Forecasting Dashboard

> **Assignment 2** · Interactive Streamlit dashboard for  
> Pattern Recognition in Financial Time Series using **STFT + CNN**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🌐 Live Demo

> After deploying, your link will be:  
> `https://<your-app-name>.streamlit.app`

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

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "FinSignal: STFT + CNN Stock Dashboard"
git remote add origin https://github.com/<your-username>/finsignal.git
git branch -M main
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Select your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy** ✅

That's it — your dashboard will be live in ~2 minutes!

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

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

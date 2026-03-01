# Economic Stress Index (ESI) — NLP Dashboard

A full **macroeconomic analysis pipeline + AI-powered conversational dashboard**
built with Streamlit, LangChain, and Gemini 2.5 Flash.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the data pipeline (first time only)
```bash
python run_pipeline.py          # all 14 steps
# or just the core US steps:
python run_pipeline.py 1 2 3 4 5 6 7 8 9 10
```
This downloads FRED data and generates all CSVs / charts used by the dashboard.

### 3. Launch the dashboard
```bash
streamlit run dashboard.py
```
Open your browser at **http://localhost:8501**

---

## 🔑 How to Get Your FREE Gemini API Key

The AI chat assistant uses **Gemini 2.5 Flash** — Google's fastest free model.

**Steps:**
1. Go to **https://aistudio.google.com/app/apikey**
2. Sign in with your Google account (free, no credit card)
3. Click **"Create API Key"**
4. Copy the key (starts with `AIza…`)
5. Open the dashboard → paste the key in the **left sidebar** field

> ✅ No credit card required. The free tier provides generous usage.

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 📈 ESI Overview | Composite stress index, engine scores, annual heatmap |
| 🔩 Engine Deep Dive | Per-engine analysis with sub-indicators and correlations |
| 🤖 ML Forecast | 6-month recession probability from XGBoost / LightGBM |
| 🌍 Regime Analysis | GMM stress regimes, timeline, scatter plots |
| 💬 AI Chat | Gemini 2.5 Flash with full project data as context |

---

## 🤖 AI Chat Capabilities

The assistant knows your project's:
- Current ESI score and stress level
- All 5 engine scores and their recent history
- 6-month recession probability from the ML backtest
- Current stress regime from the GMM classifier
- ML model performance (AUC, AP score, Brier)

Sample questions:
- "What is the current ESI and stress level?"
- "Which engine is most elevated right now?"
- "What is the recession probability for the next 6 months?"
- "Compare today's stress to the 2008 financial crisis"
- "What regime are we in and what typically follows?"
- "Explain the Dynamic Factor Model in simple terms"

---

## 🔐 API Keys

| Key | Where | Purpose |
|-----|-------|---------|
| Gemini API | Dashboard sidebar (runtime) | AI chat |
| FRED API   | config.py → FRED_API_KEY   | Data download |

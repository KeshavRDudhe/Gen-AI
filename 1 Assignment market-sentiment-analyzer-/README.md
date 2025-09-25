# Real‑Time Market Sentiment Analyzer (LangChain + Gemini + MLflow)

A reference implementation using **LangChain**, **Google Gemini‑2.0‑flash**, **mlflow**, and a pluggable **news search** layer (Exa / Tavily / Brave).

## Features
- Company → Ticker resolution (static map → Wikipedia hint → optional yfinance validation)
- Recent news fetching via Exa/Tavily/Brave (first available), fallback to Wikipedia
- Structured JSON output via Pydantic schema (people/places/other companies/industries/implications/confidence)
- Full logging to mlflow (params, artifacts, simple span metrics)
- Optional Streamlit UI

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U -r requirements.txt
cp .env.example .env  # edit with your keys
python sentiment_chain.py --company "Google" --top_k 6 --days 7 --verbose
```

### Streamlit UI
```bash
pip install streamlit
streamlit run app.py
```

## Environment Variables
Copy `.env.example` to `.env` and fill as needed:

```env
# --- LLM ---
GOOGLE_API_KEY=your_google_api_key

# --- mlflow ---
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=market-sentiment-analyzer

# --- Search (use any one or more) ---
EXA_API_KEY=
TAVILY_API_KEY=
BRAVE_API_KEY=

# --- Options ---
NEWS_LOOKBACK_DAYS=7
NEWS_TOP_K=6
LLM_NAME=gemini-2.0-flash
```

## Example Output
See `sample_output.json` for the shape returned by the chain.

## MLflow
Start a local server (optional):
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
The script logs params and a `sentiment.json` artifact per run.

## Project Structure
```
market-sentiment-analyzer/
├── app.py
├── sentiment_chain.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── sample_output.json
├── tests/
│   └── test_ticker.py
└── extras/
    └── RAG_ALL.py  # (your uploaded file, preserved as-is)
```

## GitHub: How to Publish
```bash
git init
git add .
git commit -m "Initial commit: Market Sentiment Analyzer"
git branch -M main
git remote add origin https://github.com/<your-username>/market-sentiment-analyzer.git
git push -u origin main
```

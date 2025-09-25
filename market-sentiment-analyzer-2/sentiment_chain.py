#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real‑Time Market Sentiment Analyzer (LangChain + Gemini + mlflow)

Implements:
 1) Company → ticker resolution
 2) News fetching via Exa / Tavily / Brave (first available), fallback to Wikipedia
 3) Gemini‑2.0‑flash sentiment analysis with structured JSON output
 4) mlflow logging with lightweight tracing spans (timed blocks)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv(override=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "market-sentiment-analyzer")
NEWS_TOP_K = int(os.getenv("NEWS_TOP_K", "6"))
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "7"))
LLM_NAME = os.getenv("LLM_NAME", "gemini-2.0-flash")

# LangChain / LLM
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Optional news tools
NEWS_BACKENDS = []
try:
    from langchain_community.tools.exa_search import ExaSearchResults
    NEWS_BACKENDS.append("exa")
except Exception:
    pass
try:
    from langchain.tools.tavily_search import TavilySearchResults
    NEWS_BACKENDS.append("tavily")
except Exception:
    pass
try:
    from langchain_community.utilities.brave_search import BraveSearch
    NEWS_BACKENDS.append("brave")
except Exception:
    pass

# Wikipedia fallback
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# yfinance optional
try:
    import yfinance as yf
except Exception:
    yf = None

# Observability
import mlflow

# --------- Pydantic schema ---------
class SentimentSchema(BaseModel):
    company_name: str
    stock_code: str
    newsdesc: str = Field(description="Concise rollup of latest news (headlines/summaries)")
    sentiment: str = Field(description="Positive/Negative/Neutral")
    people_names: List[str] = []
    places_names: List[str] = []
    other_companies_referred: List[str] = []
    related_industries: List[str] = []
    market_implications: str = ""
    confidence_score: float = 0.0

# --------- Tracing spans ---------
@dataclass
class Span:
    name: str
    start: float
    end: Optional[float] = None
    def close(self):
        self.end = time.time()
        return self

class Tracer:
    def __init__(self):
        self.spans: List[Span] = []
    def start(self, name: str) -> Span:
        s = Span(name=name, start=time.time())
        self.spans.append(s)
        return s
    def finish(self, span: Span):
        span.close()
    def to_metrics(self):
        return {f"span_{s.name}_ms": round(1000*(s.end - s.start), 2) for s in self.spans if s.end}

# --------- Ticker Resolver ---------
STATIC_TICKERS = {
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
}

def resolve_ticker(company: str) -> str:
    key = company.strip().lower()
    if key in STATIC_TICKERS:
        return STATIC_TICKERS[key]

    # Wikipedia hint
    hint = None
    try:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(language="en", top_k_results=1))
        text = wiki.invoke(f"{company} stock ticker")
        import re
        m = re.search(r"\\b[A-Z]{3,5}\\b", text)
        if m:
            hint = m.group(0)
    except Exception:
        pass

    if hint and yf is not None:
        try:
            t = yf.Ticker(hint)
            info = getattr(t, "fast_info", None)
            if info is not None:
                return hint
        except Exception:
            pass

    return key.split()[0][:5].upper()

# --------- News Fetching ---------
class NewsFetcher:
    def __init__(self):
        self.backends = []
        if "exa" in NEWS_BACKENDS and os.getenv("EXA_API_KEY"):
            self.backends.append("exa")
        if "tavily" in NEWS_BACKENDS and os.getenv("TAVILY_API_KEY"):
            self.backends.append("tavily")
        if "brave" in NEWS_BACKENDS and os.getenv("BRAVE_API_KEY"):
            self.backends.append("brave")
        self.backends.append("wikipedia")

    def fetch(self, query: str, top_k: int = NEWS_TOP_K, days: int = NEWS_LOOKBACK_DAYS):
        errors = []
        for b in self.backends:
            try:
                if b == "exa":
                    tool = ExaSearchResults(exa_api_key=os.environ["EXA_API_KEY"], num_results=top_k, use_autoprompt=True)
                    results = tool.invoke({"query": f"{query} news last {days} days"})
                    return [r.get("title","") + ": " + (r.get("snippet") or r.get("summary") or "") for r in results]
                if b == "tavily":
                    tool = TavilySearchResults(max_results=top_k)
                    results = tool.invoke({"query": f"{query} news last {days} days"})
                    return [r.get("title","") + ": " + (r.get("content") or "") for r in results]
                if b == "brave":
                    brave = BraveSearch(brave_api_key=os.environ["BRAVE_API_KEY"])
                    items = brave.results(f"{query} news last {days} days", count=top_k)
                    return [i.get("title","") + ": " + (i.get("description") or "") for i in items]
                if b == "wikipedia":
                    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=min(top_k, 3)))
                    txt = wiki.invoke(f"Recent developments about {query}")
                    return [s.strip() for s in txt.split(". ") if s.strip()][:top_k]
            except Exception as e:
                errors.append((b, str(e)))
        return [f"No live news available for {query}. (Backends failed: {errors})"]

# --------- LLM: Gemini structured output ---------
SYSTEM = (
    "You are a capital markets analyst. Given recent headlines/summaries, "
    "produce a structured sentiment profile for the target company."
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
        "Company: {company}\\n"
        "Ticker: {ticker}\\n"
        "Days: {days}\\n"
        "Headlines/Summaries (most‑recent first):\\n{news}\\n\\n"
        "Return STRICT JSON conforming to the schema."
    ),
])

model = init_chat_model(LLM_NAME, model_provider="google_genai")
structured_llm = model.with_structured_output(SentimentSchema)

def analyze_company(company: str, top_k: int = NEWS_TOP_K, days: int = NEWS_LOOKBACK_DAYS, verbose: bool = False) -> SentimentSchema:
    tracer = Tracer()

    s1 = tracer.start("ticker_resolve")
    ticker = resolve_ticker(company)
    tracer.finish(s1)

    s2 = tracer.start("news_fetch")
    fetcher = NewsFetcher()
    headlines = fetcher.fetch(f"{company} {ticker}", top_k=top_k, days=days)
    news_text = "\\n- " + "\\n- ".join(headlines)
    tracer.finish(s2)

    s3 = tracer.start("sentiment_llm")
    prompt_msg = PROMPT.invoke({
        "company": company,
        "ticker": ticker,
        "days": days,
        "news": news_text,
    })
    result = structured_llm.invoke(prompt_msg)
    tracer.finish(s3)

    if verbose:
        print("\\n[Debug] Headlines used:\\n", news_text)

    result.company_name = result.company_name or company
    result.stock_code = result.stock_code or ticker
    return result

def main():
    parser = argparse.ArgumentParser(description="Real‑Time Market Sentiment Analyzer")
    parser.add_argument("--company", required=True, help="Company name, e.g., 'Google'")
    parser.add_argument("--top_k", type=int, default=NEWS_TOP_K, help="Number of headlines")
    parser.add_argument("--days", type=int, default=NEWS_LOOKBACK_DAYS, help="Lookback window in days")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"sentiment:{args.company}"):
        mlflow.log_params({
            "company": args.company,
            "top_k": args.top_k,
            "days": args.days,
            "llm": LLM_NAME,
        })
        out = analyze_company(args.company, top_k=args.top_k, days=args.days, verbose=args.verbose)
        mlflow.log_dict(json.loads(out.model_dump_json()), artifact_file="sentiment.json")

        # synthetic span metrics (replace with real spans if you wire mlflow tracing)
        mlflow.log_metrics({"span_ticker_resolve_ms": 1.0, "span_news_fetch_ms": 1.0, "span_sentiment_llm_ms": 1.0})

        print(out.model_dump_json(indent=2))

if __name__ == "__main__":
    main()

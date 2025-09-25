#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal Real-Time Market Sentiment Analyzer
Uses Exa for news + Google Gemini (via LangChain) for structured sentiment.
"""

import argparse, os, json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.tools.exa_search import ExaSearchResults
from langchain_exa import ExaSearchResults


load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in .env")
if not EXA_API_KEY:
    raise RuntimeError("Set EXA_API_KEY in .env")


# --- Schema ---
class SentimentSchema(BaseModel):
    company_name: str
    stock_code: str
    newsdesc: str = Field(description="Concise rollup of latest news")
    sentiment: str = Field(description="Positive/Negative/Neutral")
    people_names: List[str] = []
    places_names: List[str] = []
    other_companies_referred: List[str] = []
    related_industries: List[str] = []
    market_implications: str = ""
    confidence_score: float = 0.0

# --- Prompt ---
SYSTEM = "You are a capital markets analyst. Produce a structured sentiment profile."
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Company: {company}\nTicker: {ticker}\nNews:\n{news}\nReturn STRICT JSON.")
])

# --- Init model ---
model = init_chat_model(
    "gemini-2.0-flash",
    model_provider="google_genai",
    api_key=GOOGLE_API_KEY,
)
structured_llm = model.with_structured_output(SentimentSchema)

# --- Exa fetch ---
def fetch_news(query: str, top_k: int, days: int):
    tool = ExaSearchResults(exa_api_key=EXA_API_KEY, num_results=top_k, use_autoprompt=True)
    resp = tool.invoke({"query": f"{query} news last {days} days"})
    return [f"{r.title}: {r.text}" for r in resp.results]


# --- Analyze ---
def analyze(company: str, ticker: str, top_k: int, days: int):
    news = "\n- " + "\n- ".join(fetch_news(company, top_k, days))
    prompt_msg = PROMPT.invoke({"company": company, "ticker": ticker, "news": news})
    return structured_llm.invoke(prompt_msg)

# --- CLI ---
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--company", required=True)
    p.add_argument("--ticker", default="")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--days", type=int, default=7)
    args = p.parse_args()

    out = analyze(args.company, args.ticker or args.company[:5].upper(), args.top_k, args.days)
    print(out.model_dump_json(indent=2))

"""
agentic_rag_simplified.py
----------------------------------------
LangGraph workflow implementing an Agentic RAG pipeline:
Retriever → Generator → Critique → Refinement
"""

import os
from typing import TypedDict, List
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ---- Initialize Components ----
emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("agentic-rag")


# ---- State Definition ----
class RAGState(TypedDict):
    question: str
    snippets: List[str]
    answer: str
    critique: str


# ---- Retriever Node ----
def retrieve_kb(state: RAGState):
    q_vec = emb.embed_query(state["question"])
    res = index.query(vector=q_vec, top_k=5, include_metadata=True)
    snippets = [m["metadata"]["snippet"] for m in res["matches"]]
    return {"snippets": snippets}


# ---- Generator Node ----
def generate_answer(state: RAGState):
    context = "\n".join([f"[KB{i+1}] {s}" for i, s in enumerate(state["snippets"])])
    prompt = ChatPromptTemplate.from_template(
        "Answer this question using the context:\n{context}\n\nQuestion: {question}\nInclude citations like [KB1]."
    )
    msg = prompt.invoke({"context": context, "question": state["question"]})
    ans = llm.invoke(msg)
    return {"answer": ans.content}


# ---- Critique Node ----
def critique_answer(state: RAGState):
    critique_prompt = ChatPromptTemplate.from_template(
        "Evaluate completeness.\n\nQuestion: {question}\nAnswer: {answer}\nContext: {context}\n\n"
        "Return either 'COMPLETE' or 'REFINE: <missing keywords>'."
    )
    msg = critique_prompt.invoke({
        "question": state["question"],
        "answer": state["answer"],
        "context": "\n".join(state["snippets"]),
    })
    critique = llm.invoke(msg).content.strip()
    return {"critique": critique}


# ---- Refinement Node ----
def refine_answer(state: RAGState):
    if state["critique"].startswith("REFINE"):
        missing = state["critique"].split(":", 1)[1]
        q_vec = emb.embed_query(missing)
        res = index.query(vector=q_vec, top_k=1, include_metadata=True)
        extra = res["matches"][0]["metadata"]["snippet"]
        full_ctx = "\n".join(state["snippets"] + [extra])
        prompt = ChatPromptTemplate.from_template(
            "Refine the answer using the context:\n{context}\n\nQuestion: {question}\nProvide improved answer with citations."
        )
        msg = prompt.invoke({"context": full_ctx, "question": state["question"]})
        ans = llm.invoke(msg)
        return {"answer": ans.content}
    return {"answer": state["answer"]}


# ---- Build LangGraph ----
builder = StateGraph(RAGState)
builder.add_node("retrieve_kb", retrieve_kb)
builder.add_node("generate_answer", generate_answer)
builder.add_node("critique_answer", critique_answer)
builder.add_node("refine_answer", refine_answer)

builder.add_edge(START, "retrieve_kb")
builder.add_edge("retrieve_kb", "generate_answer")
builder.add_edge("generate_answer", "critique_answer")
builder.add_edge("critique_answer", "refine_answer")
builder.add_edge("refine_answer", END)
graph = builder.compile()


# ---- Run Quick Tests ----
if __name__ == "__main__":
    questions = [
        "What are best practices for caching?",
        "How should I set up CI/CD pipelines?",
        "What are performance tuning tips?",
        "How do I version my APIs?",
        "What should I consider for error handling?",
    ]
    for q in questions:
        result = graph.invoke({"question": q})
        print("\n🧩 Question:", q)
        print("🪶 Critique:", result["critique"])
        print("✅ Final Answer:", result["answer"])

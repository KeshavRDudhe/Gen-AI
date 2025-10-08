"""
index_kb.py
----------------------------------------
Loads the KB dataset, creates Gemini embeddings,
and uploads vectors to Pinecone.
"""

import json, os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---- Load KB JSON ----
with open("self_critique_loop_dataset.json") as f:
    kb = json.load(f)

# ---- Initialize Embedding Model ----
emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ---- Initialize Pinecone ----
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "agentic-rag"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# ---- Embed & Upsert ----
vectors = []
for item in kb:
    vec = emb.embed_query(item["answer_snippet"])
    vectors.append({
        "id": item["doc_id"],
        "values": vec,
        "metadata": {
            "question": item["question"],
            "snippet": item["answer_snippet"]
        }
    })

index.upsert(vectors)
print(f"✅ Indexed {len(vectors)} KB entries into Pinecone.")

# assignment2_course_recommender.py
"""
Simple Personalized Course Recommendation Engine using Gemini Embeddings
Author: Keshav Dudhe
"""

# -----------------------------
# Install dependencies
# -----------------------------
# !pip install pandas chromadb langchain-google-genai --quiet

import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# Step 1: Load dataset
# -----------------------------
url = "https://raw.githubusercontent.com/Bluedata-Consulting/GAAPB01-training-code-base/refs/heads/main/Assignments/assignment2dataset.csv"
df = pd.read_csv(url)

print("✅ Dataset loaded successfully.")
print(f"Total courses: {len(df)}")
print(df.head(3)[["course_id", "title", "description"]])

# -----------------------------
# Step 2: Initialize Gemini Embeddings
# -----------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Compute embeddings for all course descriptions
print("🔍 Generating embeddings for all courses (this may take a few seconds)...")
course_embeddings = embeddings.embed_documents(df["Description"].tolist())

print("✅ Embeddings generated.")

# Convert to numpy for similarity computation
course_embeddings = np.array(course_embeddings)

# -----------------------------
# Step 3: Define Recommendation Function
# -----------------------------
def recommend_courses(user_query: str, top_k: int = 5):
    """
    Given a user query, returns top-k recommended courses.
    """
    query_embedding = np.array(embeddings.embed_query(user_query)).reshape(1, -1)
    sims = cosine_similarity(query_embedding, course_embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    recs = df.iloc[top_indices][["course_id", "title", "description"]]
    recs["Similarity"] = sims[top_indices]
    return recs

# -----------------------------
# Step 4: Run Sample Queries
# -----------------------------
sample_queries = [
    "I’ve completed the ‘Python Programming for Data Science’ course and enjoy data visualization.",
    "I know Azure basics and want to manage containers and build CI/CD pipelines.",
    "My background is in ML fundamentals; I’d like to specialize in neural networks and production workflows.",
    "I want to learn to build and deploy microservices with Kubernetes.",
    "I’m interested in blockchain and smart contracts but have no prior experience."
]

for i, q in enumerate(sample_queries, 1):
    print(f"\n-----------------------------")
    print(f"🔹 Query {i}: {q}")
    print(f"-----------------------------")
    results = recommend_courses(q)
    for idx, row in results.iterrows():
        print(f"{row['title']} (Score: {row['Similarity']:.3f})")

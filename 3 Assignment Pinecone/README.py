# 🧠 Simplified Agentic RAG System (Gemini + LangGraph)

A lightweight **Agentic RAG** pipeline that:
- Retrieves top-5 KB snippets (Pinecone)
- Generates Gemini-based answers with citations
- Performs self-critique for completeness
- Optionally refines the answer if missing info is detected

---

## 📦 Setup

```bash
git clone https://github.com/<your-username>/agentic_rag_assignment.git
cd agentic_rag_assignment
pip install -r requirements.txt

# DocuQuery

**Intelligent document Q&A — powered by RAG**

Upload your PDFs, get instant answers with source citations.

---

### Getting started

1. Click **📎** to upload one or more documents
2. Receive an auto-generated summary and suggested questions
3. Ask anything — DocuQuery searches across all your documents simultaneously
4. Click **📄 Show Sources** to see exactly which page each answer came from
5. Rate answers with **👍 / 👎** to help improve quality

---

### Search modes

| Mode | Best for |
|------|----------|
| 🤖 **Auto** | Default — system picks the best mode |
| 🔤 **Zone 1** (α=0.15) | Names, codes, IDs, exact terms |
| ⚖️ **Zone 2** (α=0.50) | Mixed factual + conceptual queries |
| 🧠 **Zone 3** (α=0.85) | Concepts, explanations, comparisons |

---

### Commands

- `/alpha 0.7` — manually set search weight
- `/clear_history` — reset conversation memory
- `/compare doc_id_a doc_id_b` — compare two documents
- `/feedback` — view quality summary

---

*DocuQuery v4.1 · Powered by Groq LLaMA 3.3 70B · Pinecone Hybrid Search*
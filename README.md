# 📚 Wikipedia Research Assistant

A research assistant that retrieves Wikipedia articles via a FAISS vector
index ([wiki-rag](https://github.com/RoyRin/wiki-rag)) and uses a hosted LLM
(Anthropic or OpenAI) to synthesise structured reports.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your API key(s)

```bash
cp .env.example .env
# then edit .env and fill in ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

`.env` is git-ignored — your keys are never committed or pushed.

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

> **First run:** the app automatically downloads the Wikipedia FAISS index
> (~500 MB) from HuggingFace. This happens once and is cached in `wiki_rag_data/`.
> You can also trigger it manually with the "Pre-load Wikipedia index" button
> in the sidebar.

---

## Project structure

```
wiki-research-assistant/
├── app.py            # Streamlit UI
├── rag_engine.py      # All RAG + LLM logic
├── requirements.txt
├── .env.example       # Template for your API keys (copy to .env)
└── README.md
```

---

## Choosing a provider

Pick Anthropic or OpenAI from the sidebar radio button. Each needs its own
API key, entered in the sidebar or read automatically from `.env`:

| Provider  | Env var             | Default model    |
|-----------|---------------------|-------------------|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| OpenAI    | `OPENAI_API_KEY`    | `gpt-4o-mini`        |

The Anthropic model is picked from a dropdown (`ANTHROPIC_MODELS` in
`rag_engine.py`); the OpenAI model is a free-text field so you can type any
current model name.

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository (`.env` stays local, never pushed).
2. Go to https://share.streamlit.io and connect your repo.
3. Set **Main file path** to `app.py`.
4. In App settings → Secrets, add `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`.

---

## How it works

```
User query
    │
    ▼
wiki-rag FAISS index
(top-100k Wikipedia articles, embedded with BAAI/bge-base-en)
    │
    ▼  similarity_search(query, k=6)
Retrieved article titles + excerpts
    │
    ▼
LLM prompt (Anthropic or OpenAI)  —  sources + research question
    │
    ▼
Structured JSON report
    │
    ▼
Streamlit renders:
  summary · analysis sections · key facts · cited sources
```

# 📚 Wikipedia Research Assistant

A local AI research assistant that retrieves Wikipedia articles via a FAISS vector
index ([wiki-rag](https://github.com/RoyRin/wiki-rag)) and uses Ollama to synthesise
structured reports — fully offline after first setup.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

Download from https://ollama.com, then:

```bash
ollama serve          # start the server
ollama pull llama3.2  # download a model (or mistral, phi3, etc.)
```

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
├── app.py           # Streamlit UI
├── rag_engine.py    # All RAG + Ollama logic
├── requirements.txt
└── README.md
```

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io and connect your repo.
3. Set **Main file path** to `app.py`.

> ⚠️ Streamlit Cloud does not run Ollama. For cloud deployment, swap
> `call_ollama()` in `rag_engine.py` for the Anthropic or OpenAI API.
> See the "Cloud LLM swap" section below.

---

## Cloud LLM swap (for Streamlit Cloud)

Replace the body of `call_ollama()` in `rag_engine.py` with:

```python
import anthropic

def call_ollama(prompt, system="", base_url=None, model=None):
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system,
        messages=[{"role": "role": "user", "content": prompt}],
    )
    return msg.content[0].text
```

Then add `ANTHROPIC_API_KEY` in Streamlit Cloud → App settings → Secrets.

---

## Changing the Ollama model

In `rag_engine.py`:

```python
DEFAULT_OLLAMA_MODEL = "mistral"   # or phi3, gemma3, deepseek-r1, etc.
```

Or just pick it from the sidebar dropdown while the app is running.

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
Ollama prompt  (sources + research question)
    │
    ▼
Structured JSON report
    │
    ▼
Streamlit renders:
  summary · analysis sections · key facts · cited sources
```

"""
rag_engine.py
─────────────
Handles all wiki-rag and Ollama interactions.
Keeps app.py clean — all AI logic lives here.
"""

import json
import requests
from pathlib import Path
from dataclasses import dataclass


# ── Config ─────────────────────────────────────────────────────────────────────

RAG_NAME = "wiki_index__top_100000__2025-04-11"
RAG_DIR  = Path("wiki_rag_data")

DEFAULT_OLLAMA_URL   = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"

ANTHROPIC_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
    "claude-fable-5",
]
DEFAULT_ANTHROPIC_MODEL = ANTHROPIC_MODELS[0]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class WikiSource:
    title: str
    excerpt: str
    url: str


@dataclass
class ReportSection:
    title: str
    content: str


@dataclass
class ResearchReport:
    query: str
    summary: str
    sections: list[ReportSection]
    key_facts: list[str]
    further_reading: list[str]
    sources: list[WikiSource]


# ── RAG loader (cached so it only downloads once) ──────────────────────────────

_vectorstore = None


def load_vectorstore(status_callback=None):
    """
    Download (first run) and return the wiki-rag FAISS vectorstore.
    Pass an optional callback(msg: str) to stream progress to the UI.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    try:
        from wiki_rag import rag
    except ImportError:
        raise RuntimeError(
            "wiki_rag is not installed.\n"
            "Run:  pip install git+https://github.com/RoyRin/wiki-rag.git"
        )

    if status_callback:
        status_callback("📥 Downloading Wikipedia index from HuggingFace (first run — ~500 MB, takes a few minutes)…")

    embedding = rag.PromptedBGE(model_name="BAAI/bge-base-en")

    _vectorstore = rag.download_and_build_rag_from_huggingface(
        embeddings=embedding,
        rag_name=RAG_NAME,
        save_dir=RAG_DIR,
    )

    if status_callback:
        status_callback("✅ Wikipedia index loaded.")

    return _vectorstore


# ── Wikipedia helpers ──────────────────────────────────────────────────────────

def title_to_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + title.strip().replace(" ", "_")


def retrieve_sources(query: str, k: int = 6) -> list[WikiSource]:
    """Run similarity search and return WikiSource objects."""
    vs = load_vectorstore()
    results = vs.similarity_search(query, k=k)
    if not results:
        raise ValueError(f"No Wikipedia articles found for: '{query}'")

    return [
        WikiSource(
            title=r.metadata.get("title", "Unknown"),
            excerpt=r.page_content[:700].strip(),
            url=title_to_url(r.metadata.get("title", "")),
        )
        for r in results
    ]


# ── Ollama helpers ─────────────────────────────────────────────────────────────

def check_ollama(base_url: str = DEFAULT_OLLAMA_URL) -> dict:
    """Return {"ok": bool, "models": [...]}"""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return {"ok": True, "models": models}
    except Exception:
        return {"ok": False, "models": []}


def call_ollama(
    prompt: str,
    system: str = "",
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
) -> str:
    """Send a prompt to Ollama and return the text response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 2048},
    }
    try:
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=180)
        r.raise_for_status()
        return r.json().get("response", "")
    except requests.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}.\n"
            "Make sure Ollama is running:  ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def call_anthropic(
    prompt: str,
    system: str = "",
    api_key: str = "",
    model: str = DEFAULT_ANTHROPIC_MODEL,
) -> str:
    """Send a prompt to the Anthropic API and return the text response."""
    if not api_key:
        raise RuntimeError("An Anthropic API key is required.")

    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic is not installed.\n"
            "Run:  pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.3,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
    except anthropic.AuthenticationError:
        raise RuntimeError("Anthropic rejected the API key. Check that it is valid.")
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic request failed: {e}")


# ── Report generation ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a meticulous research librarian. "
    "You write structured, factual reports grounded only in the provided Wikipedia sources. "
    "Always cite sources by their number in square brackets like [1] or [2][3]."
)

REPORT_TEMPLATE = """\
You are a research librarian. Using ONLY the Wikipedia sources below, write a \
structured research report about: "{query}"

SOURCES:
{context}

Return your response as valid JSON with this EXACT structure (no extra keys):
{{
  "summary": "2-3 sentence executive summary of the topic",
  "sections": [
    {{"title": "Section heading", "content": "3-5 sentences of analysis citing sources like [1] [2]"}},
    {{"title": "Another section", "content": "..."}}
  ],
  "key_facts": [
    "One concrete fact from the sources [N]",
    "Another fact [N]"
  ],
  "further_reading": ["Related topic 1", "Related topic 2", "Related topic 3"]
}}

Rules:
- Write 3-5 sections covering different aspects of the topic.
- Every factual claim must cite at least one source number.
- key_facts should be 4-6 specific, interesting facts.
- further_reading should be Wikipedia-style topic suggestions.
- Return ONLY the JSON object — no markdown fences, no preamble.
"""


def build_prompt(query: str, sources: list[WikiSource]) -> str:
    context = "\n\n".join(
        f"[{i+1}] {s.title}\n{s.excerpt}"
        for i, s in enumerate(sources)
    )
    return REPORT_TEMPLATE.format(query=query, context=context)


def parse_report_json(raw: str, query: str, sources: list[WikiSource]) -> ResearchReport:
    """Parse Ollama JSON output into a ResearchReport, with a safe fallback."""
    cleaned = (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        data = json.loads(cleaned)
        sections = [
            ReportSection(title=s["title"], content=s["content"])
            for s in data.get("sections", [])
        ]
        return ResearchReport(
            query=query,
            summary=data.get("summary", ""),
            sections=sections,
            key_facts=data.get("key_facts", []),
            further_reading=data.get("further_reading", []),
            sources=sources,
        )
    except (json.JSONDecodeError, KeyError):
        # Graceful degradation: wrap raw text in a single section
        return ResearchReport(
            query=query,
            summary=f"Research findings for: {query}",
            sections=[ReportSection(title="Findings", content=raw)],
            key_facts=[],
            further_reading=[],
            sources=sources,
        )


def generate_report(
    query: str,
    num_sources: int = 6,
    provider: str = "ollama",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    anthropic_api_key: str = "",
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
) -> ResearchReport:
    """Full pipeline: retrieve → prompt → generate → parse.

    `provider` selects which LLM backend writes the report: "ollama" (local,
    default) or "anthropic" (hosted API, requires anthropic_api_key).
    """
    sources = retrieve_sources(query, k=num_sources)
    prompt  = build_prompt(query, sources)

    if provider == "anthropic":
        raw = call_anthropic(
            prompt, system=SYSTEM_PROMPT, api_key=anthropic_api_key, model=anthropic_model
        )
    else:
        raw = call_ollama(
            prompt, system=SYSTEM_PROMPT, base_url=ollama_url, model=ollama_model
        )

    return parse_report_json(raw, query, sources)

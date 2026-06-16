"""
app.py  —  Wikipedia Research Assistant
────────────────────────────────────────
Run locally:   streamlit run app.py
Deploy:        push to GitHub → connect at share.streamlit.io

Requirements:
  • An API key for Anthropic and/or OpenAI (see .env.example)
  • wiki-rag index  →  auto-downloaded on first use (~500 MB)
"""

import os

from dotenv import load_dotenv

load_dotenv()  # pulls ANTHROPIC_API_KEY / OPENAI_API_KEY from a local .env file, if present.
               # .env is git-ignored — see .env.example for the expected format.

import streamlit as st
from rag_engine import (
    generate_report,
    load_vectorstore,
    ANTHROPIC_MODELS,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    ResearchReport,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Wikipedia Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.source-card {
    background: #f8f9fa;
    border-left: 3px solid #4a90d9;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
}
.source-card a { color: #1a73e8; text-decoration: none; font-weight: 600; }
.source-card a:hover { text-decoration: underline; }
.source-excerpt { color: #555; margin-top: 0.3rem; line-height: 1.5; }

.section-box {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}
.section-title { font-weight: 600; font-size: 1.05rem; color: #1a1a2e; margin-bottom: 0.5rem; }
.section-content { color: #333; line-height: 1.7; font-size: 0.95rem; }

.fact-item {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    margin-bottom: 0.5rem;
    font-size: 0.92rem;
    color: #333;
    line-height: 1.5;
}
.fact-dot {
    min-width: 8px; height: 8px;
    background: #4a90d9;
    border-radius: 50%;
    margin-top: 6px;
}

.summary-box {
    background: linear-gradient(135deg, #eef2ff 0%, #f0f9ff 100%);
    border-left: 4px solid #4a90d9;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    font-size: 1rem;
    color: #1a1a2e;
    line-height: 1.7;
    margin-bottom: 1.5rem;
}

.chip {
    display: inline-block;
    background: #e8f0fe;
    color: #1a73e8;
    border-radius: 16px;
    padding: 4px 12px;
    font-size: 0.82rem;
    margin: 3px;
    font-weight: 500;
}

.status-ok  { color: #2e7d32; font-weight: 600; }
.status-err { color: #c62828; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    provider = st.radio(
        "LLM provider",
        options=["Anthropic (API)", "OpenAI (API)"],
        help="Both are hosted APIs and require an API key (see .env.example).",
    )

    anthropic_api_key = ""
    anthropic_model = DEFAULT_ANTHROPIC_MODEL
    openai_api_key = ""
    openai_model = DEFAULT_OPENAI_MODEL

    if provider == "Anthropic (API)":
        anthropic_api_key = st.text_input(
            "Anthropic API key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Defaults to the ANTHROPIC_API_KEY environment variable if set.",
        )
        if anthropic_api_key:
            st.markdown('<span class="status-ok">● API key set</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">● No API key provided</span>', unsafe_allow_html=True)

        anthropic_model = st.selectbox(
            "Model",
            options=ANTHROPIC_MODELS,
            index=0,
            help="Choose which Claude model to use.",
        )
    else:
        openai_api_key = st.text_input(
            "OpenAI API key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Defaults to the OPENAI_API_KEY environment variable if set.",
        )
        if openai_api_key:
            st.markdown('<span class="status-ok">● API key set</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">● No API key provided</span>', unsafe_allow_html=True)

        openai_model = st.text_input(
            "Model",
            value=DEFAULT_OPENAI_MODEL,
            help="Any OpenAI chat model name, e.g. gpt-4o-mini, gpt-4o, gpt-4.1.",
        )

    st.markdown("---")

    num_sources = st.slider(
        "Sources to retrieve",
        min_value=3,
        max_value=10,
        value=6,
        help="More sources = richer report, but slower generation.",
    )

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "Retrieves Wikipedia articles using a local FAISS vector index, "
        "then asks an LLM (Anthropic or OpenAI) to synthesise a structured report."
    )
    st.caption("📦 [wiki-rag on GitHub](https://github.com/RoyRin/wiki-rag)")
    st.caption("🤖 [Anthropic](https://www.anthropic.com)  ·  🧠 [OpenAI](https://platform.openai.com)")

    st.markdown("---")
    if st.button("⬇️ Pre-load Wikipedia index", use_container_width=True):
        with st.spinner("Downloading index (first run only, ~500 MB)…"):
            try:
                load_vectorstore()
                st.success("Index ready!")
            except Exception as e:
                st.error(str(e))


# ── Main ───────────────────────────────────────────────────────────────────────

st.title("📚 Wikipedia Research Assistant")
st.caption("Ask a research question — get a structured report grounded in Wikipedia.")

st.markdown("**Try an example:**")
examples = [
    "History of the Roman Empire",
    "How CRISPR gene editing works",
    "Causes of the French Revolution",
    "Quantum computing applications",
    "Marie Curie's discoveries",
]
cols = st.columns(len(examples))
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True, key=example):
        st.session_state["query_input"] = example

st.markdown("")

query = st.text_input(
    "Research question",
    placeholder="e.g. The development of the internet",
    key="query_input",
    label_visibility="collapsed",
)

run = st.button("🔍 Research", type="primary")

# ── Run ────────────────────────────────────────────────────────────────────────

if run and query.strip():
    if provider == "Anthropic (API)":
        if not anthropic_api_key:
            st.error("Please enter an Anthropic API key in the sidebar.")
            st.stop()
    else:
        if not openai_api_key:
            st.error("Please enter an OpenAI API key in the sidebar.")
            st.stop()

    with st.spinner("🔎 Searching Wikipedia + generating report…"):
        try:
            report: ResearchReport = generate_report(
                query=query.strip(),
                num_sources=num_sources,
                provider="openai" if provider == "OpenAI (API)" else "anthropic",
                anthropic_api_key=anthropic_api_key,
                anthropic_model=anthropic_model,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
            )
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        except ValueError as e:
            st.warning(str(e))
            st.stop()

    # ── Render ─────────────────────────────────────────────────────────────────

    st.markdown("---")
    st.subheader(f"📄 {report.query}")

    st.markdown(
        f'<div class="summary-box">{report.summary}</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 1])

    with left:
        st.markdown("### Analysis")
        for section in report.sections:
            st.markdown(
                f'<div class="section-box">'
                f'<div class="section-title">{section.title}</div>'
                f'<div class="section-content">{section.content}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if report.further_reading:
            st.markdown("### Further reading")
            chips = " ".join(
                f'<span class="chip">{t}</span>' for t in report.further_reading
            )
            st.markdown(chips, unsafe_allow_html=True)

    with right:
        if report.key_facts:
            st.markdown("### Key facts")
            for fact in report.key_facts:
                st.markdown(
                    f'<div class="fact-item">'
                    f'<div class="fact-dot"></div><div>{fact}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### Sources")
        for i, source in enumerate(report.sources):
            st.markdown(
                f'<div class="source-card">'
                f'<a href="{source.url}" target="_blank">[{i+1}] {source.title}</a>'
                f'<div class="source-excerpt">{source.excerpt[:200]}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

elif run and not query.strip():
    st.warning("Please enter a research question.")

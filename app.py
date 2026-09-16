"""ExpoPrep: Career Fair & Networking Co-Pilot.

Provider-agnostic Streamlit agent using LiteLLM (Gemini / xAI Grok).
Run with: streamlit run app.py
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import streamlit as st
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from litellm import completion
from pypdf import PdfReader

load_dotenv()

# ---------------------------------------------------------------------------
# Models / constants
# ---------------------------------------------------------------------------

MODELS = {
    "Gemini 2.5 Flash": "gemini/gemini-2.5-flash",
    "Grok Beta / Grok 2": "xai/grok-2-latest",
}
# Gemini 2.5 Flash is the default id; some keys now 404 it and need a successor.
GEMINI_FALLBACKS = [
    "gemini/gemini-2.5-flash",
    "gemini/gemini-flash-latest",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-3.6-flash",
]
DEFAULT_MODEL_LABEL = "Gemini 2.5 Flash"
DEFAULT_ROLE = "Forward Deployed AI Engineer / Intern"
COMPANY_PRESETS = ["Galatiq", "Rice University AI Lab"]
MAX_REACT_TURNS = 5

SEARCH_WEB_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the public web for recent news, initiatives, tech stack, "
            "career-fair presence, or products related to a target company or role. "
            "Use this before writing icebreakers and high-signal questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Focused search query (company + topic).",
                }
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = """You are ExpoPrep, a career-fair and networking co-pilot.

You receive a candidate resume, a target company, and a target role.
Your job is to produce a concise, high-signal prep briefing the candidate can use
on the expo floor in under two minutes.

Always use the search_web tool (one or more queries) to gather:
- what the company actually builds / researches
- engineering focus and likely tech stack
- recent news, initiatives, hiring, or expo/career-fair presence

Then produce a FINAL answer in GitHub-flavored Markdown with EXACTLY these sections:

## 🏢 Company Snapshot
Core business, engineering focus, and tech stack (bullet points).

## 🎯 Elevator Pitch
2–4 sentences aligning the candidate's resume strengths with the company's domain.
Write it in first person so they can say it out loud.

## 🧊 3 Contextual Icebreakers
Three numbered icebreakers that reference recent news, products, or expo presence.
Each should be one or two spoken sentences, not generic small talk.

## 💡 3 High-Signal Questions
Three numbered technical or product questions for a recruiter or engineer.
They should demonstrate genuine homework, not "tell me about your culture."

Rules:
- Do not invent citations. If search is thin, say so briefly and still be useful.
- Keep the briefing tight: scannable on a phone between booths.
- After you have enough search context, stop calling tools and write the briefing.
"""


# ---------------------------------------------------------------------------
# Tools (pure Python)
# ---------------------------------------------------------------------------

def search_web(query: str) -> str:
    """Return a clean summary of DuckDuckGo text results for *query*."""
    query = (query or "").strip()
    if not query:
        return "No query provided."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
    except Exception as exc:  # network / rate-limit / parser failures
        return f"Web search failed for {query!r}: {exc}"

    if not results:
        return f"No web results found for {query!r}."

    blocks: list[str] = []
    for i, item in enumerate(results, start=1):
        title = (item.get("title") or "Untitled").strip()
        snippet = (item.get("body") or item.get("snippet") or "").strip()
        link = (item.get("href") or item.get("link") or "").strip()
        blocks.append(f"{i}. {title}\n   {snippet}\n   {link}")
    return "\n\n".join(blocks)


def extract_pdf_text(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF (Streamlit UploadedFile or file-like)."""
    if uploaded_file is None:
        return ""

    try:
        reader = PdfReader(uploaded_file)
        pages: list[str] = []
        for page in reader.pages:
            chunk = page.extract_text() or ""
            if chunk.strip():
                pages.append(chunk)
        text = "\n\n".join(pages).strip()
        return text or "(PDF contained no extractable text.)"
    except Exception as exc:
        return f"(Could not read PDF: {exc})"


# ---------------------------------------------------------------------------
# LiteLLM helpers
# ---------------------------------------------------------------------------

def _api_key_for_model(model_id: str) -> str | None:
    if model_id.startswith("gemini/"):
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if model_id.startswith("xai/"):
        return os.getenv("XAI_API_KEY")
    return None


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    """Normalize a LiteLLM assistant message into an OpenAI-style dict."""
    tool_calls = getattr(message, "tool_calls", None)
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(message, "content", None) or "",
    }
    if tool_calls:
        serialized = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            serialized.append(
                {
                    "id": getattr(tc, "id", ""),
                    "type": getattr(tc, "type", "function") or "function",
                    "function": {
                        "name": getattr(fn, "name", "") if fn else "",
                        "arguments": getattr(fn, "arguments", "{}") if fn else "{}",
                    },
                }
            )
        payload["tool_calls"] = serialized
    return payload


def run_react_loop(
    *,
    model_id: str,
    company: str,
    role: str,
    resume_text: str,
    log: Callable[[str], None],
) -> str:
    """Autonomous tool-using loop. Returns the model's final markdown briefing."""
    api_key = _api_key_for_model(model_id)
    if not api_key:
        raise RuntimeError(
            f"No API key found for model `{model_id}`. "
            "Set GEMINI_API_KEY or XAI_API_KEY in your .env file."
        )

    user_payload = (
        f"Target company: {company}\n"
        f"Target role: {role}\n\n"
        f"Candidate resume:\n{resume_text.strip() or '(No resume provided.)'}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]

    active_model = model_id

    def _complete(**kwargs: Any):
        nonlocal active_model
        if active_model.startswith("gemini/"):
            candidates = [active_model] + [m for m in GEMINI_FALLBACKS if m != active_model]
        else:
            candidates = [active_model]
        last_err: Exception | None = None
        for candidate in candidates:
            try:
                result = completion(model=candidate, api_key=api_key, **kwargs)
                if candidate != active_model:
                    log(f"Model `{active_model}` unavailable — switched to `{candidate}`.")
                    active_model = candidate
                return result
            except Exception as exc:
                last_err = exc
                err_text = str(exc).lower()
                retryable = (
                    "notfound" in err_text
                    or "404" in err_text
                    or "no longer available" in err_text
                )
                if retryable and candidate != candidates[-1]:
                    log(f"`{candidate}` rejected; trying next Gemini model id.")
                    continue
                raise
        raise last_err or RuntimeError("No model accepted the request.")

    for turn in range(1, MAX_REACT_TURNS + 1):
        log(f"**Turn {turn}/{MAX_REACT_TURNS}** — calling `{active_model}`…")
        response = _complete(
            messages=messages,
            tools=[SEARCH_WEB_TOOL],
            tool_choice="auto",
        )
        message = response.choices[0].message
        assistant = _assistant_message_to_dict(message)
        messages.append(assistant)

        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls:
            final = (assistant.get("content") or "").strip()
            if final:
                log("Model returned a final briefing (no further tool calls).")
                return final
            log("Empty completion with no tool calls — stopping.")
            return "_The model returned an empty response._"

        log(f"Model requested **{len(tool_calls)}** tool call(s).")
        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"query": raw_args}

            if name != "search_web":
                tool_output = f"Unknown tool `{name}`."
            else:
                query = str(args.get("query", "")).strip()
                log(f"🔍 `search_web` query: `{query}`")
                tool_output = search_web(query)
                log(f"📄 Tool output:\n```\n{tool_output}\n```")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": tool_output,
                }
            )

    log("Reached maximum ReAct turns — requesting a final briefing without tools.")
    wrap = _complete(
        messages=messages
        + [
            {
                "role": "user",
                "content": (
                    "You have reached the tool-call limit. "
                    "Write the final ExpoPrep briefing now using the evidence you already have. "
                    "Do not call any more tools."
                ),
            }
        ],
    )
    return (wrap.choices[0].message.content or "").strip() or (
        "_Stopped after 5 turns without a briefing._"
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ExpoPrep Co-Pilot",
    layout="wide",
    page_icon="🎯",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; max-width: 1200px; }
      .expoprep-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0ea5e9 140%);
        border-radius: 16px;
        padding: 1.35rem 1.6rem;
        color: #f8fafc;
        margin-bottom: 1.1rem;
      }
      .expoprep-hero h1 { font-size: 1.7rem; margin: 0 0 0.35rem 0; }
      .expoprep-hero p { margin: 0; opacity: 0.88; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="expoprep-hero">
      <h1>🎯 ExpoPrep: Career Fair &amp; Networking Co-Pilot</h1>
      <p>Research the booth, stitch your resume to the company, walk in with icebreakers and questions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----- Sidebar -----
with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox(
        "Model",
        options=list(MODELS.keys()),
        index=list(MODELS.keys()).index(DEFAULT_MODEL_LABEL),
        help="LiteLLM routes this to Gemini or xAI without changing agent logic.",
    )
    model_id = MODELS[model_label]
    st.caption(f"`{model_id}`")

    st.subheader("API keys")
    gemini_ok = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    xai_ok = bool(os.getenv("XAI_API_KEY"))
    gemini_col, xai_col = st.columns(2)
    with gemini_col:
        if gemini_ok:
            st.badge("GEMINI_API_KEY", color="green")
        else:
            st.badge("GEMINI_API_KEY missing", color="red")
    with xai_col:
        if xai_ok:
            st.badge("XAI_API_KEY", color="green")
        else:
            st.badge("XAI_API_KEY missing", color="red")

    st.divider()
    st.subheader("Resume")
    uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    resume_from_pdf = ""
    if uploaded is not None:
        resume_from_pdf = extract_pdf_text(uploaded)
        st.success(f"Extracted {len(resume_from_pdf):,} characters from PDF.")

    with st.expander("Or paste resume text", expanded=not bool(resume_from_pdf)):
        pasted = st.text_area(
            "Resume text",
            value="",
            height=220,
            placeholder="Paste your resume here if you don't have a PDF…",
            label_visibility="collapsed",
        )

    resume_text = resume_from_pdf.strip() or pasted.strip()

    role = st.text_input("Target role", value=DEFAULT_ROLE)

# ----- Main panel -----
def _apply_company_preset(name: str) -> None:
    st.session_state.company_input = name


if "company_input" not in st.session_state:
    st.session_state.company_input = ""

st.text_input(
    "Target company",
    placeholder="Who are you walking up to?",
    key="company_input",
)
st.caption("Quick presets")
preset_cols = st.columns(len(COMPANY_PRESETS))
for col, name in zip(preset_cols, COMPANY_PRESETS):
    col.button(
        name,
        use_container_width=True,
        on_click=_apply_company_preset,
        args=(name,),
    )

company = (st.session_state.company_input or "").strip()

generate = st.button(
    "Generate Prep Briefing",
    type="primary",
    use_container_width=True,
    disabled=not company,
)

status_slot = st.empty()
with st.expander("Agent Reasoning & Tool Calls", expanded=False):
    trace_slot = st.empty()
output_slot = st.empty()
download_slot = st.empty()


def _render_briefing(markdown: str, *, download_key: str) -> None:
    output_slot.markdown(markdown)
    download_slot.download_button(
        label="Download prep_briefing.md",
        data=markdown,
        file_name="prep_briefing.md",
        mime="text/markdown",
        use_container_width=True,
        key=download_key,
    )


if generate:
    if not resume_text:
        st.warning("Add a resume PDF or paste resume text in the sidebar for a sharper pitch.")

    needed = _api_key_for_model(model_id)
    if not needed:
        st.error(
            f"Missing API key for `{model_id}`. "
            "Add GEMINI_API_KEY or XAI_API_KEY to your `.env` file and restart Streamlit."
        )
        st.stop()

    traces: list[str] = []

    def log_step(markdown: str) -> None:
        traces.append(markdown)
        trace_slot.markdown("\n\n".join(traces))
        status_slot.info(markdown.split("\n", 1)[0])

    try:
        with st.spinner("ExpoPrep is researching and drafting your briefing…"):
            briefing = run_react_loop(
                model_id=model_id,
                company=company,
                role=role,
                resume_text=resume_text,
                log=log_step,
            )
    except Exception as exc:
        status_slot.error(f"Agent failed: {exc}")
        trace_slot.exception(exc)
        st.stop()

    st.session_state["briefing"] = briefing
    st.session_state["trace"] = "\n\n".join(traces)
    status_slot.success("Prep briefing ready.")
    _render_briefing(briefing, download_key="download_fresh")
elif "briefing" in st.session_state:
    trace_slot.markdown(st.session_state.get("trace", "_No trace stored._"))
    _render_briefing(st.session_state["briefing"], download_key="download_cached")

# 🎯 ExpoPrep — Career Fair & Networking Agent

ExpoPrep is an autonomous preparation co-pilot for students walking up to company booths at career fairs. It pairs real-time web intelligence with resume parsing to produce high-signal pitches, contextual icebreakers, and proactive skill-gap defenses in seconds.

`Python 3.10+` · `Streamlit` · `LiteLLM` · `Gemini / Grok Ready`

## Key Features

- **Autonomous multi-step research** — ReAct loop with DuckDuckGo search and company-news tools.
- **Resume stitching** — Extracts structured background from candidate PDFs (`pypdf`) and aligns it to the target company and role.
- **Enterprise-grade security** —  Prompt hardening, output sanitization against key leakage, and a demo access gate.
- **Provider-agnostic core** — LiteLLM routes the same agent logic to Gemini Flash, xAI Grok, or other compatible endpoints.

## How to Use

1. Enter the **target company** (or use a preset).
2. Upload a **candidate resume** (PDF) or paste text in the sidebar.
3. Generate a briefing: talking points, icebreakers, latest-news hooks, and skill-gap defenses.

## Local Setup

```bash
git clone https://github.com/Flugschnitzel/expo-prep-demo.git
cd expo-prep-demo/
```

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` in the repo root:

```bash
GEMINI_API_KEY="your-gemini-key"
DEMO_ACCESS_CODE="your-demo-code"  # required to unlock the UI
# XAI_API_KEY="optional-xai-key"
```

```bash
streamlit run app.py
```

Unlock with `DEMO_ACCESS_CODE`, add keys in the sidebar or via `.env`, then generate a prep briefing. Optional: paste BYOK Gemini / xAI keys in the sidebar; they override environment values for that session.

# 🛡️ Azure Prompt Shield — Shielded Chat PoC

A Streamlit proof-of-concept that places **two Azure Content Safety guardrails** in front of every chat message before it reaches the LLM.

## How it works

```
User prompt
    │
    ▼
┌─────────────────────────────────────────┐
│  Check 1 — Azure Prompt Shield          │  Blocks prompt injection & jailbreaks
│  POST …/text:shieldPrompt               │  (attempts to hijack the AI system)
└─────────────────────────────────────────┘
    │ SAFE
    ▼
┌─────────────────────────────────────────┐
│  Check 2 — Azure Content Safety         │  Blocks harmful content
│  POST …/text:analyze                    │  (Hate / Violence / SelfHarm / Sexual)
└─────────────────────────────────────────┘
    │ SAFE
    ▼
  LLM (via IQ API / litellm proxy)
```

> **Key distinction:** Prompt Shield detects *attack patterns* (e.g. "Ignore all previous instructions…"). Content Analysis detects *harmful topics* (hate speech, violence, etc.). You need both.

## Quick start

```bash
# 1. Clone
git clone https://github.com/bn-mrajpurohit/prompt-security-poc.git
cd prompt-security-poc

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env and fill in your keys

# 5. Run
streamlit run app.py
```

## Configuration

All credentials are read from `.env` (or environment variables) first, with sidebar inputs as fallback.

| Variable | Description |
|---|---|
| `AZURE_CONTENT_SAFETY_ENDPOINT` | Your Azure Content Safety resource URL |
| `AZURE_CONTENT_SAFETY_KEY` | Azure subscription key |
| `LITELLM_API_BASE` | LLM proxy base URL (default: `https://api.iq.cudasvc.com`) |
| `LITELLM_API_KEY` | API key for the LLM proxy |

## Azure APIs used

| API | Endpoint | Purpose |
|---|---|---|
| Prompt Shield | `POST /contentsafety/text:shieldPrompt?api-version=2024-02-15-preview` | Injection & jailbreak detection |
| Analyze Text | `POST /contentsafety/text:analyze?api-version=2023-10-01` | Harmful content scoring (0/2/4/6 severity) |

## PM Demo feature

Expand **"🛠️ Under the Hood: Shield Analysis Data"** below the chat to see the raw JSON responses from both Azure APIs — useful for demonstrating exactly which signals fired.

## Stack

- [Streamlit](https://streamlit.io) — UI
- [Azure Content Safety](https://learn.microsoft.com/azure/ai-services/content-safety/) — Prompt Shield + text analysis
- [OpenAI Python SDK](https://github.com/openai/openai-python) — LLM calls via litellm-compatible proxy
- [python-dotenv](https://pypi.org/project/python-dotenv/) — `.env` loading

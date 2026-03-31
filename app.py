"""
Azure Prompt Shield — Shielded Chat Interface (PoC)
====================================================
requirements.txt:
    streamlit>=1.35.0
    requests>=2.31.0
    openai>=1.30.0
    python-dotenv>=1.0.0

Run:
    streamlit run app.py
"""

import os

import requests
from dotenv import load_dotenv

# Load .env before any os.environ.get() calls so env-var fallbacks work.
load_dotenv()

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LITELLM_API_BASE can be overridden via .env; hard-coded value is the default.
LITELLM_BASE_URL = os.environ.get("LITELLM_API_BASE", "https://api.iq.cudasvc.com")
# Model name exposed by the IQ API — adjust to whatever is deployed.
LITELLM_MODEL = "gpt-4o-mini"

# Placeholder shown when no real LLM key is available yet.
_NO_KEY_RESPONSE = (
    "⚠️  *LLM key not configured.* Your prompt was safe and would have been "
    "forwarded to the model. Set `LITELLM_API_KEY` (or enter it in the sidebar) "
    "to enable live responses."
)


# ---------------------------------------------------------------------------
# Azure Prompt Shield — core analysis function
# ---------------------------------------------------------------------------

def analyze_prompt(user_text: str, endpoint: str, key: str) -> dict | None:
    """
    Send *user_text* to the Azure Content Safety Prompt Shield REST API.

    The azure-ai-contentsafety SDK v1.0.0 does not yet include ShieldPrompt
    models, so we call the preview REST endpoint directly with `requests`.

    API reference:
      POST {endpoint}/contentsafety/text:shieldPrompt?api-version=2024-02-15-preview

    Returns a dict with keys:
        - "attackDetected"     (bool) — True if userPromptAnalysis flagged an attack
        - "userPromptAnalysis" (dict) — raw per-category block from the API
        - "raw"                (dict) — full API response body

    Returns None on configuration/network errors (st.error already called).
    """
    if not endpoint or not key:
        st.error(
            "⚠️ Azure Content Safety credentials are missing. "
            "Please enter them in the sidebar."
        )
        return None

    # Build the REST URL — strip trailing slash to avoid double-slash.
    url = (
        f"{endpoint.rstrip('/')}"
        "/contentsafety/text:shieldPrompt"
        "?api-version=2024-02-15-preview"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/json",
    }
    # 'documents' can hold RAG/grounding docs for doc-injection detection.
    # Left empty here since we only shield the live user prompt.
    payload = {"userPrompt": user_text, "documents": []}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)

        # Surface HTTP-level errors (401 Unauthorized, 404, etc.) clearly.
        if not resp.ok:
            st.error(
                f"🔴 Azure API error {resp.status_code}: {resp.text}\n\n"
                "Check that your endpoint URL and API key are correct."
            )
            return None

        raw: dict = resp.json()

        # userPromptAnalysis.attackDetected is the primary signal.
        upa: dict = raw.get("userPromptAnalysis", {})
        attack_detected: bool = bool(upa.get("attackDetected", False))

        return {
            "attackDetected": attack_detected,
            "userPromptAnalysis": upa,
            "raw": raw,
        }

    except requests.exceptions.Timeout:
        st.error("🔴 Request to Azure Prompt Shield timed out. Check your endpoint URL.")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"🔴 Unexpected error calling Azure Prompt Shield: {exc}")
        return None


# ---------------------------------------------------------------------------
# Azure Content Safety — harmful content analysis (hate, violence, etc.)
# ---------------------------------------------------------------------------

# Severity levels returned by the API: 0=Safe, 2=Low, 4=Medium, 6=High.
# We block anything at or above this threshold.
HARM_SEVERITY_THRESHOLD = 2


def analyze_text_content(user_text: str, endpoint: str, key: str) -> dict | None:
    """
    Call the Azure Content Safety *analyzeText* API to score the prompt across
    four harm categories: Hate, SelfHarm, Sexual, Violence.

    This is a SEPARATE check from Prompt Shield:
      - Prompt Shield  → detects prompt-injection / jailbreak ATTACKS
      - analyzeText    → detects HARMFUL CONTENT (hate speech, violence, etc.)

    Returns a dict with keys:
        - "harmDetected"        (bool) — True if any category >= threshold
        - "categoriesAnalysis"  (list) — raw per-category severity scores
        - "raw"                 (dict) — full API response body

    Returns None on error (st.error already called).
    """
    if not endpoint or not key:
        return None  # credentials error already surfaced by analyze_prompt

    url = (
        f"{endpoint.rstrip('/')}"
        "/contentsafety/text:analyze"
        "?api-version=2023-10-01"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": user_text,
        "categories": ["Hate", "SelfHarm", "Sexual", "Violence"],
        "outputType": "FourSeverityLevels",  # 0 / 2 / 4 / 6
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)

        if not resp.ok:
            st.warning(
                f"⚠️ Content analysis API error {resp.status_code}: {resp.text}"
            )
            return None

        raw: dict = resp.json()
        categories: list = raw.get("categoriesAnalysis", [])

        # Flag as harmful if any category score meets or exceeds threshold.
        harm_detected = any(
            c.get("severity", 0) >= HARM_SEVERITY_THRESHOLD for c in categories
        )

        return {
            "harmDetected": harm_detected,
            "categoriesAnalysis": categories,
            "raw": raw,
        }

    except requests.exceptions.Timeout:
        st.warning("⚠️ Content analysis API timed out.")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        st.warning(f"⚠️ Unexpected error in content analysis: {exc}")
        return None


# ---------------------------------------------------------------------------
# LLM call via IQ API
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict], llm_api_key: str) -> str:
    """
    Forward the (already-shielded) conversation to the IQ API.

    *messages* follows the OpenAI chat format:
        [{"role": "user"|"assistant"|"system", "content": "..."}]

    Returns the assistant's reply as a plain string.
    """
    if not llm_api_key:
        return _NO_KEY_RESPONSE

    try:
        client = OpenAI(
            api_key=llm_api_key,
            base_url=LITELLM_BASE_URL,
        )
        completion = client.chat.completions.create(
            model=LITELLM_MODEL,
            messages=messages,
        )
        return completion.choices[0].message.content or "(empty response)"
    except Exception as exc:  # pylint: disable=broad-except
        return f"⚠️ LLM call failed: {exc}"


# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Shielded Chat — Azure Prompt Shield PoC",
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar — configuration panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    st.subheader("🔷 Azure Content Safety")

    # Prefer environment variables; fall back to sidebar text inputs.
    default_endpoint = os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT", "")
    default_key      = os.environ.get("AZURE_CONTENT_SAFETY_KEY", "")

    azure_endpoint = st.text_input(
        "Endpoint URL",
        value=default_endpoint,
        placeholder="https://<your-resource>.cognitiveservices.azure.com",
        help="Your Azure Content Safety resource endpoint.",
    )
    azure_key = st.text_input(
        "API Key",
        value=default_key,
        type="password",
        help="Azure Content Safety subscription key.",
    )

    st.markdown("---")
    st.subheader("🤖 Model via IQ")

    default_llm_key = os.environ.get("LITELLM_API_KEY", "")
    llm_api_key = st.text_input(
        "IQ API Key",
        value=default_llm_key,
        type="password",
        help="API key for the IQ API at api.iq.cudasvc.com.",
    )
    llm_model = st.text_input(
        "Model name",
        value=LITELLM_MODEL,
        help="Model identifier exposed by the IQ API.",
    )

    st.markdown("---")
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.session_state.last_shield_data = None
        st.rerun()

    st.markdown("---")
    st.caption(
        "Azure Prompt Shield PoC · built with "
        "[azure-ai-contentsafety](https://pypi.org/project/azure-ai-contentsafety/) "
        "& Streamlit"
    )


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str, "blocked": bool}
    st.session_state.messages = []

if "last_shield_data" not in st.session_state:
    # Holds the raw Azure Prompt Shield response for the most recent prompt.
    st.session_state.last_shield_data = None

if "last_content_data" not in st.session_state:
    # Holds the raw analyzeText (harm categories) response for the most recent prompt.
    st.session_state.last_content_data = None


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🛡️ Shielded Chat Interface")
st.markdown(
    "Every message passes through **two Azure Content Safety checks** before "
    "reaching the LLM:\n"
    "1. **Prompt Shield** — blocks prompt-injection & jailbreak *attacks*\n"
    "2. **Content Analysis** — blocks harmful content (hate, violence, self-harm, sexual)"
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    role    = msg["role"]
    content = msg["content"]
    blocked = msg.get("blocked", False)

    with st.chat_message(role):
        if blocked:
            # Render blocked user messages in a red warning box.
            st.error(f"🚫 *[BLOCKED]* {content}")
        else:
            st.markdown(content)


# ---------------------------------------------------------------------------
# Chat input & interception flow
# ---------------------------------------------------------------------------

user_input: str | None = st.chat_input("Type your message…")

if user_input:
    # ---- 1. Show the user's message immediately in the UI -----------------
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- 2a. Check 1 — Azure Prompt Shield (injection / jailbreak) --------
    shield_result: dict | None = None
    with st.spinner("🔍 Check 1/2 — Scanning for prompt injection & jailbreaks…"):
        shield_result = analyze_prompt(user_input, azure_endpoint, azure_key)

    # ---- 2b. Check 2 — Azure Content Safety analyzeText (harmful content) -
    content_result: dict | None = None
    with st.spinner("🔍 Check 2/2 — Scanning for harmful content…"):
        content_result = analyze_text_content(user_input, azure_endpoint, azure_key)

    # Persist both results for the "Under the Hood" expander.
    st.session_state.last_shield_data  = shield_result
    st.session_state.last_content_data = content_result

    # ---- 3. Determine block reason (either check can independently block) --
    attack_blocked = shield_result is not None and shield_result["attackDetected"]
    harm_blocked   = content_result is not None and content_result["harmDetected"]

    # ---- 4a. BLOCKED by Prompt Shield (injection / jailbreak) -------------
    if attack_blocked:
        with st.chat_message("assistant"):
            st.error(
                "🚨 **Prompt Injection / Jailbreak Detected — Access Blocked.**\n\n"
                "Azure Prompt Shield identified this as an attempt to manipulate "
                "or bypass the AI system. The prompt has **not** been forwarded to the LLM."
            )
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "blocked": True}
        )

    # ---- 4b. BLOCKED by Content Safety (harmful content) ------------------
    elif harm_blocked:
        # Surface which categories triggered and at what severity.
        triggered = [
            f"{c['category']} (severity {c['severity']})"
            for c in (content_result.get("categoriesAnalysis") or [])
            if c.get("severity", 0) >= HARM_SEVERITY_THRESHOLD
        ]
        triggered_str = ", ".join(triggered) if triggered else "unknown category"

        with st.chat_message("assistant"):
            st.error(
                f"🚨 **Harmful Content Detected — Access Blocked.**\n\n"
                f"Azure Content Safety flagged: **{triggered_str}**.\n\n"
                "This prompt has **not** been forwarded to the LLM."
            )
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "blocked": True}
        )

    # ---- 4c. SAFE — both checks passed, forward to LLM --------------------
    elif shield_result is not None:
        # Append to history only after passing both checks.
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "blocked": False}
        )

        # Build the full conversation context for the LLM.
        llm_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. All user messages have been "
                    "pre-screened by Azure Content Safety (Prompt Shield + harm analysis)."
                ),
            }
        ] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if not m.get("blocked")
        ]

        with st.spinner("💬 Waiting for LLM response…"):
            assistant_reply = call_llm(llm_messages, llm_api_key)

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_reply, "blocked": False}
        )

    # shield_result is None → credentials error already shown; do nothing else.


# ---------------------------------------------------------------------------
# PM Demo Feature — "Under the Hood" expander
# ---------------------------------------------------------------------------

st.markdown("---")
with st.expander("🛠️ Under the Hood: Shield Analysis Data", expanded=False):
    st.markdown(
        "Raw responses from **both** Azure safety checks for the most recent prompt."
    )

    shield_data  = st.session_state.last_shield_data
    content_data = st.session_state.last_content_data

    if shield_data is None and content_data is None:
        st.info("No prompt has been analyzed yet. Send a message above to see results here.")
    else:
        # ================================================================
        # CHECK 1 — Prompt Shield (injection / jailbreak)
        # ================================================================
        st.subheader("🔒 Check 1: Prompt Shield (Injection / Jailbreak)")

        if shield_data is None:
            st.warning("Prompt Shield result unavailable.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Shield Result",
                    "🚨 ATTACK" if shield_data["attackDetected"] else "✅ SAFE",
                )
            with c2:
                st.metric(
                    "Attack Detected",
                    "Yes" if shield_data["attackDetected"] else "No",
                )

            upa = shield_data.get("userPromptAnalysis", {})
            if isinstance(upa, dict) and upa:
                cols = st.columns(max(len(upa), 1))
                for i, (cat, val) in enumerate(upa.items()):
                    with cols[i % len(cols)]:
                        display = ("🔴 True" if val else "🟢 False") if isinstance(val, bool) else str(val)
                        st.metric(label=cat, value=display)

            with st.expander("📋 Raw JSON — Prompt Shield"):
                st.json(shield_data["raw"])

        st.markdown("---")

        # ================================================================
        # CHECK 2 — Content Analysis (harmful content categories)
        # ================================================================
        st.subheader("🔍 Check 2: Content Analysis (Hate / Violence / Self-Harm / Sexual)")

        if content_data is None:
            st.warning("Content analysis result unavailable.")
        else:
            c3, c4 = st.columns(2)
            with c3:
                st.metric(
                    "Content Result",
                    "🚨 HARMFUL" if content_data["harmDetected"] else "✅ SAFE",
                )
            with c4:
                st.metric(
                    "Harm Detected",
                    "Yes" if content_data["harmDetected"] else "No",
                )

            # Severity per category — 0=safe, 2=low, 4=medium, 6=high
            categories = content_data.get("categoriesAnalysis", [])
            if categories:
                severity_label = {0: "🟢 0 — Safe", 2: "🟡 2 — Low", 4: "🟠 4 — Medium", 6: "🔴 6 — High"}
                cat_cols = st.columns(len(categories))
                for i, cat in enumerate(categories):
                    sev = cat.get("severity", 0)
                    with cat_cols[i]:
                        st.metric(
                            label=cat.get("category", "?"),
                            value=severity_label.get(sev, f"Severity {sev}"),
                        )

            with st.expander("📋 Raw JSON — Content Analysis"):
                st.json(content_data["raw"])

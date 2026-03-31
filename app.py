"""
Shielded Chat Interface — multi-provider safety PoC
=====================================================
Supported safety providers:
  • Azure Content Safety  (Prompt Shield + analyzeText)
  • AWS Bedrock Guardrails

Run:
    streamlit run app.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from openai import OpenAI

from safety_providers import PROVIDER_NAMES, SafetyResult, build_provider

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LITELLM_BASE_URL = os.environ.get("LITELLM_API_BASE", "https://api.iq.cudasvc.com")
LITELLM_MODEL = "gpt-4o-mini"

_NO_KEY_RESPONSE = (
    "⚠️  *LLM key not configured.* Your prompt was safe and would have been "
    "forwarded to the model. Set `LITELLM_API_KEY` (or enter it in the sidebar) "
    "to enable live responses."
)


# ---------------------------------------------------------------------------
# LLM call via IQ API
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict], llm_api_key: str) -> str:
    if not llm_api_key:
        return _NO_KEY_RESPONSE
    try:
        client = OpenAI(api_key=llm_api_key, base_url=LITELLM_BASE_URL)
        completion = client.chat.completions.create(
            model=LITELLM_MODEL, messages=messages
        )
        return completion.choices[0].message.content or "(empty response)"
    except Exception as exc:
        return f"⚠️ LLM call failed: {exc}"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Shielded Chat — Multi-Provider Safety PoC",
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar — configuration panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    # ---- Safety provider selection ----------------------------------------
    st.subheader("🛡️ Safety Provider")
    selected_provider_name: str = st.selectbox(
        "Provider",
        PROVIDER_NAMES,
        help="Choose which safety API analyses prompts before they reach the LLM.",
    )

    # ---- Provider-specific credentials (loaded from env, not shown in UI) --
    if selected_provider_name == "Azure Content Safety":
        provider_kwargs = {
            "azure_endpoint": os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT", ""),
            "azure_key": os.environ.get("AZURE_CONTENT_SAFETY_KEY", ""),
        }

    elif selected_provider_name == "AWS Bedrock Guardrails":
        provider_kwargs = {
            "guardrail_id": os.environ.get("AWS_GUARDRAIL_ID", ""),
            "guardrail_version": os.environ.get("AWS_GUARDRAIL_VERSION", "DRAFT"),
            "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
        }

    st.markdown("---")

    # ---- LLM settings -----------------------------------------------------
    st.subheader("🤖 Model via IQ")
    llm_api_key = st.text_input(
        "IQ API Key",
        value="",
        type="password",
        placeholder="Enter your IQ API key…",
        help="Your personal API key for the IQ API at api.iq.cudasvc.com.",
    )
    llm_model = st.text_input(
        "Model name",
        value=LITELLM_MODEL,
        help="Model identifier exposed by the IQ API.",
    )

    st.markdown("---")
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.session_state.last_safety_result = None
        st.rerun()

    st.markdown("---")
    st.caption(
        "Shielded Chat PoC · supports Azure Content Safety & AWS Bedrock Guardrails"
    )


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_safety_result" not in st.session_state:
    st.session_state.last_safety_result = None

if "last_provider_name" not in st.session_state:
    st.session_state.last_provider_name = None


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🛡️ Shielded Chat Interface")
st.markdown(
    f"Every message is analysed by **{selected_provider_name}** before reaching the LLM. "
    "Blocked prompts are never forwarded."
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("blocked"):
            st.error(f"🚫 *[BLOCKED]* {msg['content']}")
        else:
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Chat input & interception flow
# ---------------------------------------------------------------------------

user_input: str | None = st.chat_input("Type your message…")

if user_input:
    # 1. Show the user's message in the UI immediately.
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Build the selected provider and analyse the prompt.
    provider = build_provider(selected_provider_name, **provider_kwargs)

    with st.spinner(f"🔍 Analysing with {selected_provider_name}…"):
        result: SafetyResult = provider.analyze(user_input)

    st.session_state.last_safety_result = result
    st.session_state.last_provider_name = selected_provider_name

    # 3a. Provider configuration / network error.
    if result.error:
        st.error(f"🔴 Safety provider error: {result.error}")

    # 3b. Prompt blocked.
    elif result.blocked:
        with st.chat_message("assistant"):
            st.error(
                f"🚨 **Prompt Blocked.**\n\n{result.reason or 'Blocked by safety provider.'}\n\n"
                "This prompt has **not** been forwarded to the LLM."
            )
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "blocked": True}
        )

    # 3c. Safe — forward to LLM.
    else:
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "blocked": False}
        )

        llm_messages = [
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant. All user messages have been "
                    f"pre-screened by {selected_provider_name}."
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


# ---------------------------------------------------------------------------
# "Under the Hood" expander — safety analysis details
# ---------------------------------------------------------------------------

st.markdown("---")
with st.expander("🛠️ Under the Hood: Safety Analysis", expanded=False):
    result: SafetyResult | None = st.session_state.last_safety_result
    provider_name: str | None = st.session_state.last_provider_name

    if result is None:
        st.info("No prompt has been analysed yet. Send a message above to see results here.")
    else:
        st.markdown(f"**Provider:** {provider_name}")

        if result.error:
            st.error(f"Provider error: {result.error}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Result", "🚨 BLOCKED" if result.blocked else "✅ SAFE")
            with col2:
                st.metric("Blocked", "Yes" if result.blocked else "No")

            if result.reason:
                st.warning(f"**Reason:** {result.reason}")

            # ---- Azure-specific detailed view --------------------------------
            if provider_name == "Azure Content Safety" and result.details:
                shield = result.details.get("shield") or {}
                content = result.details.get("content") or {}

                st.subheader("🔒 Check 1: Prompt Shield (Injection / Jailbreak)")
                if not shield:
                    st.warning("Prompt Shield result unavailable.")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(
                            "Shield Result",
                            "🚨 ATTACK" if shield.get("attackDetected") else "✅ SAFE",
                        )
                    with c2:
                        st.metric(
                            "Attack Detected",
                            "Yes" if shield.get("attackDetected") else "No",
                        )
                    upa = shield.get("userPromptAnalysis", {})
                    if isinstance(upa, dict) and upa:
                        cols = st.columns(max(len(upa), 1))
                        for i, (cat, val) in enumerate(upa.items()):
                            display = (
                                ("🔴 True" if val else "🟢 False")
                                if isinstance(val, bool)
                                else str(val)
                            )
                            with cols[i % len(cols)]:
                                st.metric(label=cat, value=display)
                    with st.expander("📋 Raw JSON — Prompt Shield"):
                        st.json(shield.get("raw", {}))

                st.markdown("---")
                st.subheader("🔍 Check 2: Content Analysis (Hate / Violence / Self-Harm / Sexual)")
                if not content:
                    st.warning("Content analysis result unavailable.")
                else:
                    c3, c4 = st.columns(2)
                    with c3:
                        st.metric(
                            "Content Result",
                            "🚨 HARMFUL" if content.get("harmDetected") else "✅ SAFE",
                        )
                    with c4:
                        st.metric(
                            "Harm Detected",
                            "Yes" if content.get("harmDetected") else "No",
                        )
                    categories = content.get("categoriesAnalysis", [])
                    if categories:
                        severity_label = {
                            0: "🟢 0 — Safe",
                            2: "🟡 2 — Low",
                            4: "🟠 4 — Medium",
                            6: "🔴 6 — High",
                        }
                        cat_cols = st.columns(len(categories))
                        for i, cat in enumerate(categories):
                            sev = cat.get("severity", 0)
                            with cat_cols[i]:
                                st.metric(
                                    label=cat.get("category", "?"),
                                    value=severity_label.get(sev, f"Severity {sev}"),
                                )
                    with st.expander("📋 Raw JSON — Content Analysis"):
                        st.json(content.get("raw", {}))

            # ---- AWS or generic fallback ------------------------------------
            else:
                with st.expander("📋 Raw Details"):
                    st.json(result.details)

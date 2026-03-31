"""
Safety provider abstraction layer.

Usage:
    provider = AzureShieldProvider(endpoint, key)
    result   = provider.analyze(user_text)
    if result.error:
        ...handle config/network error...
    if result.blocked:
        ...reject prompt...
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Standardised result
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    """Uniform return type from every SafetyProvider.analyze() call."""

    blocked: bool
    # Human-readable explanation when blocked=True.
    reason: str | None = None
    # Provider-specific raw data shown in "Under the Hood".
    details: dict = field(default_factory=dict)
    # Non-None when the API call itself failed (credentials, network, …).
    error: str | None = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SafetyProvider(ABC):
    """All safety back-ends implement this interface."""

    @abstractmethod
    def analyze(self, text: str) -> SafetyResult:
        """Analyse *text* and return a SafetyResult.

        Implementations must NEVER raise — surface errors via
        ``SafetyResult(blocked=False, error="<message>")``.
        """


# ---------------------------------------------------------------------------
# Azure Content Safety (Prompt Shield + analyzeText)
# ---------------------------------------------------------------------------

class AzureShieldProvider(SafetyProvider):
    """
    Runs two Azure Content Safety checks in sequence:
      1. Prompt Shield  — detects prompt-injection / jailbreak attacks
      2. analyzeText    — detects harmful content (hate, violence, self-harm, sexual)

    Credentials are read from constructor args (pass env-var values from the
    caller — keeps this module free of Streamlit imports).

    Severity levels returned by analyzeText: 0=Safe, 2=Low, 4=Medium, 6=High.
    """

    HARM_SEVERITY_THRESHOLD = 2

    def __init__(self, endpoint: str, key: str) -> None:
        self.endpoint = endpoint.rstrip("/") if endpoint else ""
        self.key = key

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> SafetyResult:
        if not self.endpoint or not self.key:
            return SafetyResult(
                blocked=False,
                error="Azure Content Safety credentials are missing. "
                      "Set AZURE_CONTENT_SAFETY_ENDPOINT and AZURE_CONTENT_SAFETY_KEY.",
            )

        shield = self._shield_prompt(text)
        if shield is None:
            return SafetyResult(
                blocked=False,
                error="Azure Prompt Shield call failed — check endpoint / key.",
            )

        content = self._analyze_content(text)
        # analyzeText failure is non-fatal for the shield check; we surface it
        # in details but do not abort.

        attack_blocked = shield.get("attackDetected", False)
        harm_blocked = (
            content is not None and content.get("harmDetected", False)
        )

        reason: str | None = None
        if attack_blocked:
            reason = "Prompt injection / jailbreak attack detected by Azure Prompt Shield."
        elif harm_blocked:
            triggered = [
                f"{c['category']} (severity {c['severity']})"
                for c in (content.get("categoriesAnalysis") or [])
                if c.get("severity", 0) >= self.HARM_SEVERITY_THRESHOLD
            ]
            reason = f"Harmful content detected: {', '.join(triggered) or 'unknown category'}."

        return SafetyResult(
            blocked=attack_blocked or harm_blocked,
            reason=reason,
            details={"shield": shield, "content": content},
        )

    # ------------------------------------------------------------------
    # Private helpers — no Streamlit calls, return None on error
    # ------------------------------------------------------------------

    def _shield_prompt(self, text: str) -> dict | None:
        """Call the Prompt Shield REST endpoint; return structured dict or None."""
        import requests  # lazy import — only needed when this provider is used

        url = (
            f"{self.endpoint}"
            "/contentsafety/text:shieldPrompt"
            "?api-version=2024-02-15-preview"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json",
        }
        payload = {"userPrompt": text, "documents": []}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            if not resp.ok:
                return None

            raw: dict = resp.json()
            upa: dict = raw.get("userPromptAnalysis", {})
            return {
                "attackDetected": bool(upa.get("attackDetected", False)),
                "userPromptAnalysis": upa,
                "raw": raw,
            }
        except Exception:
            return None

    def _analyze_content(self, text: str) -> dict | None:
        """Call the analyzeText endpoint; return structured dict or None."""
        import requests

        url = (
            f"{self.endpoint}"
            "/contentsafety/text:analyze"
            "?api-version=2023-10-01"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "categories": ["Hate", "SelfHarm", "Sexual", "Violence"],
            "outputType": "FourSeverityLevels",
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            if not resp.ok:
                return None

            raw: dict = resp.json()
            categories: list = raw.get("categoriesAnalysis", [])
            harm_detected = any(
                c.get("severity", 0) >= self.HARM_SEVERITY_THRESHOLD
                for c in categories
            )
            return {
                "harmDetected": harm_detected,
                "categoriesAnalysis": categories,
                "raw": raw,
            }
        except Exception:
            return None


# ---------------------------------------------------------------------------
# AWS Bedrock Guardrails
# ---------------------------------------------------------------------------

class AWSGuardrailProvider(SafetyProvider):
    """
    Uses the AWS Bedrock ``apply_guardrail`` API to evaluate a prompt.

    Credentials are resolved by boto3 in the standard order
    (env vars → ~/.aws/credentials → IAM role).

    Required config:
        guardrail_id      — Guardrail resource ID from AWS console
        guardrail_version — e.g. "DRAFT" or a published version number
        region            — AWS region where the guardrail is deployed
    """

    def __init__(
        self,
        guardrail_id: str,
        guardrail_version: str,
        region: str = "us-east-1",
    ) -> None:
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.region = region

    def analyze(self, text: str) -> SafetyResult:
        if not self.guardrail_id or not self.guardrail_version:
            return SafetyResult(
                blocked=False,
                error="AWS Guardrail credentials are missing. "
                      "Set AWS_GUARDRAIL_ID and AWS_GUARDRAIL_VERSION.",
            )

        try:
            import boto3  # lazy import

            client = boto3.client("bedrock-runtime", region_name=self.region)
            response = client.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source="INPUT",
                content=[{"text": {"text": text}}],
            )
        except Exception as exc:
            return SafetyResult(
                blocked=False,
                error=f"AWS Guardrail call failed: {exc}",
            )

        action: str = response.get("action", "NONE")
        blocked = action == "GUARDRAIL_INTERVENED"

        reason: str | None = None
        if blocked:
            outputs = response.get("outputs", [])
            reason = (
                outputs[0].get("text", "Blocked by AWS Bedrock Guardrail.")
                if outputs
                else "Blocked by AWS Bedrock Guardrail."
            )

        # Scrub the raw response of boto3 ResponseMetadata noise for display.
        raw_display = {
            k: v for k, v in response.items() if k != "ResponseMetadata"
        }

        return SafetyResult(
            blocked=blocked,
            reason=reason,
            details={"action": action, "raw": raw_display},
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

PROVIDER_NAMES = ["Azure Content Safety", "AWS Bedrock Guardrails"]


def build_provider(name: str, **kwargs) -> SafetyProvider:
    """Instantiate a provider by display name, forwarding keyword args."""
    if name == "Azure Content Safety":
        return AzureShieldProvider(
            endpoint=kwargs.get("azure_endpoint", ""),
            key=kwargs.get("azure_key", ""),
        )
    if name == "AWS Bedrock Guardrails":
        return AWSGuardrailProvider(
            guardrail_id=kwargs.get("guardrail_id", ""),
            guardrail_version=kwargs.get("guardrail_version", "DRAFT"),
            region=kwargs.get("aws_region", "us-east-1"),
        )
    raise ValueError(f"Unknown provider: {name!r}")

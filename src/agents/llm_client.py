"""Unified LLM Client abstraction supporting Anthropic Claude, Google Gemini, and OpenAI.

Enables seamless switching of LLM providers via the LLM_PROVIDER or JOBSCOUT_PROVIDER environment variable:
- 'anthropic' (default): Claude models (claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5)
- 'gemini': Google Gemini models (gemini-2.5-flash, gemini-2.5-pro)
- 'openai': OpenAI models (gpt-4o, gpt-4o-mini)
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any


def _get_secret(key: str, default: str | None = None) -> str | None:
    """Safely retrieves a configuration key or API key from os.environ,
    Streamlit secrets (st.secrets), or a default fallback."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1500,
    ) -> str:
        pass

    @abstractmethod
    def generate_structured_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        pass


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from anthropic import Anthropic
        self.api_key = api_key or _get_secret("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Please add it to your Streamlit Secrets or .env file.")
        self.client = Anthropic(api_key=self.api_key)
        self.model = model or _get_secret("JOBSCOUT_MODEL", "claude-haiku-4-5")

    def _thinking_kwargs(self) -> dict:
        if "haiku" in self.model.lower():
            return {}
        return {"thinking": {"type": "adaptive"}}

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1500,
    ) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            **self._thinking_kwargs(),
        )
        return next(b.text for b in resp.content if b.type == "text")

    def generate_structured_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai
        self.api_key = api_key or _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please add GEMINI_API_KEY to your Streamlit Secrets (under Advanced Settings) or your .env file.")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model or _get_secret("JOBSCOUT_MODEL", "gemini-2.5-flash")
        # Flash-tier models only for free-tier compatibility
        self.fallback_models = [self.model, "gemini-2.5-flash", "gemini-1.5-flash"]

    def _call_with_retry(self, prompt: str, is_json: bool = False, max_retries: int = 4) -> str:
        import time
        import random
        last_err = None

        for model_name in self.fallback_models:
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    if response and response.text:
                        return response.text
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    # Retry on temporary high-demand (503), rate-limits (429), or network timeouts
                    if any(code in err_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "timeout")):
                        sleep_s = (1.5 ** attempt) + random.uniform(0.2, 0.8)
                        time.sleep(sleep_s)
                    else:
                        break  # Try next fallback model if unsupported model error

        raise last_err or RuntimeError("Gemini API call failed after retries.")

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1500,
    ) -> str:
        prompt = f"{system_prompt}\n\nUser Request:\n{user_prompt}" if system_prompt else user_prompt
        return self._call_with_retry(prompt, is_json=False)

    def generate_structured_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        prompt = (
            f"{system_prompt}\n\n"
            f"You MUST respond ONLY with a valid JSON object adhering strictly to this JSON Schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Input:\n{user_prompt}"
        )
        raw = self._call_with_retry(prompt, is_json=True)
        return parse_json_resiliently(raw)


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from openai import OpenAI
        self.api_key = api_key or _get_secret("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please add it to your Streamlit Secrets or .env file.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model or _get_secret("JOBSCOUT_MODEL", "gpt-4o-mini")

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1500,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return resp.choices[0].message.content or ""

    def generate_structured_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema, "strict": False},
                },
            )
            text = resp.choices[0].message.content or "{}"
        except Exception:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
            )
            text = resp.choices[0].message.content or "{}"

        return parse_json_resiliently(text)


def parse_json_resiliently(text: str) -> dict[str, Any]:
    """Parse JSON safely even with single quotes, unescaped newlines/tabs, or markdown fences."""
    if not text or not str(text).strip():
        return {}
    raw = str(text).strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    raw = raw.strip()
    if not raw:
        return {}

    # 1. Direct standard parse (lenient strict=False)
    try:
        res = json.loads(raw, strict=False)
        if isinstance(res, dict):
            return res
    except Exception:
        pass

    # 2. Extract substring between first { and last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        sub = raw[start:end+1]
        try:
            res = json.loads(sub, strict=False)
            if isinstance(res, dict):
                return res
        except Exception:
            pass

        # 3. Python dict literal eval (handles single quotes, True/False/None)
        try:
            import ast
            res = ast.literal_eval(sub)
            if isinstance(res, dict):
                return res
        except Exception:
            pass

        # 4. Clean control chars and quote repair
        import re
        try:
            cleaned = re.sub(r'[\x00-\x1f]+', ' ', sub)
            res = json.loads(cleaned, strict=False)
            if isinstance(res, dict):
                return res
        except Exception:
            pass

        try:
            # Replace single quotes with double quotes around keys and values
            repaired = re.sub(r"'([a-zA-Z0-9_]+)':", r'"\1":', sub)
            repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
            repaired = repaired.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
            res = json.loads(repaired, strict=False)
            if isinstance(res, dict):
                return res
        except Exception:
            pass

    return {}


class OmniRouteClient(BaseLLMClient):
    """Client for OmniRoute / local OpenAI-compatible routing gateways.
    OmniRoute rotates multiple API keys, balances traffic, and handles failovers.
    """
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        from openai import OpenAI
        self.base_url = (
            base_url
            or os.getenv("OMNIROUTE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "http://localhost:8000/v1"
        )
        self.api_key = (
            api_key
            or os.getenv("OMNIROUTE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "omniroute-local"
        )
        self.model = model or os.getenv("JOBSCOUT_MODEL") or os.getenv("OMNIROUTE_MODEL", "auto/fast")
        self.fallback_models = [self.model, "auto/fast", "auto/chat", "auto/best-coding", "auto"]
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=12.0, max_retries=1)

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1500,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        last_err = None
        for m in self.fallback_models:
            try:
                resp = self.client.chat.completions.create(
                    model=m,
                    max_tokens=max_tokens,
                    messages=messages,
                    timeout=12.0,
                )
                if resp.choices and resp.choices[0].message.content:
                    return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                continue
        # Direct Gemini fallback if key is present
        if os.getenv("GEMINI_API_KEY"):
            try:
                return GeminiClient().generate_text(system_prompt, user_prompt, max_tokens)
            except Exception:
                pass
        raise last_err or RuntimeError("OmniRoute failed across all fallback models.")

    def generate_structured_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        prompt = (
            f"{system_prompt}\n\n"
            f"You MUST respond ONLY with a valid JSON object adhering strictly to this JSON Schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Input:\n{user_prompt}"
        )
        messages = [{"role": "user", "content": prompt}]

        last_err = None
        for m in self.fallback_models:
            try:
                try:
                    resp = self.client.chat.completions.create(
                        model=m,
                        max_tokens=max_tokens,
                        messages=messages,
                        timeout=12.0,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "response", "schema": schema, "strict": False},
                        },
                    )
                    text = resp.choices[0].message.content or "{}"
                except Exception:
                    resp = self.client.chat.completions.create(
                        model=m,
                        max_tokens=max_tokens,
                        messages=messages,
                        timeout=12.0,
                    )
                    text = resp.choices[0].message.content or "{}"

                parsed = parse_json_resiliently(text)
                if parsed:
                    return parsed
            except Exception as e:
                last_err = e
                continue

        # Direct Gemini fallback if key is present
        if _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY"):
            try:
                return GeminiClient().generate_structured_json(system_prompt, user_prompt, schema, max_tokens)
            except Exception:
                pass

        # Safe fallback matching schema structure
        fallback = {}
        for prop, prop_spec in schema.get("properties", {}).items():
            if prop_spec.get("type") == "object":
                fallback[prop] = {}
            elif prop_spec.get("type") == "array":
                fallback[prop] = []
            elif prop_spec.get("type") == "string":
                fallback[prop] = ""
            elif prop_spec.get("type") in ("integer", "number"):
                fallback[prop] = 70
        return fallback


def get_llm_client() -> BaseLLMClient:
    """Factory to get the configured LLM client instance."""
    provider = _get_secret("LLM_PROVIDER") or _get_secret("JOBSCOUT_PROVIDER", "")
    provider = provider.lower().strip() if provider else ""

    if not provider:
        if _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY"):
            provider = "gemini"
        elif _get_secret("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif _get_secret("OPENAI_API_KEY"):
            provider = "openai"
        elif _get_secret("OMNIROUTE_BASE_URL"):
            provider = "omniroute"
        else:
            provider = "gemini"

    if provider in ("omniroute", "proxy", "local_proxy", "litellm"):
        return OmniRouteClient()
    elif provider in ("gemini", "google"):
        return GeminiClient()
    elif provider in ("openai", "gpt"):
        return OpenAIClient()
    else:
        return AnthropicClient()

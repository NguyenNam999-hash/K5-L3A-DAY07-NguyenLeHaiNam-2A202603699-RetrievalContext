"""OpenAI-compatible chat client with retries and usage/cost capture."""
from __future__ import annotations

import json
import time
from typing import Any


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        session_id: str = "",
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        if not api_key or not base_url or not model:
            raise ValueError("api_key, base_url and model are required")
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("openai is not installed; run pip install -r requirements.txt") from exc
        self._openai = openai
        headers = {"User-Agent": "day7-agentic-rag/1.0"}
        if session_id:
            headers["x-opencode-session"] = session_id
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            default_headers=headers,
        )
        self.model = model
        self.max_retries = max_retries
        self.calls = 0
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "list_cost_usd": 0.0,
        }

    @staticmethod
    def _message_to_dict(message: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ]
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            payload["reasoning_content"] = reasoning
        return payload

    @staticmethod
    def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
        if usage is None:
            return None
        payload: dict[str, Any]
        if hasattr(usage, "model_dump"):
            payload = usage.model_dump()
        elif isinstance(usage, dict):
            payload = dict(usage)
        else:
            payload = {"raw": str(usage)}
        cost = getattr(usage, "cost", None)
        if cost is not None:
            payload["cost_usd"] = float(cost)
        opencode = getattr(usage, "opencode", None)
        if opencode is not None:
            payload["opencode"] = opencode
            if isinstance(opencode, dict) and opencode.get("list_cost") is not None:
                payload["list_cost_usd"] = float(opencode["list_cost"])
        return payload

    def _record_usage(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.total_usage[key] += int(usage.get(key) or 0)
        list_cost = usage.get("list_cost_usd")
        if list_cost is None:
            opencode = usage.get("opencode") or {}
            list_cost = opencode.get("list_cost") if isinstance(opencode, dict) else None
        if list_cost is None:
            list_cost = usage.get("cost_usd")
        if list_cost is not None:
            self.total_usage["list_cost_usd"] += float(list_cost)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        temperature: float = 0.0,
        max_tokens: int = 1200,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        openai = self._openai
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            ) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise LLMError(f"LLM request failed after {self.max_retries + 1} attempts: {exc}") from exc
                time.sleep(min(2 ** attempt, 20))
            except openai.APIStatusError as exc:
                if exc.status_code in (401, 403):
                    raise LLMError(f"LLM authentication/permission error ({exc.status_code}): {exc}") from exc
                last_error = exc
                if attempt >= self.max_retries:
                    raise LLMError(f"LLM request failed: {exc}") from exc
                time.sleep(min(2 ** attempt, 20))
        else:  # pragma: no cover - loop always breaks or raises
            raise LLMError(f"LLM request failed: {last_error}")
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        self.calls += 1
        usage = self._usage_to_dict(getattr(response, "usage", None))
        self._record_usage(usage)
        choice = response.choices[0]
        return {
            "message": self._message_to_dict(choice.message),
            "finish_reason": choice.finish_reason,
            "usage": usage,
            "latency_ms": elapsed_ms,
            "model": getattr(response, "model", self.model),
        }

    def as_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "usage": self.total_usage,
        }

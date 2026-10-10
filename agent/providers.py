"""Provider adapters sharing the OpenAI chat/tool message format."""

from __future__ import annotations

import os
from typing import Any


def secret(config: dict, key: str = "api_key") -> str | None:
    """Explicit environment references fail closed instead of using a dummy key."""
    env = config.get(f"{key}_env")
    if env:
        value = os.environ.get(env)
        if not value:
            raise ValueError(f"Required environment variable is not set: {env}")
        return value
    return config.get(key) or None


class Provider:
    def __init__(self, config: dict):
        self.config = config
        self.backend = config.get("backend", "openai")
        if self.backend not in {"openai", "anthropic", "litellm"}:
            raise ValueError(f"Unknown LLM backend: {self.backend}")
        if not config.get("model"):
            raise ValueError("llm.model is required")
        self.key = secret(config)
        self.client: Any = None
        options = {
            "timeout": config.get("timeout", 60),
            "max_retries": config.get("max_retries", 2),
        }
        if config.get("base_url"):
            options["base_url"] = config["base_url"]
        if self.backend == "openai":
            from openai import OpenAI

            default_key = (
                os.getenv("OPENAI_API_KEY")
                if not config.get("base_url")
                or config["base_url"].rstrip("/") == "https://api.openai.com/v1"
                else None
            )
            self.client = OpenAI(
                api_key=self.key or default_key or "not-needed",
                **options,
            )
        elif self.backend == "anthropic":
            from anthropic import Anthropic

            self.client = Anthropic(
                api_key=self.key or os.getenv("ANTHROPIC_API_KEY"), **options
            )

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        params = dict(self.config.get("parameters", {}))
        reserved = {
            "model",
            "messages",
            "tools",
            "stream",
            "api_key",
            "api_base",
            "base_url",
            "system",
        }
        if reserved.intersection(params):
            raise ValueError(
                "llm.parameters cannot override routing, messages, tools, or credentials"
            )
        for key in ("temperature", "max_tokens"):
            if self.config.get(key) is not None:
                params[key] = self.config[key]
        params["model"] = self.config["model"]
        if self.backend == "anthropic":
            return self._anthropic(messages, tools, params)
        params["messages"] = messages
        if tools:
            params["tools"] = tools
        if self.backend == "litellm":
            try:
                from litellm import completion
            except ImportError as exc:
                raise RuntimeError(
                    "Install requirements-providers.txt for the litellm backend"
                ) from exc
            params["timeout"] = self.config.get("timeout", 60)
            params["num_retries"] = self.config.get("max_retries", 2)
            if self.key:
                params["api_key"] = self.key
            if self.config.get("base_url"):
                params["api_base"] = self.config["base_url"]
            response = completion(**params)
        else:
            response = self.client.chat.completions.create(**params)
        message = response.choices[0].message
        result = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = [
                call.model_dump(exclude_none=True) for call in message.tool_calls
            ]
        return result

    def _anthropic(self, messages, tools, params):
        import json

        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        converted = []
        for message in messages:
            role = message["role"]
            if role == "system":
                continue
            if role == "tool":
                role = "user"
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"],
                    }
                ]
            else:
                blocks = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message.get("tool_calls", []):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "input": json.loads(call["function"]["arguments"]),
                        }
                    )
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": blocks})
        params.setdefault("max_tokens", 1024)
        params.update(system=system, messages=converted)
        if tools:
            params["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]
        response = self.client.messages.create(**params)
        result = {
            "role": "assistant",
            "content": "".join(b.text for b in response.content if b.type == "text"),
        }
        calls = [
            {
                "id": b.id,
                "type": "function",
                "function": {"name": b.name, "arguments": json.dumps(b.input)},
            }
            for b in response.content
            if b.type == "tool_use"
        ]
        if calls:
            result["tool_calls"] = calls
        return result

    def close(self):
        if self.client is not None:
            self.client.close()

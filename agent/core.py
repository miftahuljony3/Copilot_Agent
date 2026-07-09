"""
agent/core.py
Central agent loop: receives a user message, decides which tools to call,
builds a prompt with conversation history from memory, sends the request
to the configured LLM backend, and returns the response.
"""

from __future__ import annotations

import logging
from typing import Any

from .memory import Memory
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


class PersonalAgent:
    """
    Orchestrates a single conversational session.

    Parameters
    ----------
    config : dict
        Parsed contents of config.yaml (or a dict with equivalent keys).
    memory : Memory
        Persistent memory store for the agent.
    tools : ToolRegistry
        Registry of callable tools the agent may invoke.
    """

    def __init__(self, config: dict, memory: Memory, tools: ToolRegistry) -> None:
        self.config = config
        self.memory = memory
        self.tools = tools
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send *user_message* to the agent and return the assistant reply."""
        # 1. Persist the user turn
        self.memory.add("user", user_message)

        # 2. Build the messages list that will be sent to the LLM
        messages = self._build_messages()

        # 3. Call the LLM
        reply = self._call_llm(messages)

        # 4. Persist the assistant turn
        self.memory.add("assistant", reply)

        return reply

    def reset(self) -> None:
        """Clear the current conversation history (keeps long-term memory)."""
        self.memory.clear_conversation()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self) -> list[dict]:
        """Construct the messages list from system prompt + conversation history."""
        system_prompt = self.config.get("agent", {}).get(
            "system_prompt",
            "You are a helpful personal AI assistant.",
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.memory.get_conversation())
        return messages

    def _call_llm(self, messages: list[dict]) -> str:
        """
        Call the configured LLM backend.
        Supports OpenAI-compatible REST APIs (Ollama, LM Studio, OpenAI, etc.)
        and the Anthropic Claude API.
        """
        backend = self.config.get("llm", {}).get("backend", "openai")

        if backend == "anthropic":
            return self._call_anthropic(messages)

        # Default: OpenAI-compatible
        return self._call_openai_compatible(messages)

    def _call_openai_compatible(self, messages: list[dict]) -> str:
        """OpenAI / Ollama / LM Studio / any OpenAI-compatible endpoint."""
        llm_cfg = self.config.get("llm", {})
        response = self._client.chat.completions.create(
            model=llm_cfg.get("model", "gpt-3.5-turbo"),
            messages=messages,
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=llm_cfg.get("max_tokens", 1024),
        )
        return (response.choices[0].message.content or "").strip()

    def _call_anthropic(self, messages: list[dict]) -> str:
        """Anthropic Claude API."""
        import anthropic  # lazy import

        llm_cfg = self.config.get("llm", {})
        # Anthropic expects system message passed separately
        system = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        human_messages = [m for m in messages if m["role"] != "system"]

        response = self._client.messages.create(
            model=llm_cfg.get("model", "claude-3-haiku-20240307"),
            system=system,
            messages=human_messages,
            max_tokens=llm_cfg.get("max_tokens", 1024),
        )
        return response.content[0].text.strip()

    def _build_client(self) -> Any:
        """Instantiate and return the appropriate API client."""
        llm_cfg = self.config.get("llm", {})
        backend = llm_cfg.get("backend", "openai")

        if backend == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=llm_cfg.get("api_key", ""))

        # OpenAI-compatible
        import openai
        return openai.OpenAI(
            api_key=llm_cfg.get("api_key") or "not-needed",
            base_url=llm_cfg.get("base_url", "http://localhost:11434/v1"),
        )

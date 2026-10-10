"""Bounded model/tool loop with configurable API providers and MCP clients."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from jsonschema import validate

from .mcp_client import MCPTools
from .memory import Memory
from .providers import Provider
from .tools import ToolRegistry


class PersonalAgent:
    def __init__(
        self,
        config: dict,
        memory: Memory,
        tools: ToolRegistry,
        approve_tool: Callable[[str, dict], bool] | None = None,
    ):
        self.config, self.memory, self.tools = config, memory, tools
        self.approve_tool = approve_tool
        self.provider = Provider(config.get("llm", {}))

    def chat(self, user_message: str) -> str:
        """Synchronous CLI entry point; async callers should await achat()."""
        return asyncio.run(self.achat(user_message))

    async def achat(self, user_message: str) -> str:
        agent_cfg = self.config.get("agent", {})
        policy = self.config.get("tools", {})
        enabled = policy.get("enabled", True)
        rounds = int(policy.get("max_rounds", 8))
        max_calls = int(policy.get("max_calls", 32))
        max_chars = int(policy.get("max_result_chars", 20000))
        if rounds < 1 or max_calls < 1 or max_chars < 1:
            raise ValueError("Tool limits must be positive")
        messages = [
            {
                "role": "system",
                "content": agent_cfg.get(
                    "system_prompt", "You are a helpful personal AI assistant."
                ),
            },
            *self.memory.get_conversation(),
            {"role": "user", "content": user_message},
        ]
        # Only successful user/final-answer pairs enter memory. Intermediate tool
        # protocol messages stay together in this turn, never a truncated window.
        mcp_config = self.config.get("mcp", {}) if enabled else {}
        async with MCPTools(mcp_config) as remote:
            schemas = (self.tools.schemas() + remote.schemas) if enabled else []
            lookup = {
                t["function"]["name"]: t["function"]["parameters"] for t in schemas
            }
            calls_made = 0
            for _ in range(rounds):
                answer = await asyncio.to_thread(
                    self.provider.complete, messages, schemas
                )
                messages.append(answer)
                calls = answer.get("tool_calls", [])
                if not calls:
                    reply = (answer.get("content") or "").strip()
                    self.memory.add("user", user_message)
                    self.memory.add("assistant", reply)
                    return reply
                for call in calls:
                    calls_made += 1
                    if calls_made > max_calls:
                        raise RuntimeError(
                            "Tool call limit reached; narrow your request"
                        )
                    name = call["function"]["name"]
                    try:
                        if name not in lookup:
                            raise ValueError("Unknown tool")
                        args = json.loads(call["function"]["arguments"])
                        if not isinstance(args, dict):
                            raise TypeError("Tool arguments must be a JSON object")
                        validate(args, lookup[name])
                        if policy.get("require_confirmation", True) and (
                            self.approve_tool is None
                            or not self.approve_tool(name, args)
                        ):
                            result = "[ToolDenied] User approval is required; tool was not executed."
                        elif name in remote.tools:
                            result = await remote.call(name, args)
                        else:
                            result = await asyncio.to_thread(
                                self.tools.call, name, **args
                            )
                    except Exception as exc:  # noqa: BLE001 — isolate untrusted tool failures
                        # Exceptions may include provider URLs, headers, or secrets.
                        result = f"[ToolError] {type(exc).__name__}; check arguments and server configuration."
                    if len(result) > max_chars:
                        result = result[:max_chars] + "\n[Tool result truncated]"
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": result}
                    )
        raise RuntimeError("Tool round limit reached; narrow your request")

    def reset(self):
        self.memory.clear_conversation()

    def close(self):
        self.provider.close()

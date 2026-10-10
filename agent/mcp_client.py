"""Opt-in MCP tools over stdio and Streamable HTTP (MCP SDK 1.x)."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from contextlib import AsyncExitStack
from datetime import timedelta

from .providers import secret


class MCPTools:
    """Connections live for one turn; all contexts enter/exit in the same task."""

    def __init__(self, config: dict):
        self.config = config
        self.stack = AsyncExitStack()
        self.tools: dict[str, tuple] = {}
        self.schemas: list[dict] = []

    async def __aenter__(self):
        try:
            for server, cfg in self.config.get("servers", {}).items():
                if cfg.get("enabled", True):
                    await self._connect(server, cfg)
            return self
        except BaseException:
            await self.stack.aclose()
            raise

    async def __aexit__(self, *args):
        return await self.stack.__aexit__(*args)

    async def _connect(self, server, cfg):
        from mcp import ClientSession, StdioServerParameters

        timeout = float(cfg.get("timeout", 30))
        if timeout <= 0:
            raise ValueError("MCP timeout must be positive")
        transport = cfg.get("transport", "stdio")
        if transport == "stdio":
            from mcp.client.stdio import stdio_client

            # SDK adds its minimal platform environment. Only named secrets are forwarded.
            env = dict(cfg.get("env", {}))
            for name in cfg.get("env_passthrough", []):
                if name not in os.environ:
                    raise ValueError(
                        f"Required MCP environment variable is not set: {name}"
                    )
                env[name] = os.environ[name]
            params = StdioServerParameters(
                command=cfg["command"], args=cfg.get("args", []), env=env
            )
            streams = await self.stack.enter_async_context(stdio_client(params))
        elif transport == "streamable_http":
            from mcp.client.streamable_http import streamablehttp_client

            headers = dict(cfg.get("headers", {}))
            token = secret(cfg, "bearer_token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            streams = await self.stack.enter_async_context(
                streamablehttp_client(
                    cfg["url"],
                    headers=headers,
                    timeout=timedelta(seconds=timeout),
                    sse_read_timeout=timedelta(seconds=timeout),
                )
            )
        else:
            raise ValueError(f"Unsupported MCP transport: {transport}")
        session = await self.stack.enter_async_context(
            ClientSession(
                streams[0], streams[1], read_timeout_seconds=timedelta(seconds=timeout)
            )
        )
        await session.initialize()
        cursor = None
        seen = set()
        while True:
            listing = await session.list_tools(cursor=cursor)
            for tool in listing.tools:
                if "allowed_tools" in cfg and tool.name not in cfg["allowed_tools"]:
                    continue
                identity = f"{server}\0{tool.name}"
                suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
                prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", f"mcp_{server}_{tool.name}")[
                    :50
                ]
                name = f"{prefix}_{suffix}"
                if name in self.tools:
                    raise ValueError(f"Duplicate MCP tool: {name}")
                self.tools[name] = (session, tool.name, timeout)
                self.schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": f"[{server}] {tool.description or tool.name}",
                            "parameters": tool.inputSchema,
                        },
                    }
                )
            cursor = listing.nextCursor
            if not cursor:
                break
            if cursor in seen:
                raise ValueError("MCP server repeated a pagination cursor")
            seen.add(cursor)

    async def call(self, name, arguments):
        session, remote_name, timeout = self.tools[name]
        result = await asyncio.wait_for(
            session.call_tool(remote_name, arguments), timeout
        )
        # JSON preserves structured content and isError, not just text blocks.
        return result.model_dump_json(exclude_none=True)

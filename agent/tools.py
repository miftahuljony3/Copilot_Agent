"""
agent/tools.py
A lightweight tool-calling framework.

Define tools by subclassing BaseTool and decorating them with @ToolRegistry.register,
or register callables directly via ToolRegistry.register_function().

Built-in tools
--------------
- WebSearch  : simple DuckDuckGo Lite scrape (no API key required)
- Calculator : evaluates a safe arithmetic expression
- NoteTaker  : appends a note to data/notes.md
"""

from __future__ import annotations

import ast
import logging
import operator
import pathlib
from abc import ABC, abstractmethod
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """Abstract base class every tool must implement."""

    name: str = ""          # unique snake_case identifier
    description: str = ""   # one-sentence description shown to the LLM

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool and return a string result."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Stores and dispatches tools.

    Usage
    -----
    registry = ToolRegistry()
    registry.register(MyTool())
    result = registry.call("my_tool", param="value")
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Add a tool to the registry."""
        self._tools[tool.name] = tool
        logger.debug("Tool registered: %s", tool.name)

    def call(self, name: str, **kwargs) -> str:
        """Look up and invoke a tool by name.  Returns an error string if not found."""
        if name not in self._tools:
            return f"[ToolError] Unknown tool: '{name}'. Available: {self.list_names()}"
        try:
            return self._tools[name].run(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool '%s' raised an exception", name)
            return f"[ToolError] {name} failed: {exc}"

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def describe_all(self) -> str:
        """Return a formatted string describing all registered tools (for the system prompt)."""
        if not self._tools:
            return "No tools available."
        lines = [f"- {t.name}: {t.description}" for t in self._tools.values()]
        return "\n".join(lines)

    @classmethod
    def with_defaults(cls) -> "ToolRegistry":
        """Return a registry pre-loaded with the built-in tools."""
        reg = cls()
        reg.register(Calculator())
        reg.register(NoteTaker())
        reg.register(WebSearch())
        return reg


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

_SAFE_OPS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


class Calculator(BaseTool):
    """Evaluate a safe arithmetic expression without using eval()."""

    name = "calculator"
    description = "Evaluate a safe arithmetic expression, e.g. '2 + 3 * 4'. Returns the numeric result."

    def run(self, expression: str = "") -> str:  # noqa: D102
        try:
            result = self._safe_eval(ast.parse(expression, mode="eval").body)
            return str(result)
        except Exception as exc:  # noqa: BLE001
            return f"[CalculatorError] {exc}"

    def _safe_eval(self, node: ast.expr):  # type: ignore[override]
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(
                    f"Unsupported constant type: {type(node.value).__name__}"
                )
            return node.value
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            return _SAFE_OPS[op_type](self._safe_eval(node.left), self._safe_eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            return _SAFE_OPS[op_type](self._safe_eval(node.operand))
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class NoteTaker(BaseTool):
    """Append a freeform note to a local markdown file."""

    name = "note_taker"
    description = "Save an important note or piece of information to data/notes.md for later review."

    def run(self, note: str = "") -> str:  # noqa: D102
        notes_path = pathlib.Path("data/notes.md")
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        import datetime
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        with notes_path.open("a", encoding="utf-8") as f:
            f.write(f"\n---\n**{timestamp}**\n{note}\n")
        return f"Note saved to {notes_path}."


class WebSearch(BaseTool):
    """Perform a simple web search using the DuckDuckGo Lite HTML page."""

    name = "web_search"
    description = "Search the web for a query and return a short summary of the top results."

    def run(self, query: str = "") -> str:  # noqa: D102
        try:
            import urllib.parse
            import urllib.request

            url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote_plus(query)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Strip HTML tags with a minimal approach (no third-party deps)
            import re
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            # Return first 800 chars as a snippet
            return text[:800] if text else "No results found."
        except Exception as exc:  # noqa: BLE001
            return f"[WebSearchError] {exc}"

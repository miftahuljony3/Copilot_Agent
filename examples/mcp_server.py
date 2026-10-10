"""Trusted local MCP demo: python examples/mcp_server.py (stdio)."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator-demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")

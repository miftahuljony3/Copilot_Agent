"""
agent/cli.py
Command-line interface for the personal agent.

Commands
--------
  chat        Start an interactive chat session.
  reset       Clear the current conversation window.
  export      Export conversation history as training data.
  stats       Show training data statistics.
  history     Print recent conversation turns from long-term memory.
"""

from __future__ import annotations

import logging
import pathlib
import sys

import click
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .core import PersonalAgent
from .memory import Memory
from .tools import ToolRegistry
from .trainer import Trainer

console = Console()
logger = logging.getLogger(__name__)

CONFIG_DEFAULT = pathlib.Path("config.yaml")


def _load_config(config_path: pathlib.Path) -> dict:
    if not config_path.exists():
        console.print(
            f"[red]Config file not found: {config_path}[/red]\n"
            "Copy config.yaml.example to config.yaml and fill in your settings."
        )
        sys.exit(1)
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_agent(config: dict) -> tuple[PersonalAgent, Memory, Trainer]:
    memory = Memory(
        db_path=config.get("memory", {}).get("db_path", "data/memory.db"),
        max_window=config.get("memory", {}).get("max_window", 20),
    )
    tools = ToolRegistry.with_defaults()
    agent = PersonalAgent(config=config, memory=memory, tools=tools)
    trainer = Trainer(memory=memory)
    return agent, memory, trainer


@click.group()
@click.option(
    "--config",
    default=str(CONFIG_DEFAULT),
    show_default=True,
    help="Path to config.yaml",
)
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """Personal AI Agent — local & trainable."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = pathlib.Path(config)


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Start an interactive chat session. Type 'exit' or Ctrl-C to quit."""
    config = _load_config(ctx.obj["config_path"])
    agent, _memory, _trainer = _build_agent(config)

    agent_name = config.get("agent", {}).get("name", "Agent")
    console.print(
        Panel(
            f"[bold green]Personal Agent — {agent_name}[/bold green]\n"
            "Type [bold]exit[/bold] or press [bold]Ctrl-C[/bold] to quit.\n"
            "Type [bold]/reset[/bold] to clear conversation history.",
            expand=False,
        )
    )

    try:
        while True:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "bye"):
                console.print("[yellow]Goodbye![/yellow]")
                break
            if user_input == "/reset":
                agent.reset()
                console.print("[yellow]Conversation reset.[/yellow]")
                continue

            reply = agent.chat(user_input)
            console.print(
                Panel(
                    Markdown(reply),
                    title=f"[bold magenta]{agent_name}[/bold magenta]",
                    expand=False,
                )
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Session ended.[/yellow]")


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def reset(ctx: click.Context) -> None:
    """Clear the in-memory conversation window."""
    config = _load_config(ctx.obj["config_path"])
    _agent, memory, _ = _build_agent(config)
    memory.clear_conversation()
    console.print("[green]Conversation window cleared.[/green]")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--fmt",
    type=click.Choice(["openai", "alpaca", "sharegpt"]),
    default="openai",
    show_default=True,
    help="Output fine-tune format.",
)
@click.pass_context
def export(ctx: click.Context, fmt: str) -> None:
    """Export conversation history as a fine-tuning dataset."""
    config = _load_config(ctx.obj["config_path"])
    _agent, _memory, trainer = _build_agent(config)

    system_prompt = config.get("agent", {}).get(
        "system_prompt", "You are a helpful personal AI assistant."
    )

    raw_path = trainer.export_raw()
    console.print(f"[green]Raw pairs exported to:[/green] {raw_path}")

    out_path = trainer.build_finetune_dataset(fmt=fmt, system_prompt=system_prompt)  # type: ignore[arg-type]
    console.print(f"[green]Fine-tune dataset ({fmt}) written to:[/green] {out_path}")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show statistics about the collected training data."""
    config = _load_config(ctx.obj["config_path"])
    _agent, _memory, trainer = _build_agent(config)
    data = trainer.stats()
    for key, val in data.items():
        console.print(f"  [bold]{key}[/bold]: {val}")


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--limit", default=20, show_default=True, help="Number of turns to show.")
@click.pass_context
def history(ctx: click.Context, limit: int) -> None:
    """Print recent turns from long-term memory."""
    config = _load_config(ctx.obj["config_path"])
    _agent, memory, _ = _build_agent(config)
    turns = memory.get_history(limit=limit)
    for turn in turns:
        role_color = "cyan" if turn["role"] == "user" else "magenta"
        console.print(
            f"[{role_color}]{turn['role'].upper()}[/{role_color}] "
            f"[dim]{turn.get('timestamp', '')}[/dim]\n"
            f"{turn['content']}\n"
        )

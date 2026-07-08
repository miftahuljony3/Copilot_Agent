"""
main.py
Entry point for the Personal Agent.

Usage
-----
  python main.py chat             # interactive chat
  python main.py export           # export training data (OpenAI format)
  python main.py export --fmt alpaca
  python main.py stats            # training data statistics
  python main.py history          # recent conversation turns
  python main.py --help           # all commands

The CLI is built with Click; run any sub-command with --help for details.
"""

import logging
import pathlib
import sys

import yaml

from agent.cli import cli


def _configure_logging(config_path: pathlib.Path = pathlib.Path("config.yaml")) -> None:
    """Read logging settings from config and apply them."""
    level = logging.INFO
    log_file: str | None = None

    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        log_cfg = cfg.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
        log_file = log_cfg.get("file")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )


if __name__ == "__main__":
    _configure_logging()
    cli(obj={})

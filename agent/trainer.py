"""
agent/trainer.py
Utilities for manually curating and preparing training data from the
agent's conversation history, ready for fine-tuning an LLM.

Workflow
--------
1. Run the agent in normal chat mode (all turns are stored in SQLite).
2. Call `Trainer.export_raw()` to dump raw conversation pairs as JSONL.
3. Review / edit the JSONL file to keep only good examples.
4. Call `Trainer.build_finetune_dataset()` to convert to the OpenAI
   fine-tune format (or Alpaca / ShareGPT format).
5. Upload the resulting JSONL to your fine-tuning platform.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Literal

from .memory import DEFAULT_EXPORT_LIMIT

logger = logging.getLogger(__name__)

Format = Literal["openai", "alpaca", "sharegpt"]


class Trainer:
    """
    Converts raw conversation pairs into training-ready datasets.

    Parameters
    ----------
    memory : Memory
        The agent's Memory instance (used as the data source).
    output_dir : str | pathlib.Path
        Directory where dataset files will be written.
    """

    def __init__(self, memory, output_dir: str | pathlib.Path = "data/training") -> None:
        self.memory = memory
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_raw(self, limit: int = DEFAULT_EXPORT_LIMIT) -> pathlib.Path:
        """
        Export raw (prompt, completion) pairs from memory to JSONL.
        Returns the path of the written file.
        """
        count = self.memory.export_training_pairs(
            str(self.output_dir / "raw_pairs.jsonl"),
            limit=limit,
        )
        logger.info("Exported %d raw pairs.", count)
        return self.output_dir / "raw_pairs.jsonl"

    def build_finetune_dataset(
        self,
        source: str | pathlib.Path | None = None,
        fmt: Format = "openai",
        system_prompt: str = "You are a helpful personal AI assistant.",
    ) -> pathlib.Path:
        """
        Convert raw pairs to the target fine-tune format and write to disk.

        Parameters
        ----------
        source : path to the raw JSONL, defaults to ``data/training/raw_pairs.jsonl``
        fmt    : output format — 'openai', 'alpaca', or 'sharegpt'
        system_prompt : system message prepended to each example

        Returns the path of the output file.
        """
        source = pathlib.Path(source) if source else self.output_dir / "raw_pairs.jsonl"
        if not source.exists():
            raise FileNotFoundError(
                f"Raw pairs file not found: {source}. Run export_raw() first."
            )

        out_path = self.output_dir / f"finetune_{fmt}.jsonl"
        converter = {
            "openai": self._to_openai,
            "alpaca": self._to_alpaca,
            "sharegpt": self._to_sharegpt,
        }[fmt]

        written = 0
        with source.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                record = converter(pair, system_prompt)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        logger.info("Wrote %d fine-tune examples (%s) to %s", written, fmt, out_path)
        return out_path

    # ------------------------------------------------------------------
    # Format converters
    # ------------------------------------------------------------------

    @staticmethod
    def _to_openai(pair: dict, system_prompt: str) -> dict:
        """OpenAI fine-tune JSONL format (chat completions)."""
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pair["prompt"]},
                {"role": "assistant", "content": pair["completion"]},
            ]
        }

    @staticmethod
    def _to_alpaca(pair: dict, system_prompt: str) -> dict:
        """Stanford Alpaca instruction format."""
        return {
            "instruction": system_prompt,
            "input": pair["prompt"],
            "output": pair["completion"],
        }

    @staticmethod
    def _to_sharegpt(pair: dict, system_prompt: str) -> dict:
        """ShareGPT / FastChat format."""
        return {
            "system": system_prompt,
            "conversations": [
                {"from": "human", "value": pair["prompt"]},
                {"from": "gpt", "value": pair["completion"]},
            ],
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return basic statistics about the collected training data."""
        raw = self.output_dir / "raw_pairs.jsonl"
        if not raw.exists():
            return {"total_pairs": 0, "note": "Run export_raw() first."}

        total, total_prompt_len, total_completion_len = 0, 0, 0
        with raw.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                total += 1
                total_prompt_len += len(pair.get("prompt", ""))
                total_completion_len += len(pair.get("completion", ""))

        return {
            "total_pairs": total,
            "avg_prompt_chars": round(total_prompt_len / total, 1) if total else 0,
            "avg_completion_chars": round(total_completion_len / total, 1) if total else 0,
        }

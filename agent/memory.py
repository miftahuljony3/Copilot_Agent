"""
agent/memory.py
Handles short-term conversation history (in-memory list) and long-term
persistent storage (SQLite via sqlite-utils).
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
from typing import Literal

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]
DEFAULT_EXPORT_LIMIT = 10_000


class Memory:
    """
    Dual-layer memory:
    - **Short-term**: a sliding window of recent turns kept in RAM.
    - **Long-term**: every turn is written to SQLite so the agent can
      recall past sessions and so training examples can be harvested later.

    Parameters
    ----------
    db_path : str | pathlib.Path
        Path to the SQLite file (created if it doesn't exist).
    max_window : int
        Maximum number of turns kept in the short-term window.
    """

    def __init__(
        self,
        db_path: str | pathlib.Path = "data/memory.db",
        max_window: int = 20,
    ) -> None:
        self.max_window = max_window
        self._conversation: list[dict] = []  # short-term window
        self._db = self._init_db(pathlib.Path(db_path))

    # ------------------------------------------------------------------
    # Short-term (conversation window)
    # ------------------------------------------------------------------

    def add(self, role: Role, content: str) -> None:
        """Append a turn to both the short-term window and long-term store."""
        turn = {"role": role, "content": content}
        self._conversation.append(turn)
        # Trim window
        if len(self._conversation) > self.max_window:
            self._conversation = self._conversation[-self.max_window:]
        self._persist(role, content)

    def get_conversation(self) -> list[dict]:
        """Return the current short-term window (list of {role, content})."""
        return list(self._conversation)

    def clear_conversation(self) -> None:
        """Wipe the short-term window without touching the database."""
        self._conversation.clear()
        logger.debug("Short-term memory cleared.")

    # ------------------------------------------------------------------
    # Long-term (SQLite)
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 100) -> list[dict]:
        """Return the *limit* most recent turns from the long-term store."""
        rows = list(
            self._db["turns"].rows_where(
                order_by="id desc", limit=limit
            )
        )
        rows.reverse()
        return rows

    def export_training_pairs(
        self, output_path: str = "data/training/pairs.jsonl", limit: int = DEFAULT_EXPORT_LIMIT
    ) -> int:
        """
        Export consecutive (user, assistant) pairs from long-term memory
        as JSONL for fine-tuning.  Returns the number of pairs written.
        """
        import jsonlines  # lazy import

        rows = self.get_history(limit=limit)
        pairs: list[dict] = []
        for i, row in enumerate(rows):
            if row["role"] == "user" and i + 1 < len(rows):
                next_row = rows[i + 1]
                if next_row["role"] == "assistant":
                    pairs.append(
                        {
                            "prompt": row["content"],
                            "completion": next_row["content"],
                            "session_id": row.get("session_id", ""),
                            "timestamp": row.get("timestamp", ""),
                        }
                    )

        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(output_path, mode="w") as writer:
            writer.write_all(pairs)

        logger.info("Exported %d training pairs to %s", len(pairs), output_path)
        return len(pairs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_db(self, path: pathlib.Path):
        """Create (or open) the SQLite database and ensure schema exists."""
        try:
            import sqlite_utils
        except ImportError as exc:
            raise ImportError(
                "sqlite-utils is required for persistent memory. "
                "Run: pip install sqlite-utils"
            ) from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite_utils.Database(path)
        if "turns" not in db.table_names():
            db["turns"].create(
                {
                    "id": int,
                    "session_id": str,
                    "role": str,
                    "content": str,
                    "timestamp": str,
                },
                pk="id",
            )
        return db

    def _persist(self, role: Role, content: str) -> None:
        """Write a single turn to the long-term SQLite store."""
        self._db["turns"].insert(
            {
                "session_id": self._session_id(),
                "role": role,
                "content": content,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }
        )

    @staticmethod
    def _session_id() -> str:
        """Return the ISO-date string used as a session identifier."""
        return datetime.datetime.utcnow().strftime("%Y-%m-%d")

"""Coverage for pure-logic paths: trainer, memory pairing, registry, tools.

These tests require no network or paid credentials; they exercise the
data-preparation and tool-dispatch layers that the provider/MCP tests skip.
"""

from __future__ import annotations

import json

import pytest

from agent.core import PersonalAgent
from agent.memory import Memory
from agent.tools import BaseTool, Calculator, NoteTaker, ToolRegistry, WebSearch
from agent.trainer import Trainer

# ---------------------------------------------------------------------------
# Memory: training-pair extraction
# ---------------------------------------------------------------------------


def test_export_training_pairs(tmp_path):
    memory = Memory(tmp_path / "m.db")
    try:
        memory.add("user", "hi")
        memory.add("assistant", "hello")
        memory.add("user", "bye")
        memory.add("assistant", "goodbye")
        out = tmp_path / "pairs.jsonl"
        assert memory.export_training_pairs(str(out)) == 2
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [(r["prompt"], r["completion"]) for r in rows] == [
            ("hi", "hello"),
            ("bye", "goodbye"),
        ]
    finally:
        memory.close()


def test_export_skips_unpaired_turns(tmp_path):
    memory = Memory(tmp_path / "m.db")
    try:
        # A user turn with no assistant reply must not emit a pair.
        memory.add("user", "orphan")
        memory.add("user", "q")
        memory.add("assistant", "a")
        out = tmp_path / "pairs.jsonl"
        assert memory.export_training_pairs(str(out)) == 1
        row = json.loads(out.read_text().splitlines()[0])
        assert row["prompt"] == "q"
    finally:
        memory.close()


def test_history_returns_chronological_order(tmp_path):
    memory = Memory(tmp_path / "m.db")
    try:
        for i in range(5):
            memory.add("user", f"m{i}")
        recent = memory.get_history(limit=3)
        assert [r["content"] for r in recent] == ["m2", "m3", "m4"]
    finally:
        memory.close()


# ---------------------------------------------------------------------------
# Trainer: format converters
# ---------------------------------------------------------------------------


@pytest.fixture
def trainer(tmp_path):
    memory = Memory(tmp_path / "m.db")
    memory.add("user", "ping")
    memory.add("assistant", "pong")
    yield Trainer(memory, output_dir=tmp_path / "training")
    memory.close()


def test_export_raw_then_stats(trainer):
    trainer.export_raw()
    stats = trainer.stats()
    assert stats["total_pairs"] == 1
    assert stats["avg_prompt_chars"] == 4.0
    assert stats["avg_completion_chars"] == 4.0


def test_stats_without_export(tmp_path):
    memory = Memory(tmp_path / "m.db")
    try:
        trainer = Trainer(memory, output_dir=tmp_path / "t")
        assert trainer.stats()["total_pairs"] == 0
    finally:
        memory.close()


@pytest.mark.parametrize("fmt", ["openai", "alpaca", "sharegpt"])
def test_build_finetune_dataset(trainer, fmt):
    trainer.export_raw()
    out = trainer.build_finetune_dataset(fmt=fmt, system_prompt="SYS")
    record = json.loads(out.read_text().splitlines()[0])
    if fmt == "openai":
        assert [m["role"] for m in record["messages"]] == [
            "system",
            "user",
            "assistant",
        ]
        assert record["messages"][1]["content"] == "ping"
    elif fmt == "alpaca":
        assert record == {"instruction": "SYS", "input": "ping", "output": "pong"}
    else:
        assert record["system"] == "SYS"
        assert record["conversations"][0] == {"from": "human", "value": "ping"}


def test_build_requires_export_first(trainer):
    with pytest.raises(FileNotFoundError, match="export_raw"):
        trainer.build_finetune_dataset(fmt="openai")


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_registry_rejects_duplicate():
    reg = ToolRegistry()
    reg.register(Calculator())
    with pytest.raises(ValueError, match="Duplicate"):
        reg.register(Calculator())


def test_registry_unknown_tool_returns_error():
    reg = ToolRegistry()
    assert "Unknown tool" in reg.call("nope")


def test_registry_describe_and_schemas():
    reg = ToolRegistry.with_defaults()
    assert "calculator" in reg.describe_all()
    names = {s["function"]["name"] for s in reg.schemas()}
    assert names == {"calculator", "note_taker", "web_search"}


def test_registry_isolates_tool_exceptions():
    class Boom(BaseTool):
        name = "boom"
        description = "always fails"

        def run(self, **_):
            raise RuntimeError("secret detail")

    reg = ToolRegistry()
    reg.register(Boom())
    result = reg.call("boom")
    assert result.startswith("[ToolError]")
    assert "secret detail" not in result  # only the exception type is surfaced


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("10 / 4", "2.5"),
        ("10 % 3", "1"),
        ("-5 + 2", "-3"),
        ("2 ** 8", "256"),
    ],
)
def test_calculator_valid(expr, expected):
    assert Calculator().run(expr) == expected


@pytest.mark.parametrize(
    "expr",
    ["__import__('os')", "1 << 2", "'a' + 'b'", "x + 1"],
)
def test_calculator_rejects_unsafe(expr):
    assert "Error" in Calculator().run(expr)


def test_calculator_length_guard():
    assert "too long" in Calculator().run("1+" * 300 + "1")


# ---------------------------------------------------------------------------
# NoteTaker
# ---------------------------------------------------------------------------


def test_note_taker_appends(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    msg = NoteTaker().run("remember the milk")
    assert "saved" in msg.lower()
    content = (tmp_path / "data" / "notes.md").read_text()
    assert "remember the milk" in content
    assert "UTC" in content


# ---------------------------------------------------------------------------
# WebSearch (offline: the network call is monkeypatched)
# ---------------------------------------------------------------------------


def test_web_search_network_failure_is_contained(monkeypatch):
    import urllib.request

    def boom(*_a, **_k):
        raise OSError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert WebSearch().run("anything").startswith("[WebSearchError]")


# ---------------------------------------------------------------------------
# Core: oversized tool results are truncated before re-entering the model loop
# ---------------------------------------------------------------------------


def _assistant_call(name="calculator", arguments='{"expression":"2+3"}', cid="c1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def test_oversized_tool_result_is_truncated(tmp_path):
    memory = Memory(tmp_path / "m.db")
    agent = PersonalAgent(
        {
            "llm": {"model": "test", "base_url": "http://localhost:9999/v1"},
            "tools": {"max_result_chars": 20},
        },
        memory,
        ToolRegistry.with_defaults(),
        lambda *_: True,
    )
    captured: list = []

    def complete(messages, _schemas):
        captured.append([dict(m) for m in messages])
        if len(captured) == 1:
            return _assistant_call()
        return {"role": "assistant", "content": "done"}

    agent.provider.complete = complete
    agent.tools.call = lambda *a, **k: "X" * 500  # type: ignore[method-assign]
    try:
        assert agent.chat("calc") == "done"
        tool_msg = captured[1][-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"].endswith("[Tool result truncated]")
        assert len(tool_msg["content"]) < 500
    finally:
        agent.close()
        memory.close()

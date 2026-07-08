# Personal Agent

A locally-hostable, self-trainable personal AI agent built in Python.
Use it to learn how LLM agents work, accumulate conversation data, and
progressively fine-tune your own model.

---

## Project structure

```
Copilot_Agent/
├── agent/
│   ├── __init__.py      # package exports
│   ├── core.py          # agent loop (LLM calls, tool dispatch)
│   ├── memory.py        # short-term window + long-term SQLite store
│   ├── tools.py         # tool framework + built-in tools
│   ├── trainer.py       # export & convert training data
│   └── cli.py           # Click CLI (chat / export / stats / history)
├── data/
│   ├── memory.db        # SQLite conversation store (auto-created)
│   ├── notes.md         # notes saved by the note_taker tool
│   └── training/        # exported fine-tune datasets
├── logs/
│   └── agent.log        # runtime log (auto-created)
├── config.yaml          # all settings (LLM, memory, training)
├── main.py              # CLI entry point
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start a local model (recommended for privacy)

Install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3
ollama serve          # starts the OpenAI-compatible API on :11434
```

Or use [LM Studio](https://lmstudio.ai) — just start the local server and
update `base_url` in `config.yaml`.

### 3. Configure

Edit `config.yaml`:

```yaml
llm:
  backend: "openai"
  base_url: "http://localhost:11434/v1"
  model: "llama3"
```

For cloud models (OpenAI / Anthropic) see the commented examples in `config.yaml`.

### 4. Chat

```bash
python main.py chat
```

---

## CLI reference

| Command | Description |
|---------|-------------|
| `python main.py chat` | Interactive chat session |
| `python main.py reset` | Clear in-memory conversation window |
| `python main.py history` | Show recent turns from long-term memory |
| `python main.py export` | Export raw + OpenAI fine-tune JSONL |
| `python main.py export --fmt alpaca` | Export in Alpaca format |
| `python main.py export --fmt sharegpt` | Export in ShareGPT format |
| `python main.py stats` | Print training data statistics |

---

## Built-in tools

| Tool | Description |
|------|-------------|
| `calculator` | Safe arithmetic expression evaluator |
| `note_taker` | Appends a note to `data/notes.md` |
| `web_search` | Searches DuckDuckGo Lite (no API key needed) |

Add your own tools by subclassing `agent.tools.BaseTool` and registering
them in `agent/tools.py`.

---

## Training workflow

All conversations are persisted in `data/memory.db`.  To build a fine-tune
dataset:

```bash
# 1. Export raw pairs
python main.py export

# 2. Review / curate data/training/raw_pairs.jsonl manually

# 3. Convert to Alpaca format for fine-tuning with tools like Axolotl / Unsloth
python main.py export --fmt alpaca
```

The resulting JSONL files in `data/training/` are ready to upload to
OpenAI fine-tuning, [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl),
[Unsloth](https://github.com/unslothai/unsloth), or any JSONL-based
fine-tune pipeline.

---

## Adding a new LLM backend

1. Add a new `_call_<backend>` method in `agent/core.py`.
2. Add the backend name to the `if/elif` chain in `_call_llm()`.
3. Add relevant keys to `config.yaml`.

---

## License

MIT
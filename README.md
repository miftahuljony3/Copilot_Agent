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
# Copilot Agent

A collection of custom GitHub Copilot agent profiles and repository guidance.

> Update this README to match the actual agents and workflows in this repository.

## Contents

- `.github/agents/` — specialized agent profiles.
- `.github/copilot-instructions.md` — repository-wide guidance for Copilot.
- `docs/agent-evaluation.md` — manual scenarios for checking agent behavior.

## Getting started

1. Clone or download this repository.
2. Review the agent profiles in `.github/agents/`.
3. Copy or adapt the profiles for your own repository if desired.
4. Open the repository in a Copilot-compatible environment.
5. Select a custom agent from the agent picker, if your environment supports it.

Custom agent availability and supported features can vary by Copilot client and configuration. Check the documentation for the client you use.

## Included agents

### Repository Researcher

Inspects repository files and reports evidence-backed findings. It distinguishes observed facts from assumptions and identifies information it could not verify.

### Implementation Planner

Turns a feature request or review findings into a prioritized implementation plan with affected files, acceptance criteria, and risks.

### Code Reviewer

Reviews changes for correctness, security, and maintainability. It reports findings with file and line references and does not modify files.

## Customizing these agents

Before using these profiles in another project:

- Replace generic guidance with the project’s actual languages, build commands, and conventions.
- Keep each agent’s responsibility focused.
- Grant only the tools the agent needs.
- Test agent behavior using the scenarios in `docs/agent-evaluation.md`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.

# Personal Agent — API models + MCP

A Python chat agent with SQLite conversation history, provider-native tool calling,
and optional Model Context Protocol (MCP) tools. The repository also contains
GitHub Copilot profiles under `.github/agents/`; those profiles are separate from
this Python runtime and do not change Copilot's supported models.

## Quick start

Requires **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py chat               # defaults to Ollama localhost:11434/v1, llama3
```

Start Ollama and pull the configured model first, or choose a cloud profile:

```bash
export OPENAI_API_KEY="your-key"   # PowerShell: $env:OPENAI_API_KEY="your-key"
python main.py chat --profile openai
python main.py chat --profile openai --model YOUR_MODEL_ID
```

Use `config.local.yaml` (gitignored) for your settings:

```bash
cp config.yaml config.local.yaml
python main.py --config config.local.yaml chat
```

## Model providers

| Backend | Use |
| --- | --- |
| `openai` | OpenAI Chat Completions and compatible APIs: Ollama, LM Studio, OpenRouter, DeepSeek, Groq, vLLM, OmniRoute |
| `anthropic` | Native Anthropic Messages API, including tool-use/result blocks |
| `litellm` | Optional adapters for Gemini, Azure, Bedrock and other LiteLLM-supported providers |

Model IDs are configurable, not restricted to a hard-coded list. **Not every AI
model or API is interchangeable:** this is a text/chat runtime. Tool calls require
model **and endpoint** support. Images, audio, streaming, OpenAI Responses-only
models, and arbitrary proprietary API formats are not implemented. An unsupported
API needs an adapter or an OpenAI-compatible gateway.

### Custom API / OmniRoute

```yaml
llm:
  backend: openai
  base_url: http://localhost:20128/v1
  api_key_env: OMNIROUTE_API_KEY
  model: auto
  timeout: 60
  max_retries: 2
```

For OpenRouter use `https://openrouter.ai/api/v1` and its model ID; for LM Studio
use `http://localhost:1234/v1`. Specify the full API base including `/v1` when
required. If `base_url` is omitted, the native SDK default is used. A keyless local
OpenAI-compatible service receives `not-needed` as its key. Set `api_key_env` for
cloud services; a missing named variable fails immediately. Inline `api_key` is
supported for backward compatibility but should not be committed.

Named `profiles` replace the entire `llm` block, avoiding accidental reuse of a
local URL or another provider's credentials. `--model` overrides only the model.
Optional `temperature`, `max_tokens`, and a `parameters` mapping are forwarded to
the API. Omit options your selected model does not accept; they are not silently
dropped. Routing, messages, tools and credentials cannot be overridden through
`parameters`. Anthropic defaults to 1024 output tokens if not configured.

### Additional providers

```bash
pip install -r requirements-providers.txt
export GEMINI_API_KEY="your-key"
python main.py chat --profile gemini
```

LiteLLM profiles use provider-prefixed IDs, e.g. `gemini/YOUR_MODEL_ID` or
`bedrock/YOUR_MODEL_ID`. Native provider authentication (such as AWS credentials)
is handled by LiteLLM. Model availability and permissions depend on your account.
See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for provider-specific
parameters. The native OpenAI and Anthropic paths do not require LiteLLM.

## MCP tools

The agent acts as an **MCP client**. It connects to configured servers, discovers
their tool JSON schemas, lets the model request tool calls, validates arguments,
and sends results back to the model. Supported transports: **stdio** and
**Streamable HTTP**, using the MCP Python SDK 1.x. Legacy SSE transport, MCP
resources/prompts, OAuth discovery, and exposing this agent as an MCP server are
not implemented.

No servers are enabled by default. Try the included trusted local demo:

```yaml
mcp:
  servers:
    demo:
      transport: stdio
      command: python
      args: [examples/mcp_server.py]
      allowed_tools: [add]
      timeout: 30
```

Run from the repository root with your virtual environment active, then ask
“Use the demo MCP add tool to add 17 and 25.” Use an absolute executable and script
path when running from another directory. Connections are opened once per chat
turn and closed on completion/failure; stdio servers are restarted on each turn.
Only configure trusted commands: **connecting to a stdio server executes its
command before any individual tool approval**. Commands are executed without a
shell. Child processes receive the SDK's minimal environment plus `env` values
and explicitly named `env_passthrough` variables, not every parent secret.

Remote example:

```yaml
mcp:
  servers:
    docs:
      transport: streamable_http
      url: https://your-server.example/mcp
      bearer_token_env: MCP_ACCESS_TOKEN
      allowed_tools: [search]
      timeout: 30
```

Use HTTPS for remote servers. `headers` can supply non-secret custom headers.
`enabled: false` disables a server. Omit `allowed_tools` to expose all discovered
tools; `[]` exposes none. Names are namespaced with a deterministic suffix to
avoid collisions. Pagination is supported. Server failures stop that turn rather
than silently omitting requested capabilities; check server logs for details.

## Tool execution controls

```yaml
tools:
  enabled: true
  require_confirmation: true
  max_rounds: 8
  max_calls: 32
  max_result_chars: 20000
```

The CLI asks permission for each call (default **No**). Library callers must
supply `approve_tool(name, arguments)` or calls are denied. Set
`require_confirmation: false` only for tools you intentionally trust to act
without approval. `enabled: false` disables both built-in and MCP tools, useful
for models without tool calling. MCP outputs are untrusted input, not authority.

Built-ins: `calculator`, `note_taker` (writes `data/notes.md`), and `web_search`
(DuckDuckGo Lite). Custom `BaseTool` subclasses should define a JSON Schema in
`parameters`. A turn has bounded model rounds, tool calls, and result text size.
MCP calls have timeouts; synchronous custom Python tools must implement their
own time/resource limits. These controls are not a sandbox for untrusted code.

## CLI and memory

| Command | Purpose |
| --- | --- |
| `python main.py chat [--profile NAME] [--model ID]` | Interactive chat |
| `/reset` inside chat | Clear the in-memory conversation window |
| `python main.py history --limit 20` | Recent stored turns |
| `python main.py export --fmt openai` | Export user/assistant pairs |
| `python main.py export --fmt alpaca` | Alpaca training data |
| `python main.py export --fmt sharegpt` | ShareGPT training data |
| `python main.py stats` | Exported data statistics |

Only successful user/final-answer pairs are persisted. Intermediate tool results
are transient, avoiding broken tool-call pairs when the memory window is trimmed.
SQLite stores long-term history; a new process starts a fresh short-term context.
History/export/stats work without model credentials or MCP servers. Review exported
data before fine-tuning; this project prepares data, it does not automatically
train or improve model weights. Keep your local database and API keys private.

## Development

```bash
pip install pytest
python -m pytest -q
```

Tests use mocked provider HTTP responses plus real local stdio and Streamable
HTTP MCP servers; no paid API keys are required. Cloud-provider billing, model
access and compatibility need separate smoke tests with your own credentials.

- `agent/core.py`: bounded tool loop and memory
- `agent/providers.py`: OpenAI, Anthropic, LiteLLM adapters
- `agent/mcp_client.py`: MCP transports, discovery, calls, cleanup
- `agent/tools.py`: built-in tools and schemas
- `agent/cli.py`: profiles, model override and interactive approval

MIT license.

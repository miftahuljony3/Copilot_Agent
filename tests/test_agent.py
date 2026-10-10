import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup
from anthropic import Anthropic
from click.testing import CliRunner
from openai import OpenAI

from agent.cli import cli
from agent.core import PersonalAgent
from agent.mcp_client import MCPTools
from agent.memory import Memory
from agent.providers import Provider, secret
from agent.tools import ToolRegistry


def call(name="calculator", arguments='{"expression":"2+3"}', id="call_1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


@pytest.fixture
def agent(tmp_path):
    memory = Memory(tmp_path / "test.db")
    agent = PersonalAgent(
        {"llm": {"model": "test", "base_url": "http://localhost:9999/v1"}},
        memory,
        ToolRegistry.with_defaults(),
        lambda *_: True,
    )
    yield agent
    agent.close()
    memory.close()


def test_tool_roundtrip_and_memory(agent):
    captured = []

    def complete(messages, schemas):
        captured.append(copy.deepcopy(messages))
        assert any(s["function"]["name"] == "calculator" for s in schemas)
        return call() if len(captured) == 1 else {"role": "assistant", "content": "5"}

    agent.provider.complete = complete
    assert agent.chat("What is 2+3?") == "5"
    assert captured[1][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "5"}
    assert [m["role"] for m in agent.memory.get_conversation()] == ["user", "assistant"]


@pytest.mark.parametrize(
    "arguments",
    ["not-json", "[]", '{"expression":4}', '{"expression":"1","extra":true}'],
)
def test_invalid_arguments_not_executed(agent, arguments):
    agent.tools.call = Mock()
    agent.provider.complete = Mock(
        side_effect=[
            call(arguments=arguments),
            {"role": "assistant", "content": "retry"},
        ]
    )
    agent.chat("calculate")
    agent.tools.call.assert_not_called()


def test_denied_without_callback(agent):
    agent.approve_tool = None
    agent.tools.call = Mock()
    agent.provider.complete = Mock(
        side_effect=[call(), {"role": "assistant", "content": "denied"}]
    )
    agent.chat("calculate")
    agent.tools.call.assert_not_called()


def test_limits_and_failure_do_not_pollute_memory(agent):
    agent.config["tools"] = {"max_rounds": 1}
    agent.provider.complete = Mock(return_value=call())
    with pytest.raises(RuntimeError, match="round limit"):
        agent.chat("calculate forever")
    assert agent.memory.get_conversation() == []


def test_disabled_tools_do_not_start_mcp(agent):
    agent.config.update(
        tools={"enabled": False},
        mcp={"servers": {"bad": {"command": "does-not-exist"}}},
    )

    def complete(messages, tools):
        assert tools == []
        return {"role": "assistant", "content": "plain chat"}

    agent.provider.complete = complete
    assert agent.chat("hello") == "plain chat"


def test_openai_real_sdk_request():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chat1",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {"index": 0, "finish_reason": "tool_calls", "message": call()}
                ],
            },
        )

    p = Provider(
        {"model": "custom/model", "api_key": "test", "base_url": "http://localhost/v1"}
    )
    p.client.close()
    p.client = OpenAI(
        api_key="test",
        base_url="http://localhost/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        result = p.complete(
            [{"role": "user", "content": "hi"}], ToolRegistry.with_defaults().schemas()
        )
        assert result["tool_calls"][0]["function"]["name"] == "calculator"
        assert seen[0]["model"] == "custom/model"
        assert "temperature" not in seen[0]
    finally:
        p.close()


def test_anthropic_multi_tool_result_translation():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg1",
                "type": "message",
                "role": "assistant",
                "model": "test",
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    p = Provider({"backend": "anthropic", "model": "test", "api_key": "test"})
    p.client.close()
    p.client = Anthropic(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    calls = call()
    calls["tool_calls"].extend(call(id="call_2")["tool_calls"])
    messages = [
        {"role": "system", "content": "help"},
        {"role": "user", "content": "calculate"},
        calls,
        {"role": "tool", "tool_call_id": "call_1", "content": "5"},
        {"role": "tool", "tool_call_id": "call_2", "content": "5"},
    ]
    try:
        assert (
            p.complete(messages, ToolRegistry.with_defaults().schemas())["content"]
            == "done"
        )
        assert len(seen[0]["messages"][-1]["content"]) == 2
        assert seen[0]["messages"][-1]["content"][1]["tool_use_id"] == "call_2"
        assert seen[0]["system"] == "help"
    finally:
        p.close()


def test_litellm_routing(monkeypatch):
    completion = Mock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    p = Provider({"backend": "litellm", "model": "gemini/example", "api_key": "test"})
    assert p.complete([{"role": "user", "content": "hi"}], [])["content"] == "ok"
    assert completion.call_args.kwargs["model"] == "gemini/example"
    assert completion.call_args.kwargs["api_key"] == "test"


def test_missing_key_and_unknown_backend(monkeypatch):
    monkeypatch.delenv("NONEXISTENT_AGENT_KEY", raising=False)
    with pytest.raises(ValueError, match="NONEXISTENT_AGENT_KEY"):
        secret({"api_key_env": "NONEXISTENT_AGENT_KEY", "api_key": "fallback"})
    with pytest.raises(ValueError, match="Unknown"):
        Provider({"backend": "typo", "model": "test"})


def test_stdio_mcp_end_to_end(agent):
    agent.config["mcp"] = {
        "servers": {
            "demo": {
                "command": sys.executable,
                "args": [str(Path("examples/mcp_server.py").resolve())],
                "allowed_tools": ["add"],
            }
        }
    }
    captured = []

    def complete(messages, schemas):
        captured.append(copy.deepcopy(messages))
        name = next(
            t["function"]["name"]
            for t in schemas
            if t["function"]["name"].startswith("mcp_")
        )
        return (
            call(name, '{"a":17,"b":25}')
            if len(captured) == 1
            else {"role": "assistant", "content": "42"}
        )

    agent.provider.complete = complete
    assert agent.chat("add via MCP") == "42"
    result = json.loads(captured[-1][-1]["content"])
    assert result["isError"] is False
    assert any(c.get("text") == "42" for c in result["content"])


def test_mcp_allowlist_empty():
    async def run():
        async with MCPTools(
            {
                "servers": {
                    "demo": {
                        "command": sys.executable,
                        "args": ["examples/mcp_server.py"],
                        "allowed_tools": [],
                    }
                }
            }
        ) as tools:
            assert tools.schemas == []

    asyncio.run(run())


def test_storage_commands_without_provider_credentials(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"llm:\n  backend: invalid\nmemory:\n  db_path: {tmp_path}/memory.db\ntraining:\n  output_dir: {tmp_path}/training\n"
    )
    runner = CliRunner()
    for command in ("history", "stats", "export"):
        result = runner.invoke(cli, ["--config", str(cfg), command])
        assert result.exit_code == 0, result.output


def test_streamable_http_mcp():
    import socket

    import uvicorn
    from mcp.server.fastmcp import FastMCP

    async def run():
        demo = FastMCP("http-test", stateless_http=True, json_response=True)

        @demo.tool()
        def add(a: int, b: int) -> int:
            return a + b

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(demo.streamable_http_app(), log_level="error")
        )
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            with __import__("anyio").fail_after(10):
                while not server.started:
                    if task.done():
                        await task
                    await asyncio.sleep(0.01)
            async with MCPTools(
                {
                    "servers": {
                        "http": {
                            "transport": "streamable_http",
                            "url": f"http://127.0.0.1:{port}/mcp",
                        }
                    }
                }
            ) as tools:
                result = json.loads(
                    await tools.call(next(iter(tools.tools)), {"a": 3, "b": 4})
                )
                assert result["isError"] is False
                assert result["content"][0]["text"] == "7"
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 10)
            sock.close()

    asyncio.run(run())


def test_calculator_resource_limits():
    from agent.tools import Calculator

    assert "Error" in Calculator().run("2 ** 999999999")
    assert Calculator().run("2 ** 10") == "1024"


def test_multiple_calls_budget(agent):
    agent.config["tools"] = {"max_calls": 1}
    calls = call()
    calls["tool_calls"].extend(call(id="second")["tool_calls"])
    agent.provider.complete = Mock(return_value=calls)
    agent.tools.call = Mock(return_value="5")
    with pytest.raises(RuntimeError, match="call limit"):
        agent.chat("two calculations")
    assert agent.tools.call.call_count == 1


def test_mcp_server_failure_cleans_up():
    async def run():
        with pytest.raises((ValueError, BaseExceptionGroup)):
            async with MCPTools(
                {
                    "servers": {
                        "demo": {
                            "command": sys.executable,
                            "args": ["examples/mcp_server.py"],
                        },
                        "bad": {"transport": "invalid"},
                    }
                }
            ):
                pass

    asyncio.run(run())


def test_custom_endpoint_does_not_receive_ambient_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    p = Provider({"model": "test", "base_url": "http://localhost:1234/v1"})
    try:
        assert p.client.api_key == "not-needed"
    finally:
        p.close()


def test_cli_profile_replaces_local_endpoint(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"""llm:
  backend: openai
  base_url: http://localhost:11434/v1
  api_key: local-only
profiles:
  cloud:
    backend: openai
    model: cloud-model
memory:
  db_path: {tmp_path}/memory.db
training:
  output_dir: {tmp_path}/training
""")
    fake = Mock()
    factory = Mock(return_value=fake)
    monkeypatch.setattr("agent.cli.PersonalAgent", factory)
    result = CliRunner().invoke(
        cli,
        ["--config", str(cfg), "chat", "--profile", "cloud", "--model", "override"],
        input="exit\n",
    )
    assert result.exit_code == 0, result.output
    assert factory.call_args.args[0]["llm"] == {
        "backend": "openai",
        "model": "override",
    }
    fake.close.assert_called_once()

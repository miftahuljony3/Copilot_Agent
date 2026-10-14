from __future__ import annotations

from fastapi.testclient import TestClient

from agent.core import PersonalAgent
from agent.web import create_app


def _config(tmp_path) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""agent:
  name: Test Agent
llm:
  backend: openai
  base_url: http://127.0.0.1:9/v1
  model: test
memory:
  db_path: {tmp_path.as_posix()}/memory.db
tools:
  enabled: false
""",
        encoding="utf-8",
    )
    return str(path)


def test_web_auth_chat_health_and_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WEB_TOKEN", "test-token-with-at-least-24-chars")
    monkeypatch.setenv("AGENT_CONFIG", _config(tmp_path))

    async def fake_chat(self, message):
        return f"reply: {message}"

    monkeypatch.setattr(PersonalAgent, "achat", fake_chat)
    with TestClient(create_app()) as client:
        assert client.get("/healthz").json() == {
            "status": "ok",
            "agent": "Test Agent",
            "model": "test",
        }
        assert client.get("/").status_code == 200
        assert client.post("/api/chat", json={"message": "hello"}).status_code == 401
        headers = {"Authorization": "Bearer test-token-with-at-least-24-chars"}
        response = client.post("/api/chat", json={"message": "hello"}, headers=headers)
        assert response.json() == {"reply": "reply: hello"}
        assert client.post("/api/reset", headers=headers).json() == {"status": "reset"}
        assert client.get("/api/history", headers=headers).json() == {"turns": []}


def test_web_requires_strong_token(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WEB_TOKEN", "short")
    monkeypatch.setenv("AGENT_CONFIG", _config(tmp_path))
    try:
        create_app()
    except RuntimeError as exc:
        assert "at least 24" in str(exc)
    else:
        raise AssertionError("short token was accepted")

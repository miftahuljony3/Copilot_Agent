"""Authenticated FastAPI interface for the personal agent."""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import secrets
from contextlib import asynccontextmanager

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .core import PersonalAgent
from .memory import Memory
from .tools import ToolRegistry

logger = logging.getLogger(__name__)
STATIC_DIR = pathlib.Path(__file__).with_name("static")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    reply: str


def _load_config() -> dict:
    path = pathlib.Path(os.getenv("AGENT_CONFIG", "config.yaml"))
    if not path.is_file():
        raise RuntimeError(f"Agent config file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise RuntimeError("Agent config must contain a YAML mapping")
    return config


def _storage(config: dict) -> Memory:
    memory_config = config.get("memory", {})
    return Memory(
        db_path=memory_config.get("db_path", "data/memory.db"),
        max_window=int(memory_config.get("max_window", 20)),
    )


def create_app() -> FastAPI:
    token = os.getenv("AGENT_WEB_TOKEN", "")
    if len(token) < 24:
        raise RuntimeError("AGENT_WEB_TOKEN must be set to at least 24 characters")
    config = _load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        memory = _storage(config)
        agent = PersonalAgent(config, memory, ToolRegistry.with_defaults())
        app.state.agent = agent
        app.state.memory = memory
        app.state.chat_lock = asyncio.Lock()
        try:
            yield
        finally:
            agent.close()
            memory.close()

    app = FastAPI(
        title="Personal Agent",
        version="0.2.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    def authorize(authorization: str | None = Header(default=None)) -> None:
        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(supplied, token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Valid bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/healthz")
    async def health() -> dict:
        return {
            "status": "ok",
            "agent": config.get("agent", {}).get("name", "Agent"),
            "model": config.get("llm", {}).get("model", "unknown"),
        }

    @app.post(
        "/api/chat", response_model=ChatResponse, dependencies=[Depends(authorize)]
    )
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        message = payload.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="Message cannot be blank")
        timeout = float(config.get("web", {}).get("request_timeout", 180))
        try:
            async with request.app.state.chat_lock:
                reply = await asyncio.wait_for(
                    request.app.state.agent.achat(message), timeout=timeout
                )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504, detail="Model request timed out"
            ) from exc
        except Exception as exc:
            logger.exception("Agent request failed (%s)", type(exc).__name__)
            raise HTTPException(
                status_code=502,
                detail="Agent request failed; check the service logs",
            ) from None
        return ChatResponse(reply=reply)

    @app.post("/api/reset", dependencies=[Depends(authorize)])
    async def reset(request: Request) -> dict:
        async with request.app.state.chat_lock:
            request.app.state.agent.reset()
        return {"status": "reset"}

    @app.get("/api/history", dependencies=[Depends(authorize)])
    async def history(
        request: Request, limit: int = Query(default=20, ge=1, le=100)
    ) -> dict:
        return {"turns": request.app.state.memory.get_history(limit=limit)}

    return app

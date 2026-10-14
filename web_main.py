"""Development entry point for the authenticated web interface."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "agent.web:create_app",
        factory=True,
        host=os.getenv("AGENT_HOST", "127.0.0.1"),
        port=int(os.getenv("AGENT_PORT", "8000")),
    )

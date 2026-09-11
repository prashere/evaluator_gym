"""CLI: uv run python -m evaluator_gym.sandbox"""

import os

import uvicorn

from evaluator_gym.sandbox.app import app

if __name__ == "__main__":
    host = os.getenv("SANDBOX_HOST", "0.0.0.0")
    port = int(os.getenv("SANDBOX_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)

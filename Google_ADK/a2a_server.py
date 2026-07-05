"""A2A server for the OpenAI-powered data-quality coordinator.

This wraps ``root_agent`` with ``to_a2a()`` so the coordinator can be
discovered and invoked by other A2A-compatible agents. The ``to_a2a``
helper auto-generates an Agent Card from the agent's name, description,
and tools.

Run directly:

    python -m Google_ADK.a2a_server --port 8001

Or with uvicorn (production-style):

    uvicorn Google_ADK.a2a_server:a2a_app --host 0.0.0.0 --port 8001

Then the Agent Card is available at:

    http://localhost:8001/.well-known/agent-card.json

And the JSON-RPC endpoint at:

    http://localhost:8001/  (POST with method tasks/send or message/send)
"""

from __future__ import annotations

import argparse
import os

from google.adk.a2a.utils.agent_to_a2a import to_a2a

from .agent import root_agent


# ``to_a2a`` returns an ASGI app that uvicorn can serve. We expose it at
# module scope so production deployments can run:
#   uvicorn Google_ADK.a2a_server:a2a_app --host 0.0.0.0 --port 8001
DEFAULT_PORT = int(os.getenv("A2A_PORT", "8001"))
DEFAULT_HOST = os.getenv("A2A_HOST", "0.0.0.0")

a2a_app = to_a2a(root_agent, port=DEFAULT_PORT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data-quality coordinator as an A2A server.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port (default: 8001)")
    args = parser.parse_args()

    import uvicorn

    print(f"[a2a_server] starting on http://{args.host}:{args.port}")
    print(f"[a2a_server] Agent Card: http://{args.host}:{args.port}/.well-known/agent-card.json")
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
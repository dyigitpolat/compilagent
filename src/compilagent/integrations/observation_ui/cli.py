"""Console entry point for `compilagent-observe`."""

from __future__ import annotations

import os

from .app import create_app


def main() -> None:
    import uvicorn  # lazy

    host = os.environ.get("COMPILAGENT_OBSERVE_HOST", "127.0.0.1")
    port = int(os.environ.get("COMPILAGENT_OBSERVE_PORT", "8765"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
